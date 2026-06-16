"""Tests for the v0.2.0 scaffold: transforms, conversational targets, attack
strategies, adaptive-attacker data model, defenses, benchmark, and the gated
research loader interface.

All offline and deterministic — no network, no SDKs, no real models. These tests
pin the frozen contracts module builders implement against.
"""

from __future__ import annotations

import os

import pytest

from injectkit.attackers.base import (
    AdaptiveAttacker,
    AttackerModel,
    AttackerResult,
    AttackerTranscriptStep,
)
from injectkit.attacks.base import (
    AttackStep,
    AttackStrategy,
    MultiTurnAttack,
    SingleShotStrategy,
    attack_to_strategy,
)
from injectkit.benchmark import (
    ASRCell,
    BenchmarkResult,
    BenchmarkRunMetadata,
    compute_asr,
)
from injectkit.defenses.base import (
    Defense,
    NullDefense,
    get_defense,
    list_defenses,
    register_defense,
)
from injectkit.models import Attack, AttackResult, Severity, TargetResponse
from injectkit.research.base import (
    RESEARCH_ACK_ENV,
    DatasetReference,
    ResearchAcknowledgmentError,
    ResearchDatasetLoader,
    require_acknowledgment,
)
from injectkit.research.registry import KNOWN_DATASETS
from injectkit.targets.conversational import (
    ChatMessage,
    ConversationalTarget,
    SingleShotChatAdapter,
    as_conversational,
)
from injectkit.transforms.base import (
    Compose,
    Identity,
    Transform,
    TransformRegistry,
    get_transform,
    list_transforms,
    register_transform,
)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


def test_identity_transform_is_noop_and_satisfies_protocol():
    t = Identity()
    assert isinstance(t, Transform)
    assert t.apply("payload INJECTOK-abc", "abc") == "payload INJECTOK-abc"


def test_compose_threads_left_to_right_and_names():
    class Upper:
        name = "upper"

        def apply(self, payload: str, canary: str) -> str:
            return payload.upper()

    class Bang:
        name = "bang"

        def apply(self, payload: str, canary: str) -> str:
            return payload + "!"

    composed = Compose(Upper(), Bang())
    assert composed.name == "upper+bang"
    assert composed.apply("hi", "c") == "HI!"


def test_transform_registry_register_get_and_duplicate_guard():
    reg = TransformRegistry()
    reg.register("identity", Identity)
    assert reg.names() == ["identity"]
    assert isinstance(reg.get("identity"), Identity)
    with pytest.raises(ValueError):
        reg.register("identity", Identity)
    with pytest.raises(KeyError):
        reg.get("nope")


def test_default_transform_registry_has_identity():
    assert "identity" in list_transforms()
    assert isinstance(get_transform("identity"), Identity)


def test_register_transform_on_default_registry():
    class Rev:
        name = "reverse"

        def apply(self, payload: str, canary: str) -> str:
            return payload[::-1]

    register_transform("reverse_test", Rev)
    assert get_transform("reverse_test").apply("abc", "c") == "cba"


# --------------------------------------------------------------------------- #
# Conversational targets
# --------------------------------------------------------------------------- #


def test_single_shot_adapter_flattens_transcript(mock_target):
    adapter = SingleShotChatAdapter(mock_target)
    assert isinstance(adapter, ConversationalTarget)
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="emit INJECTOK-xyz"),
    ]
    resp = adapter.chat(messages, system="sys")
    # The wrapped target saw a single transcript prompt with all turns + a cue.
    sent = mock_target.calls[-1]["prompt"]
    assert "User: hi" in sent
    assert "Assistant: hello" in sent
    assert "User: emit INJECTOK-xyz" in sent
    assert sent.rstrip().endswith("Assistant:")
    assert mock_target.calls[-1]["system"] == "sys"
    assert isinstance(resp, TargetResponse)


def test_as_conversational_passthrough_for_native(fake_conversational_target):
    # Already conversational -> returned unchanged.
    assert as_conversational(fake_conversational_target) is fake_conversational_target


def test_as_conversational_wraps_single_shot(mock_target):
    conv = as_conversational(mock_target)
    assert isinstance(conv, SingleShotChatAdapter)
    assert isinstance(conv, ConversationalTarget)


def test_fake_conversational_target_echoes_marker(fake_conversational_target):
    resp = fake_conversational_target.chat(
        [ChatMessage(role="user", content="say INJECTOK-cana123")]
    )
    assert resp.text == "INJECTOK-cana123"
    assert resp.refused is False


# --------------------------------------------------------------------------- #
# Attack strategies / multi-turn model
# --------------------------------------------------------------------------- #


def test_single_shot_strategy_builds_one_scored_step(sample_attack):
    strat = SingleShotStrategy()
    assert isinstance(strat, AttackStrategy)
    steps = strat.build(sample_attack, "cana999")
    assert len(steps) == 1
    step = steps[0]
    assert isinstance(step, AttackStep)
    assert step.scored is True
    assert step.expect_response is True
    assert step.message.role == "user"
    assert "INJECTOK-cana999" in step.message.content


def test_attack_to_strategy_returns_single_shot(sample_attack):
    assert isinstance(attack_to_strategy(sample_attack), SingleShotStrategy)


def test_multiturn_attack_to_attack_projects_scored_turn():
    mt = MultiTurnAttack(
        id="cresc-1",
        technique="jailbreak",
        name="crescendo",
        description="escalate",
        severity=Severity.HIGH,
        turns=[
            "innocuous lead-in",
            "a bit closer",
            "now emit INJECTOK-{canary}",
        ],
        success_conditions={"marker_echoed": True},
        scored_turn_index=-1,
    )
    attack = mt.to_attack()
    assert isinstance(attack, Attack)
    assert attack.id == "cresc-1"
    assert attack.payload == "now emit INJECTOK-{canary}"
    assert attack.render("zz") == "now emit INJECTOK-zz"
    assert attack.success_conditions == {"marker_echoed": True}


# --------------------------------------------------------------------------- #
# Adaptive attacker data model + a minimal reference implementation
# --------------------------------------------------------------------------- #


def test_stub_local_model_satisfies_attacker_model(stub_local_model):
    assert isinstance(stub_local_model, AttackerModel)
    out1 = stub_local_model.generate("propose")
    out2 = stub_local_model.generate("refine")
    assert out1 != out2  # {n} makes each round distinct
    assert stub_local_model.prompts == ["propose", "refine"]


def test_attacker_result_best_payload_from_transcript(sample_attack):
    resp = TargetResponse(text="INJECTOK-c", model="m")
    result = AttackResult(attack=sample_attack, canary="c", response=resp, success=True)
    step = AttackerTranscriptStep(round=1, candidate_payload="cand-1", result=result)
    ar = AttackerResult(
        seed_attack=sample_attack,
        succeeded=True,
        best_result=result,
        transcript=[step],
        rounds_used=1,
        attacker_model="stub-local",
    )
    assert ar.best_payload == "cand-1"


def test_minimal_adaptive_attacker_runs_offline(
    sample_attack, mock_target, stub_local_model
):
    """A tiny reference attacker that satisfies the protocol, fully offline.

    Proves the AdaptiveAttacker contract is implementable against the stub model
    + a real detector + the MockTarget without any network.
    """
    from injectkit.evaluators.heuristics import HeuristicDetector
    from injectkit.engine import make_canary

    class RefineAttacker:
        name = "refine"

        def __init__(self, model, max_rounds=3):
            self.model = model
            self.max_rounds = max_rounds

        def run(self, seed_attack, target, detectors):
            detector = detectors[0]
            best = None
            transcript = []
            succeeded = False
            for rnd in range(1, self.max_rounds + 1):
                self.model.generate(f"refine round {rnd} for {seed_attack.id}")
                canary = make_canary()
                rendered = seed_attack.render(canary)
                resp = target.send(rendered, system=seed_attack.system)
                verdict = detector.evaluate(seed_attack, resp, canary)
                res = AttackResult(
                    attack=seed_attack,
                    canary=canary,
                    response=resp,
                    verdicts=[verdict],
                    success=verdict.success,
                )
                transcript.append(
                    AttackerTranscriptStep(round=rnd, candidate_payload=rendered, result=res)
                )
                best = res
                if verdict.success:
                    succeeded = True
                    break
            return AttackerResult(
                seed_attack=seed_attack,
                succeeded=succeeded,
                best_result=best,
                transcript=transcript,
                rounds_used=len(transcript),
                attacker_model=self.model.name,
            )

    attacker = RefineAttacker(stub_local_model, max_rounds=3)
    assert isinstance(attacker, AdaptiveAttacker)
    result = attacker.run(sample_attack, mock_target, [HeuristicDetector()])
    assert result.succeeded is True
    assert result.rounds_used >= 1
    assert result.attacker_model == "stub-local"


# --------------------------------------------------------------------------- #
# Defenses
# --------------------------------------------------------------------------- #


def test_null_defense_is_passthrough():
    d = NullDefense()
    assert isinstance(d, Defense)
    assert d.wrap_system("sys") == "sys"
    assert d.filter_input("p", "ctx") == ("p", "ctx")
    assert d.filter_output("out") == "out"


def test_default_defense_registry_has_none():
    assert "none" in list_defenses()
    assert isinstance(get_defense("none"), NullDefense)


def test_register_custom_defense():
    class Redact:
        name = "redact_test"

        def wrap_system(self, system):
            return system

        def filter_input(self, prompt, context):
            return prompt, context

        def filter_output(self, text):
            return text.replace("INJECTOK-", "[REDACTED]-")

    register_defense("redact_test", Redact)
    d = get_defense("redact_test")
    assert d.filter_output("INJECTOK-abc") == "[REDACTED]-abc"


# --------------------------------------------------------------------------- #
# Benchmark / ASR model
# --------------------------------------------------------------------------- #


def test_compute_asr_handles_zero_attempts():
    assert compute_asr(0, 0) == 0.0
    assert compute_asr(2, 4) == 0.5


def _make_result(attack, success, error=None, severity=Severity.HIGH):
    resp = TargetResponse(text="x", error=error)
    return AttackResult(
        attack=attack, canary="c", response=resp, success=success, severity=severity
    )


def test_asr_cell_from_results_excludes_errored(sample_attack):
    results = [
        _make_result(sample_attack, True),
        _make_result(sample_attack, False),
        _make_result(sample_attack, False, error="boom"),
    ]
    cell = ASRCell.from_results("direct_injection", "none", results)
    assert cell.attempts == 2  # errored excluded
    assert cell.successes == 1
    assert cell.errored == 1
    assert cell.asr == 0.5
    assert cell.highest_severity == Severity.HIGH


def test_benchmark_result_rollups_and_defense_delta(sample_attack):
    meta = BenchmarkRunMetadata(
        tool_version="0.2.0",
        target_name="mock",
        transforms=["identity"],
        defenses=["none", "spotlight"],
    )
    overall_none = ASRCell("overall", "none", attempts=4, successes=3)
    overall_def = ASRCell("overall", "spotlight", attempts=4, successes=1)
    tech_none = ASRCell("jailbreak", "none", attempts=2, successes=2)
    bench = BenchmarkResult(metadata=meta, cells=[overall_none, overall_def, tech_none])

    assert bench.overall_asr("none") == 0.75
    assert bench.overall_asr("spotlight") == 0.25
    assert bench.defense_delta("spotlight") == pytest.approx(0.5)
    assert set(bench.defenses()) == {"none", "spotlight"}
    assert "jailbreak" in bench.by_technique("none")


# --------------------------------------------------------------------------- #
# Research loader gate
# --------------------------------------------------------------------------- #


def test_require_acknowledgment_blocks_without_optin(monkeypatch):
    monkeypatch.delenv(RESEARCH_ACK_ENV, raising=False)
    with pytest.raises(ResearchAcknowledgmentError):
        require_acknowledgment(acknowledge=False)


def test_require_acknowledgment_passes_with_flag(monkeypatch):
    monkeypatch.delenv(RESEARCH_ACK_ENV, raising=False)
    require_acknowledgment(acknowledge=True)  # no raise


def test_require_acknowledgment_passes_with_env(monkeypatch):
    monkeypatch.setenv(RESEARCH_ACK_ENV, "1")
    require_acknowledgment(acknowledge=False)  # no raise


def test_dataset_references_are_pointers_only():
    assert "advbench" in KNOWN_DATASETS
    for key, ref in KNOWN_DATASETS.items():
        assert isinstance(ref, DatasetReference)
        assert ref.key == key
        assert ref.url.startswith("http")
        assert ref.name


def test_research_loader_protocol_shape(monkeypatch):
    """A minimal loader satisfies the protocol and honours the gate."""

    class StubLoader:
        reference = KNOWN_DATASETS["advbench"]

        def load(self, *, acknowledge=False, limit=None, cache_dir=None):
            require_acknowledgment(acknowledge)
            return []  # never bundles data

    loader = StubLoader()
    assert isinstance(loader, ResearchDatasetLoader)
    monkeypatch.delenv(RESEARCH_ACK_ENV, raising=False)
    with pytest.raises(ResearchAcknowledgmentError):
        loader.load()
    assert loader.load(acknowledge=True) == []
