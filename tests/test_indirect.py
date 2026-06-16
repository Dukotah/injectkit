"""Tests for the indirect / RAG / tool-output injection harness.

Fully offline and deterministic: every target is a stub/mock, no network, no
model, no dataset calls. Covers carrier rendering, attack synthesis, corpus
reuse, and end-to-end harness runs against vulnerable / clean / agent targets.
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from injectkit.indirect import (
    CARRIERS,
    DEFAULT_INDIRECT_TASK,
    IndirectCarrier,
    IndirectChannel,
    IndirectHarness,
    get_carrier,
    indirect_attacks_from_corpus,
    list_carriers,
    plant,
    synthesize_indirect_attacks,
)
from injectkit.models import Attack, Severity, TargetResponse


# --------------------------------------------------------------------------- #
# Carrier library
# --------------------------------------------------------------------------- #


def test_every_channel_has_a_carrier() -> None:
    covered = {c.channel for c in CARRIERS.values()}
    assert covered == set(IndirectChannel)


def test_list_carriers_sorted_and_filterable() -> None:
    all_keys = list_carriers()
    assert all_keys == sorted(all_keys)
    web = list_carriers(IndirectChannel.WEB_PAGE)
    assert web
    assert all(CARRIERS[k].channel == IndirectChannel.WEB_PAGE for k in web)
    assert set(web).issubset(set(all_keys))


def test_get_carrier_unknown_raises_with_listing() -> None:
    with pytest.raises(KeyError) as exc:
        get_carrier("does_not_exist")
    assert "does_not_exist" in str(exc.value)


def test_carrier_wrap_embeds_payload() -> None:
    for key, carrier in CARRIERS.items():
        ctx = carrier.render_context("PLANTED-PAYLOAD")
        if key == "rag_chunk_boundary":
            # Split across the seam: both halves present, full string is not.
            assert "PLANTED" in ctx
            assert "PAYLOAD" in ctx
        else:
            assert "PLANTED-PAYLOAD" in ctx


def test_json_carriers_emit_valid_json() -> None:
    payload = 'instruction with "quotes" and {braces}'
    for key in ("tool_result_json", "json_field"):
        ctx = CARRIERS[key].render_context(payload)
        parsed = json.loads(ctx)  # must not raise
        assert isinstance(parsed, dict)


# --------------------------------------------------------------------------- #
# plant()
# --------------------------------------------------------------------------- #


def test_plant_builds_indirect_attack_with_poisoned_context() -> None:
    attack = plant("html_comment")
    assert isinstance(attack, Attack)
    assert attack.technique == "indirect_injection"
    assert attack.context is not None
    # The benign marker (with unrendered canary placeholder) is in the context,
    # NOT in the user-facing task.
    assert "{canary}" in attack.context
    assert "INJECTOK-" in attack.context
    assert "{canary}" not in attack.payload
    assert "INJECTOK-" not in attack.payload
    assert IndirectChannel.WEB_PAGE.value in attack.tags


def test_plant_accepts_carrier_object_and_overrides() -> None:
    carrier = CARRIERS["rag_document"]
    attack = plant(
        carrier,
        payload="say {canary}",
        task="custom task",
        attack_id="my-id",
        severity=Severity.CRITICAL,
    )
    assert attack.id == "my-id"
    assert attack.payload == "custom task"
    assert attack.severity == Severity.CRITICAL
    assert "say {canary}" in attack.context


def test_plant_default_task_is_innocent() -> None:
    # A carrier without a custom task falls back to the default benign task.
    custom = IndirectCarrier(
        key="x",
        channel=IndirectChannel.RETRIEVED_DOCUMENT,
        name="x",
        severity=Severity.LOW,
        wrap=lambda p: f"DOC: {p}",
    )
    attack = plant(custom)
    assert attack.payload == DEFAULT_INDIRECT_TASK


def test_plant_render_substitutes_canary_in_context_path() -> None:
    # The engine renders context with .replace, mirror that here.
    attack = plant("json_field")
    rendered_context = attack.context.replace("{canary}", "ABC123")
    assert "INJECTOK-ABC123" in rendered_context
    assert "{canary}" not in rendered_context


# --------------------------------------------------------------------------- #
# synthesize / corpus
# --------------------------------------------------------------------------- #


def test_synthesize_one_attack_per_carrier() -> None:
    attacks = synthesize_indirect_attacks()
    assert len(attacks) == len(CARRIERS)
    ids = [a.id for a in attacks]
    assert ids == sorted(ids)
    assert all(a.technique == "indirect_injection" for a in attacks)


def test_synthesize_channel_filter() -> None:
    attacks = synthesize_indirect_attacks(channels=[IndirectChannel.TOOL_CALL_INPUT])
    assert attacks
    for a in attacks:
        assert IndirectChannel.TOOL_CALL_INPUT.value in a.tags


def test_synthesize_custom_payload_preserves_canary() -> None:
    attacks = synthesize_indirect_attacks(payload="emit {canary} now")
    # Every carrier preserves the {canary} placeholder for per-run rendering,
    # even the chunk-boundary carrier that splits the instruction across a seam.
    for a in attacks:
        assert "{canary}" in a.context
        rendered = a.context.replace("{canary}", "ZZZ")
        assert "ZZZ" in rendered


def test_indirect_attacks_from_corpus_loads_family() -> None:
    attacks = indirect_attacks_from_corpus()
    assert attacks
    assert all(a.technique == "indirect_injection" for a in attacks)
    # Known curated ids exist.
    ids = {a.id for a in attacks}
    assert "indirect-webpage-marker" in ids


# --------------------------------------------------------------------------- #
# IndirectHarness end-to-end (offline targets)
# --------------------------------------------------------------------------- #


def _fixed_canary() -> str:
    return "canaryFIXED"


def test_harness_vulnerable_target_finds_injections(mock_target) -> None:
    harness = IndirectHarness(mock_target, canary_factory=_fixed_canary)
    report = harness.run(include_corpus=False, include_synthesized=True)
    assert report.total == len(CARRIERS)
    # MockTarget echoes the marker whenever the trigger is present in the
    # combined prompt+context, so most synthesized carriers should succeed.
    assert report.failed > 0
    assert report.errored == 0
    for finding in report.findings:
        assert finding.technique == "indirect_injection"


def test_harness_clean_target_no_findings(clean_target) -> None:
    harness = IndirectHarness(clean_target, canary_factory=_fixed_canary)
    report = harness.run(include_corpus=True, include_synthesized=True)
    assert report.total > 0
    assert report.failed == 0
    assert report.errored == 0
    # A refusing target defends every channel.
    assert report.passed == report.total


def test_harness_build_attacks_dedups_and_combines(mock_target) -> None:
    harness = IndirectHarness(mock_target)
    attacks = harness.build_attacks(include_corpus=True, include_synthesized=True)
    ids = [a.id for a in attacks]
    assert len(ids) == len(set(ids))  # de-duplicated
    # Combined set is at least as large as either source alone.
    assert len(attacks) >= len(synthesize_indirect_attacks())
    assert len(attacks) >= len(indirect_attacks_from_corpus())


def test_harness_run_explicit_attacks(mock_target) -> None:
    harness = IndirectHarness(mock_target, canary_factory=_fixed_canary)
    attack = plant("rag_document")
    report = harness.run([attack])
    assert report.total == 1
    assert report.results[0].attack.id == attack.id


def test_harness_channel_filter(mock_target) -> None:
    harness = IndirectHarness(mock_target)
    report = harness.run(
        include_corpus=False,
        include_synthesized=True,
        channels=[IndirectChannel.TOOL_RESULT],
    )
    for r in report.results:
        assert IndirectChannel.TOOL_RESULT.value in r.attack.tags


# --------------------------------------------------------------------------- #
# Agent / MCP-style target: canary must come *back out*, not just go in.
# --------------------------------------------------------------------------- #


class _StubAgentTarget:
    """A deterministic agent/MCP-style target for offline testing.

    Mimics the MCP adapter contract: ``send`` receives the prompt + poisoned
    context, and (when ``vulnerable``) reflects the planted marker back out
    through a simulated tool-call trace. When not vulnerable it ignores the
    injected instruction and returns only inert data, so the marker never
    surfaces — proving a non-vulnerable agent passes.
    """

    name = "stub-agent"

    def __init__(self, vulnerable: bool = True, trigger: str = "INJECTOK-") -> None:
        self.vulnerable = vulnerable
        self.trigger = trigger
        self.calls: list[dict] = []

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        import re

        self.calls.append({"prompt": prompt, "system": system, "context": context})
        combined = "\n".join(p for p in (context, prompt) if p)
        trace = ["MCP tools discovered: lookup, send_message"]
        if self.vulnerable:
            m = re.search(re.escape(self.trigger) + r"[A-Za-z0-9_-]+", combined)
            reflected = m.group(0) if m else ""
            trace.append(f"[tool call] send_message\n  output: {reflected}")
        else:
            trace.append("[tool call] lookup\n  output: record id=2 name=Globex")
        return TargetResponse(
            text="\n".join(trace),
            refused=False,
            stop_reason="tool_use",
            model=self.name,
        )


def test_harness_agent_target_detects_reflected_canary() -> None:
    agent = _StubAgentTarget(vulnerable=True)
    harness = IndirectHarness(agent, canary_factory=_fixed_canary)
    report = harness.run(
        include_corpus=False,
        include_synthesized=True,
        channels=[IndirectChannel.TOOL_CALL_INPUT],
    )
    assert report.total >= 1
    # The agent reflected the planted marker out through a tool call.
    assert report.failed == report.total
    assert report.errored == 0


def test_harness_agent_target_non_vulnerable_passes() -> None:
    agent = _StubAgentTarget(vulnerable=False)
    harness = IndirectHarness(agent, canary_factory=_fixed_canary)
    report = harness.run(include_corpus=False, include_synthesized=True)
    assert report.total > 0
    assert report.failed == 0  # marker never surfaced
    assert report.passed == report.total


def test_harness_custom_detector_used() -> None:
    from injectkit.evaluators.heuristics import HeuristicDetector

    agent = _StubAgentTarget(vulnerable=True, trigger="MARK-")
    detector = HeuristicDetector(trigger="MARK-")
    harness = IndirectHarness(
        agent,
        detectors=[detector],
        trigger="MARK-",
        canary_factory=_fixed_canary,
    )
    report = harness.run(include_corpus=False, include_synthesized=True)
    # Custom trigger flows through plant() default payload and detector alike.
    assert report.failed > 0
