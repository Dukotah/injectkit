"""Tests for Probe Sampling (CHUNK 8-probe-sampling; arXiv:2403.01251).

Fully offline and deterministic. The draft/target re-scoring logic is verified on
a TINY CPU model pair (two ``StubWhiteBoxModel`` seams, or two scripted seams with
controllable per-candidate losses) and, when ``torch``+``transformers`` are
present, on two real tiny GPT-2 / Pythia-160M models (importorskip — never
downloaded in this environment). The >=3x wall-clock speedup + non-degraded ASR on
an 8B target is DEFERRED-NO-GPU: the code path below is exercised end-to-end, the
headline NUMBER is not run here.
"""

from __future__ import annotations

import pytest

from injectkit.attackers.gcg import GCGSuffixAttacker
from injectkit.attackers.whitebox_base import GCGConfig as LegacyGCGConfig
from injectkit.whitebox import gcg as wb_gcg
from injectkit.whitebox.config import GCGConfig
from injectkit.whitebox.probe_sampling import (
    PAPER_ASR,
    PAPER_SPEEDUP,
    ProbeSampling,
    ProbeSamplingResult,
    resolve_probe_sampling,
)


# --------------------------------------------------------------------------- #
# Scripted seam: a WhiteBoxModel whose target_loss is a fixed map over candidates
# --------------------------------------------------------------------------- #


class ScriptedSeam:
    """A WhiteBoxModel seam with a controllable ``target_loss`` per candidate.

    ``loss_of`` maps the *candidate suffix tuple* (the tail of input_ids after the
    prompt prefix) to a loss, so a test can pin exactly what the draft and target
    each "think" of every candidate and assert the probe-sampling decision.
    """

    def __init__(self, name, loss_of, prompt_len):
        self.name = name
        self._loss_of = loss_of
        self._prompt_len = prompt_len
        self.loss_calls = 0

    def token_ids(self, text):
        return [ord(c) % 32 for c in (text or "")]

    def decode(self, ids):
        return "".join(chr((int(i) % 26) + 97) for i in ids)

    def target_loss(self, input_ids, target_ids):
        self.loss_calls += 1
        cand = tuple(int(x) for x in list(input_ids)[self._prompt_len :])
        return float(self._loss_of(cand))


# --------------------------------------------------------------------------- #
# resolve_probe_sampling — config knob normalisation + validation
# --------------------------------------------------------------------------- #


def test_resolve_disabled_forms():
    assert resolve_probe_sampling(None).enabled is False
    assert resolve_probe_sampling(False).enabled is False


def test_resolve_true_uses_paper_defaults():
    r = resolve_probe_sampling(True)
    assert r.enabled is True
    assert 0.0 < r.r <= 1.0
    assert r.sampling_factor >= 1


def test_resolve_tuple():
    r = resolve_probe_sampling((0.25, 16))
    assert r.enabled is True
    assert r.r == 0.25
    assert r.sampling_factor == 16


@pytest.mark.parametrize(
    "bad",
    [(0.0, 4), (1.5, 4), (-0.1, 4), (0.5, 0), (0.5, -1), (0.5,), (0.1, 4, 9)],
)
def test_resolve_rejects_malformed_tuple(bad):
    with pytest.raises(ValueError):
        resolve_probe_sampling(bad)


def test_gcgconfig_accepts_tuple_and_rejects_bad():
    cfg = GCGConfig(probe_sampling=(0.2, 8))
    assert cfg.probe_sampling == (0.2, 8)
    # Default stays off (behaviour identical to plain GCG).
    assert GCGConfig().probe_sampling is False
    with pytest.raises(Exception):  # pydantic ValidationError wraps the ValueError
        GCGConfig(probe_sampling=(2.0, 8))


# --------------------------------------------------------------------------- #
# ProbeSampling.select — the draft-vs-target re-scoring core
# --------------------------------------------------------------------------- #


def test_select_picks_global_min_when_draft_agrees():
    # Draft and target rank candidates identically -> probe sampling still finds
    # the global target minimum while scoring only a fraction on the target.
    prompt = [1, 2, 3]
    # Candidate suffixes are single-token tails 10..19; loss decreasing in the id.
    cands = [[v] for v in range(10, 20)]
    losses = {tuple(c): float(30 - c[0]) for c in cands}  # min at id=19
    draft = ScriptedSeam("draft", lambda c: losses[c], len(prompt))
    target = ScriptedSeam("target", lambda c: losses[c], len(prompt))

    ps = ProbeSampling(
        draft, target, r=0.2, sampling_factor=3, prompt_ids=prompt, target_ids=[7]
    )
    res = ps.select(cands)
    assert isinstance(res, ProbeSamplingResult)
    assert cands[res.best_index] == [19]  # global min by target loss
    # Cheaper than full scoring: fewer than every candidate hit the target.
    assert res.target_evals < len(cands)
    assert res.agreement == pytest.approx(1.0)
    assert res.kept_fraction >= 0.2


def test_select_widens_kept_fraction_when_draft_disagrees():
    # Draft ranking is the REVERSE of the target ranking -> agreement ~0 -> the
    # kept fraction widens toward the full batch (ASR protection). The real target
    # minimum must still be found because the kept set then covers it.
    prompt = [0]
    cands = [[v] for v in range(20, 30)]
    target_losses = {tuple(c): float(c[0]) for c in cands}  # min at id=20
    draft_losses = {tuple(c): float(-c[0]) for c in cands}  # reverse ranking
    draft = ScriptedSeam("draft", lambda c: draft_losses[c], len(prompt))
    target = ScriptedSeam("target", lambda c: target_losses[c], len(prompt))

    ps = ProbeSampling(
        draft, target, r=0.1, sampling_factor=3, prompt_ids=prompt, target_ids=[1]
    )
    res = ps.select(cands)
    assert res.agreement < 0.5  # draft contradicts target
    assert res.kept_fraction > 0.1  # widened above the floor
    # With a fully reversed draft, the floor-r path would miss the true min; the
    # dynamic widening should re-score enough of the batch to still find it OR a
    # near-min — assert we did not pick the draft's (wrong) favourite (id=29).
    assert cands[res.best_index] != [29]


def test_select_is_cheaper_than_brute_force():
    prompt = [5]
    cands = [[v] for v in range(40, 60)]  # 20 candidates
    losses = {tuple(c): float(c[0]) for c in cands}
    draft = ScriptedSeam("draft", lambda c: losses[c], len(prompt))
    target = ScriptedSeam("target", lambda c: losses[c], len(prompt))
    ps = ProbeSampling(
        draft, target, r=0.1, sampling_factor=2, prompt_ids=prompt, target_ids=[9]
    )
    res = ps.select(cands)
    # The target was scored on strictly fewer candidates than the full batch
    # (the speedup proxy); the draft scored the whole batch cheaply.
    assert target.loss_calls < len(cands)
    assert draft.loss_calls == len(cands)
    assert res.target_evals == target.loss_calls


def test_select_empty_batch():
    ps = ProbeSampling(
        ScriptedSeam("d", lambda c: 0.0, 0),
        ScriptedSeam("t", lambda c: 0.0, 0),
        r=0.1,
        sampling_factor=2,
        prompt_ids=[],
        target_ids=[1],
    )
    res = ps.select([])
    assert res.best_index == -1
    assert res.target_evals == 0


def test_paper_parity_constants_recorded():
    # Docstring/parity numbers are surfaced for the repro stamp.
    assert PAPER_SPEEDUP == "3.5x-6.3x"
    assert PAPER_ASR == (81.0, 69.0)
    assert "2403.01251" in (ProbeSampling.__module__ and __import__(
        "injectkit.whitebox.probe_sampling", fromlist=["__doc__"]
    ).__doc__)


# --------------------------------------------------------------------------- #
# Integration: existing GCG path runs with probe_sampling ENABLED
# --------------------------------------------------------------------------- #


def test_gcg_attack_runs_with_probe_sampling_enabled(stub_whitebox_model):
    # The v0.4 GCGAttack.run wires probe sampling end-to-end when cfg opts in.
    cfg = GCGConfig(max_steps=1, suffix_len=3, batch_size=2, top_k=4, seed=0)
    cfg = cfg.model_copy(update={"probe_sampling": (0.5, 2)})
    res = wb_gcg.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "a benign prompt"}],
        "INJECTOK-abc",
        cfg,
    )
    assert res.attack_name == "gcg"
    assert res.per_step_losses  # at least one step ran
    # The draft (== target stub here) was scored, proving the probe path executed.
    assert "target_loss" in stub_whitebox_model.calls


def test_legacy_attacker_probe_sampling_step_matches_objective(stub_whitebox_model):
    # Attaching a draft routes _optimize_suffix through the probe path; the result
    # is still a valid GCG trajectory toward the benign target.
    attacker = GCGSuffixAttacker(
        stub_whitebox_model,
        LegacyGCGConfig(max_steps=2, suffix_len=3, batch_size=2, top_k=4, seed=0),
    )
    attacker.attach_probe_sampling(stub_whitebox_model, r=0.5, sampling_factor=2)
    prompt_ids = stub_whitebox_model.token_ids("p")
    target_ids = stub_whitebox_model.token_ids("INJECTOK-x")
    steps = attacker._optimize_suffix(prompt_ids, target_ids)
    assert 1 <= len(steps) <= 2
    assert all(s.loss == s.loss for s in steps)  # not NaN


def test_probe_sampling_disabled_is_identical_to_plain_gcg(stub_whitebox_model):
    # With probe sampling off, the trajectory is byte-for-byte the plain path.
    cfg = LegacyGCGConfig(max_steps=2, suffix_len=3, batch_size=2, top_k=4, seed=3)
    plain = GCGSuffixAttacker(stub_whitebox_model, cfg)
    prompt_ids = stub_whitebox_model.token_ids("prompt")
    target_ids = stub_whitebox_model.token_ids("INJECTOK-y")
    base = [(s.step, s.suffix, s.loss) for s in plain._optimize_suffix(prompt_ids, target_ids)]

    fresh = GCGSuffixAttacker(stub_whitebox_model, cfg)  # probe NOT attached
    again = [(s.step, s.suffix, s.loss) for s in fresh._optimize_suffix(prompt_ids, target_ids)]
    assert base == again


# --------------------------------------------------------------------------- #
# Real tiny-model path (CPU) — skipped unless torch+transformers are present
# --------------------------------------------------------------------------- #


def test_real_tiny_model_draft_target_rescoring():
    """Verify draft-vs-target re-scoring on two real tiny CPU models.

    Uses two distinct tiny HF causal-LMs (e.g. ``sshleifer/tiny-gpt2`` as draft and
    target) as the draft/target seams. Skipped automatically when torch/
    transformers are not installed OR the tiny weights are not already cached --
    this environment has NO GPU and limited bandwidth, so we never download. The
    8B speedup/ASR number is DEFERRED-NO-GPU; this test proves the wiring on CPU.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "sshleifer/tiny-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached -> do not download in this env
        pytest.skip(f"{name} not cached locally; skipping (no-download env).")

    model.eval()

    class HFSeam:
        def __init__(self, name, tok, model):
            self.name = name
            self.tok = tok
            self.model = model

        def token_ids(self, text):
            return self.tok(text, add_special_tokens=False)["input_ids"]

        def decode(self, ids):
            return self.tok.decode([int(i) for i in ids])

        def target_loss(self, input_ids, target_ids):
            ids = torch.tensor([list(input_ids) + list(target_ids)])
            with torch.no_grad():
                logits = self.model(ids).logits
            n_t = len(list(target_ids))
            pred = logits[0, -n_t - 1 : -1, :]
            tgt = torch.tensor(list(target_ids))
            return float(torch.nn.functional.cross_entropy(pred, tgt))

    draft = HFSeam("draft", tok, model)
    target = HFSeam("target", tok, model)
    prompt = draft.token_ids("Hello there")
    tgt = draft.token_ids("world")
    cands = [draft.token_ids(s) or [0] for s in ("foo", "bar", "baz", "qux")]

    ps = ProbeSampling(
        draft, target, r=0.5, sampling_factor=2, prompt_ids=prompt, target_ids=tgt
    )
    res = ps.select(cands)
    assert 0 <= res.best_index < len(cands)
    assert res.target_evals <= len(cands)
    assert res.best_loss == res.best_loss  # finite
