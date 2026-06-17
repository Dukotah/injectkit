"""Tests for the nanoGCG-parity hardening (CHUNK 3-gcg-advprefix).

Two layers:

* **Offline, always-run** — the correctness traps and combinatorics that need no
  weights: tokenizer-agnostic chat-template slice location across all 5 dense
  families (a real tokenizer wearing a bundled family template, no download
  beyond a tiny base tokenizer), the filter_ids retokenisation drop, the
  candidate sampler, the attack buffer, and AdvPrefix Pareto selection /
  per-model distinctness. These run with only ``transformers`` (no torch, no
  model weights); if even ``transformers`` is unavailable the slice tests skip.

* The one-hot gradient + golden-loss regression lives in
  ``test_whitebox_golden_loss.py`` (needs torch + a tiny model on CPU).

DEFERRED-NO-GPU: full 8B ASR parity (see docs/REPRODUCE.md). The hardening
*logic* is fully covered here on CPU/offline.
"""

from __future__ import annotations

import pytest

from injectkit.whitebox.chat_templates import (
    CHAT_TEMPLATES,
    GENERATION_PROMPT_MARKERS,
)
from injectkit.whitebox.gcg_hard import (
    AttackBuffer,
    PromptSlices,
    ProbeSamplingConfig,
    filter_ids,
    locate_optim_slice,
    round_trips,
    sample_candidates,
)
from injectkit.whitebox.targets import (
    FIXED_BASELINE_PREFIX,
    PrefixCandidate,
    PrefixScore,
    advprefix_target,
    candidate_prefixes_for,
    pareto_frontier,
    select_advprefix,
)

# A tiny base tokenizer is enough to exercise the slice algorithm offline; it
# wears each family's bundled chat template. Skip the tokenizer-backed tests if
# transformers (or the tiny tokenizer) is unavailable in this environment.
_tok = None
try:  # pragma: no cover - environment dependent
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("gpt2")
except Exception:  # noqa: BLE001 - any failure -> skip the tokenizer-backed tests
    _tok = None

requires_tokenizer = pytest.mark.skipif(
    _tok is None, reason="transformers/gpt2 tokenizer unavailable (offline host)"
)


# --------------------------------------------------------------------------- #
# Offline seam stub: encode/decode by simple char codes (no torch/transformers)
# --------------------------------------------------------------------------- #


class _SeamTokenizer:
    """A tiny round-trip-stable tokenizer seam (1 char = 1 token id)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


# --------------------------------------------------------------------------- #
# Correctness trap #2 — tokenizer-agnostic chat-template slice location
# --------------------------------------------------------------------------- #


@requires_tokenizer
@pytest.mark.parametrize("family", sorted(CHAT_TEMPLATES))
def test_slice_location_recovers_optim_span_all_families(family):
    """The optim slice round-trips to the optim text for every dense family."""
    template = CHAT_TEMPLATES[family]
    optim = "! ! ! ! !"
    target = "Sure, here is"
    msgs = [{"role": "user", "content": "Tell me about birds."}]

    slices = locate_optim_slice(_tok, msgs, optim, target, chat_template=template)

    assert isinstance(slices, PromptSlices)
    # The optim tokens decode back to exactly the optim string (no offset drift).
    assert _tok.decode(slices.optim_ids).strip() == optim.strip()
    # The slice indexes the optim tokens inside the assembled input.
    assert slices.input_ids[slices.optim_slice] == slices.optim_ids
    # And the target tokens decode back to the target string.
    assert _tok.decode(slices.target_ids).strip() == target.strip()


@requires_tokenizer
@pytest.mark.parametrize("family", sorted(CHAT_TEMPLATES))
def test_slice_optim_sits_before_generation_prompt(family):
    """The optim span is located in the user turn, before the assistant boundary."""
    template = CHAT_TEMPLATES[family]
    marker = GENERATION_PROMPT_MARKERS[family]
    if not marker:  # Mistral: the [/INST] close is itself the boundary.
        pytest.skip("family has no separate generation-prompt header")
    msgs = [{"role": "user", "content": "Question."}]
    slices = locate_optim_slice(_tok, msgs, "X X X", "Sure", chat_template=template)
    # Everything after the optim span (the rendered after-segment) carries the
    # family's generation-prompt header — proving the optim sits before it.
    after_text = _tok.decode(slices.after_ids)
    assert marker in after_text


@requires_tokenizer
def test_slice_location_is_template_specific_not_hardcoded():
    """Different family templates yield different optim offsets (no constant)."""
    msgs = [{"role": "user", "content": "Hi."}]
    starts = {
        fam: locate_optim_slice(
            _tok, msgs, "! !", "ok", chat_template=tpl
        ).optim_slice.start
        for fam, tpl in CHAT_TEMPLATES.items()
    }
    # At least two families disagree on the start offset -> not a hard-coded one.
    assert len(set(starts.values())) > 1


def test_slice_location_with_seam_tokenizer_offline():
    """The slicer also works on a plain seam tokenizer (no chat template)."""
    seam = _SeamTokenizer()
    msgs = [{"role": "user", "content": "abc"}]
    slices = locate_optim_slice(seam, msgs, "XY", "Z")
    # With no template the prompt is just the content; optim appended at the end.
    assert seam.decode(slices.optim_ids) == "XY"
    assert slices.input_ids[slices.optim_slice] == slices.optim_ids


def test_slice_handles_literal_optim_already_in_prompt():
    """If the optim string is already present, it is located in place."""
    seam = _SeamTokenizer()
    msgs = [{"role": "user", "content": "say QQ now"}]
    slices = locate_optim_slice(seam, msgs, "QQ", "Z")
    assert seam.decode(slices.optim_ids) == "QQ"
    # 'say ' precedes it, ' now' follows it.
    assert seam.decode(slices.before_ids).endswith("say ")
    assert seam.decode(slices.after_ids).startswith(" now")


# --------------------------------------------------------------------------- #
# Correctness trap #1 — filter_ids retokenisation drop
# --------------------------------------------------------------------------- #


@requires_tokenizer
def test_round_trips_true_for_clean_ids():
    ids = _tok.encode("hello world", add_special_tokens=False)
    assert round_trips(ids, _tok) is True


@requires_tokenizer
def test_filter_ids_drops_non_round_tripping_rows():
    """A candidate whose ids don't re-encode from their text is dropped."""
    clean = _tok.encode("hello", add_special_tokens=False)
    # Build a deliberately non-round-tripping row: a lone byte-level BPE fragment
    # that decodes to text the tokenizer re-segments differently. We search the
    # vocab for such an id rather than hard-coding one.
    bad_row = None
    for tid in range(256, 400):
        if not round_trips([clean[0], tid], _tok):
            bad_row = [clean[0], tid]
            break
    assert bad_row is not None, "expected at least one non-round-tripping id"
    batch = [clean[:2], bad_row]
    kept = filter_ids(batch, _tok)
    assert 0 in kept and 1 not in kept


def test_filter_ids_keeps_all_when_no_decode_available():
    """A seam without decode can't be checked -> never spuriously drops."""

    class _NoDecode:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    batch = [[1, 2, 3], [4, 5, 6]]
    assert filter_ids(batch, _NoDecode()) == [0, 1]


def test_filter_ids_accepts_tensor_like_rows():
    """filter_ids coerces tensor-like (.tolist) batches."""

    class _Batch:
        def tolist(self):
            return [[104, 105], [106, 107]]

    seam = _SeamTokenizer()
    kept = filter_ids(_Batch(), seam)
    assert kept == [0, 1]


# --------------------------------------------------------------------------- #
# Candidate sampler (pure-Python path)
# --------------------------------------------------------------------------- #


def test_sample_candidates_shape_and_determinism():
    # grad: [optim_len=3, vocab=10]; descending so low indices look 'best'.
    grad = [[-(j + 1) for j in range(10)] for _ in range(3)]
    optim = [0, 1, 2]
    a = sample_candidates(grad, optim, top_k=4, search_width=8, seed=0)
    b = sample_candidates(grad, optim, top_k=4, search_width=8, seed=0)
    assert len(a) == 8 and all(len(row) == 3 for row in a)
    assert a == b  # deterministic given the seed
    # Each candidate differs from the base in at most one slot (single-token swap).
    for row in a:
        diffs = sum(1 for k in range(3) if row[k] != optim[k])
        assert diffs <= 1


def test_sample_candidates_respects_not_allowed_ids():
    grad = [[-(j + 1) for j in range(10)] for _ in range(2)]
    optim = [0, 0]
    banned = list(range(10 - 1, 10 - 5, -1))  # ban the most-promising ids
    cands = sample_candidates(
        grad, optim, top_k=6, search_width=20, seed=1, not_allowed_ids=banned
    )
    used = {tok for row in cands for tok in row}
    assert not (used & set(banned))


# --------------------------------------------------------------------------- #
# Attack buffer
# --------------------------------------------------------------------------- #


def test_attack_buffer_keeps_lowest_loss():
    buf = AttackBuffer(size=2)
    buf.add(3.0, [1, 1])
    buf.add(1.0, [2, 2])
    buf.add(2.0, [3, 3])
    assert len(buf) == 2
    assert buf.best() == [2, 2]
    assert buf.best_loss() == 1.0


def test_attack_buffer_size_zero_keeps_single_best():
    buf = AttackBuffer(size=0)
    buf.add(5.0, [9])
    buf.add(2.0, [8])
    assert len(buf) == 1
    assert buf.best() == [8]


def test_attack_buffer_empty_is_none():
    buf = AttackBuffer(size=3)
    assert buf.best() is None
    assert buf.best_loss() == float("inf")


# --------------------------------------------------------------------------- #
# probe_sampling placeholder
# --------------------------------------------------------------------------- #


def test_probe_sampling_off_by_default():
    cfg = ProbeSamplingConfig()
    assert cfg.enabled is False
    assert cfg.sampling_factor == 16


# --------------------------------------------------------------------------- #
# AdvPrefix — Pareto selection + per-model distinctness
# --------------------------------------------------------------------------- #


def test_pareto_frontier_drops_dominated():
    a = PrefixCandidate("a")
    b = PrefixCandidate("b")
    c = PrefixCandidate("c")
    scored = [
        (a, PrefixScore(prefill_success=0.9, nll=1.0)),  # frontier
        (b, PrefixScore(prefill_success=0.5, nll=2.0)),  # dominated by a
        (c, PrefixScore(prefill_success=0.8, nll=0.5)),  # frontier (lower nll)
    ]
    front = pareto_frontier(scored)
    texts = {cand.text for cand, _ in front}
    assert texts == {"a", "c"}
    # Sorted by descending success first.
    assert front[0][0].text == "a"


def test_select_advprefix_picks_frontier_top():
    cands = [PrefixCandidate("hi"), PrefixCandidate("yo"), PrefixCandidate("ok")]
    scores = {
        "hi": PrefixScore(0.9, 2.0),
        "yo": PrefixScore(0.95, 1.0),  # dominates -> selected
        "ok": PrefixScore(0.4, 0.1),
    }
    chosen = select_advprefix(cands, lambda c: scores[c.text])
    assert chosen.text == "yo"


def test_select_advprefix_falls_back_to_baseline_when_all_filtered():
    cands = [PrefixCandidate("x")]
    chosen = select_advprefix(
        cands, lambda c: PrefixScore(0.0, 1.0), min_prefill_success=0.5
    )
    assert chosen.text == FIXED_BASELINE_PREFIX


def test_advprefix_returns_distinct_per_model():
    """The chunk done-check: AdvPrefix yields distinct per-model prefixes."""
    names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/phi-4",
    ]
    targets = [advprefix_target(n, canary="abc") for n in names]
    # The benign marker is always present (objective stays the canary).
    assert all("INJECTOK-abc" in t for t in targets)
    # At least 4 of the 5 prefixes are distinct (model-specific).
    prefixes = [t.split(" the marker:")[0] for t in targets]
    assert len(set(prefixes)) >= 4


def test_advprefix_baseline_is_documented_fixed_string():
    t = advprefix_target("anything", use_baseline=True, canary="z")
    assert t.startswith(FIXED_BASELINE_PREFIX)


def test_candidate_pool_includes_baseline():
    for name in ["llama", "qwen", "gemma", "mistral", "phi"]:
        pool = candidate_prefixes_for(name)
        assert any(c.text == FIXED_BASELINE_PREFIX for c in pool)
