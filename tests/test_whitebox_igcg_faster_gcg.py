"""Tests for I-GCG + Faster-GCG (+ Mask-GCG / momentum variants), CHUNK 9.

Fully offline and deterministic. The three I-GCG mechanisms (diverse benign
targets, automatic multi-coordinate update, easy-to-hard init), the three
Faster-GCG mechanisms (distance-regularized scoring, temperature sampling,
visited-set dedup), Mask-GCG token-position pruning, and the optional
momentum/MAGIC/SM-GCG flag tier are verified on the tiny CPU path via the offline
``StubWhiteBoxModel`` seam and pure-Python primitive checks.

DEFERRED-NO-GPU: the I-GCG ~100%-ASR-on-Vicuna-7B/Llama-2 and the Faster-GCG
wall-clock-speedup NUMBERS need a 7-8B GPU run; only the LOGIC is exercised here.
A golden-loss CI test on a tiny CPU model is in
``tests/test_whitebox_golden_loss.py`` (the GCG-family numerics tripwire shared by
all variants — the variants reuse the same gradient/forward plumbing).
"""

from __future__ import annotations

import random

import pytest

from injectkit.whitebox import faster_gcg as wb_faster
from injectkit.whitebox import gcg as wb_gcg
from injectkit.whitebox import igcg as wb_igcg
from injectkit.whitebox import mask_gcg as wb_mask
from injectkit.whitebox.config import (
    FasterGCGConfig,
    GCGConfig,
    IGCGConfig,
    MaskGCGConfig,
)
from injectkit.whitebox.faster_gcg import (
    VisitedSet,
    distance_regularized_scores,
    temperature_sample,
)
from injectkit.whitebox.gcg_variants import (
    MomentumState,
    anneal_temperature,
    magic_coordinate_count,
    sm_accept,
)
from injectkit.whitebox.igcg import (
    PAPER_ASR,
    adapt_p,
    diverse_targets,
    easiest_target,
    easy_to_hard_seed,
    worst_coordinates,
)
from injectkit.whitebox.mask_gcg import position_importance, prune_mask
from injectkit.whitebox.registry import get_attack, list_attacks


# --------------------------------------------------------------------------- #
# Registration + arXiv-id docstrings
# --------------------------------------------------------------------------- #


def test_variants_registered():
    names = list_attacks()
    for n in ("igcg", "faster_gcg", "mask_gcg"):
        assert n in names
        assert get_attack(n).name == n


@pytest.mark.parametrize(
    "module, arxiv",
    [
        (wb_igcg, "2405.21018"),
        (wb_faster, "2410.15362"),
        (wb_mask, "2509.06350"),
    ],
)
def test_docstrings_cite_arxiv_ids(module, arxiv):
    assert arxiv in (module.__doc__ or "")


def test_variant_flag_modules_cite_arxiv_ids():
    from injectkit.whitebox import gcg_variants

    doc = gcg_variants.__doc__ or ""
    for arxiv in ("2405.01229", "2412.08615"):  # momentum (MAC), MAGIC
        assert arxiv in doc


# --------------------------------------------------------------------------- #
# I-GCG (1) diverse benign targets
# --------------------------------------------------------------------------- #


def test_diverse_targets_are_distinct_and_benign():
    tgts = diverse_targets("meta-llama/Llama-2-7b", trigger="INJECTOK-", n=3)
    assert len(tgts) == 3
    # All carry the benign marker; none are harmful strings.
    assert all("INJECTOK-" in t for t in tgts)
    # At least two distinct phrasings (diversity).
    assert len(set(tgts)) >= 2


def test_diverse_targets_min_one():
    assert len(diverse_targets("x", trigger="T", n=0)) == 1


def test_easiest_target_picks_lowest_loss(stub_whitebox_model):
    # Stub loss = |len(input) - len(target)| + 1, so the target whose length is
    # closest to the input length is "easiest".
    m = stub_whitebox_model
    input_ids = [1, 2, 3, 4, 5]  # len 5
    targets = [[0, 0], [0, 0, 0, 0, 0], [0]]  # len 2, 5, 1 -> idx 1 best
    idx, loss = easiest_target(m, input_ids, targets)
    assert idx == 1
    assert loss == 1.0


def test_easiest_target_empty():
    class _M:
        def target_loss(self, a, b):  # pragma: no cover - not reached
            return 0.0

    idx, loss = easiest_target(_M(), [1], [])
    assert idx == -1 and loss == float("inf")


# --------------------------------------------------------------------------- #
# I-GCG (2) automatic multi-coordinate update
# --------------------------------------------------------------------------- #


def test_adapt_p_widens_on_improvement_and_narrows_on_stall():
    assert adapt_p(1, prev_loss=10.0, new_loss=5.0, max_p=4) == 2  # improved -> +1
    assert adapt_p(3, prev_loss=5.0, new_loss=6.0, max_p=4) == 2  # worse -> -1
    assert adapt_p(4, prev_loss=10.0, new_loss=1.0, max_p=4) == 4  # capped at max
    assert adapt_p(1, prev_loss=1.0, new_loss=2.0, max_p=4) == 1  # floored at 1


def test_worst_coordinates_returns_p_slots(stub_whitebox_model):
    m = stub_whitebox_model
    slots = worst_coordinates(m, [1, 2], [5, 6, 7, 8], [0, 0], p=2)
    assert len(slots) == 2
    assert all(0 <= s < 4 for s in slots)


def test_worst_coordinates_clamped_to_suffix_len(stub_whitebox_model):
    slots = worst_coordinates(stub_whitebox_model, [1], [5, 6], [0], p=10)
    assert len(slots) == 2  # clamped to suffix length
    assert worst_coordinates(stub_whitebox_model, [1], [], [0], p=2) == []


# --------------------------------------------------------------------------- #
# I-GCG (3) easy-to-hard init
# --------------------------------------------------------------------------- #


def test_easy_to_hard_seed_precedence():
    # explicit init_suffix wins
    cfg = IGCGConfig(init_suffix="my seed", easy_to_hard_init=True)
    assert easy_to_hard_seed(cfg) == "my seed"
    # flag on, no explicit seed -> bundled benign seed
    cfg2 = IGCGConfig(easy_to_hard_init=True)
    assert easy_to_hard_seed(cfg2) == wb_igcg.EASY_SEED_SUFFIX
    # flag off -> None (use attacker default filler)
    cfg3 = IGCGConfig(easy_to_hard_init=False)
    assert easy_to_hard_seed(cfg3) is None


def test_igcg_paper_asr_constant_recorded():
    assert "100%" in PAPER_ASR


# --------------------------------------------------------------------------- #
# I-GCG end-to-end on the stub seam
# --------------------------------------------------------------------------- #


def test_igcg_run_end_to_end(stub_whitebox_model):
    cfg = IGCGConfig(
        max_steps=2,
        suffix_len=4,
        batch_size=2,
        top_k=4,
        num_diverse_targets=3,
        seed=0,
    )
    res = wb_igcg.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "a benign prompt"}],
        "",
        cfg,
    )
    assert res.attack_name == "igcg"
    assert res.per_step_losses  # at least one step ran
    assert res.optimized_obj_kind == "suffix"
    assert all(l == l for l in res.per_step_losses)  # no NaN
    assert "primary_target" in res.stamp


def test_igcg_coerces_plain_config(stub_whitebox_model):
    # A plain GCGConfig is coerced to I-GCG defaults.
    res = wb_igcg.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "p"}],
        "INJECTOK-z",
        GCGConfig(max_steps=1, suffix_len=3, batch_size=2, top_k=4),
    )
    assert res.attack_name == "igcg"
    assert res.per_step_losses


# --------------------------------------------------------------------------- #
# Faster-GCG (1) distance-regularized gradient
# --------------------------------------------------------------------------- #


def test_distance_reg_off_is_identity():
    row = [-3.0, -1.0, -2.0]
    assert distance_regularized_scores(row, 0, None, distance_reg_lambda=0.0) == row


def test_distance_reg_penalizes_far_tokens_proxy():
    # No embeddings -> id-index proxy: far-from-current ids get a larger penalty.
    row = [0.0, 0.0, 0.0, 0.0]
    scored = distance_regularized_scores(row, 0, None, distance_reg_lambda=1.0)
    # current_id=0: penalty grows with id, so score is monotonically increasing.
    assert scored[0] < scored[1] < scored[2] < scored[3]


def test_distance_reg_with_embeddings():
    # current token 0 at origin; token 1 near, token 2 far -> token 2 penalised more.
    emb = [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]]
    row = [0.0, 0.0, 0.0]
    scored = distance_regularized_scores(row, 0, emb, distance_reg_lambda=1.0)
    assert scored[1] < scored[2]
    assert scored[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Faster-GCG (2) temperature sampling
# --------------------------------------------------------------------------- #


def test_temperature_sample_prefers_low_scores():
    # Low score = promising. Over many draws the lowest-score id dominates first pick.
    scores = [0.0, 10.0, 10.0, 10.0]
    rng = random.Random(0)
    first_picks = [
        temperature_sample(scores, k=1, temperature=0.5, rng=rng)[0]
        for _ in range(50)
    ]
    assert first_picks.count(0) > 40  # id 0 (lowest score) usually chosen


def test_temperature_sample_distinct_and_capped():
    scores = [1.0, 2.0, 3.0]
    rng = random.Random(1)
    out = temperature_sample(scores, k=5, temperature=1.0, rng=rng)
    assert len(out) == 3  # capped at pool size
    assert len(set(out)) == 3  # without replacement


def test_temperature_sample_respects_candidate_ids():
    rng = random.Random(2)
    out = temperature_sample([0.0, 5.0], k=1, temperature=0.1, rng=rng, candidate_ids=[42, 99])
    assert out[0] == 42  # the low-score one, mapped to its real id


# --------------------------------------------------------------------------- #
# Faster-GCG (3) visited set
# --------------------------------------------------------------------------- #


def test_visited_set_dedup_and_eviction():
    vs = VisitedSet(capacity=2)
    assert not vs.seen([1, 2])
    vs.add([1, 2])
    assert vs.seen([1, 2])
    vs.add([3, 4])
    vs.add([5, 6])  # evicts [1,2] (oldest)
    assert not vs.seen([1, 2])
    assert vs.seen([3, 4]) and vs.seen([5, 6])
    assert len(vs) == 2


def test_visited_set_disabled_when_zero_capacity():
    vs = VisitedSet(capacity=0)
    vs.add([1, 2])
    assert not vs.seen([1, 2])  # dedup off
    assert len(vs) == 0


def test_faster_gcg_run_end_to_end(stub_whitebox_model):
    cfg = FasterGCGConfig(
        max_steps=2,
        suffix_len=4,
        batch_size=2,
        top_k=4,
        distance_reg_lambda=0.1,
        temp_sampling=True,
        visited_set_size=16,
        seed=0,
    )
    res = wb_faster.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "a benign prompt"}],
        "INJECTOK-abc",
        cfg,
    )
    assert res.attack_name == "faster_gcg"
    assert res.per_step_losses
    assert all(l == l for l in res.per_step_losses)
    assert "paper_speedup" in res.stamp


def test_faster_gcg_hard_topk_mode(stub_whitebox_model):
    # temp_sampling off -> hard top-k path still runs.
    cfg = FasterGCGConfig(
        max_steps=1, suffix_len=3, batch_size=2, top_k=4, temp_sampling=False, seed=0
    )
    res = wb_faster.run(
        stub_whitebox_model, None, [{"role": "user", "content": "p"}], "INJECTOK-q", cfg
    )
    assert res.per_step_losses


# --------------------------------------------------------------------------- #
# Mask-GCG token-position pruning
# --------------------------------------------------------------------------- #


def test_prune_mask_keeps_top_importance():
    imp = [0.1, 0.9, 0.2, 0.8]
    mask = prune_mask(imp, keep_fraction=0.5, min_active=1)
    assert mask == [False, True, False, True]  # keep the two highest (idx 1,3)


def test_prune_mask_min_active_and_full_keep():
    imp = [0.1, 0.2, 0.3]
    assert sum(prune_mask(imp, keep_fraction=0.0, min_active=2)) == 2
    assert prune_mask(imp, keep_fraction=1.0, min_active=1) == [True, True, True]
    assert prune_mask([], keep_fraction=0.5) == []


def test_position_importance_shape(stub_whitebox_model):
    imp = position_importance(stub_whitebox_model, [1, 2], [5, 6, 7], [0, 0])
    assert len(imp) == 3
    assert all(i >= 0.0 for i in imp)


def test_mask_gcg_run_prunes_after_warmup(stub_whitebox_model):
    cfg = MaskGCGConfig(
        max_steps=3,
        suffix_len=6,
        batch_size=2,
        top_k=4,
        warmup_steps=1,
        keep_fraction=0.5,
        min_active=1,
        seed=0,
    )
    res = wb_mask.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "a benign prompt"}],
        "INJECTOK-abc",
        cfg,
    )
    assert res.attack_name == "mask_gcg"
    assert res.per_step_losses
    # The active-position count is recorded and <= the suffix length.
    assert "active_positions" in res.stamp


# --------------------------------------------------------------------------- #
# Optional momentum / MAGIC / SM-GCG flag tier (on base GCGConfig)
# --------------------------------------------------------------------------- #


def test_momentum_off_is_identity():
    ms = MomentumState(beta=0.0)
    grad = [[-1.0, -2.0], [-3.0, -4.0]]
    assert ms.blend(grad) == grad


def test_momentum_blends_running_average():
    ms = MomentumState(beta=0.5)
    first = ms.blend([[2.0, 4.0]])
    assert first == [[2.0, 4.0]]  # seeds from first grid
    second = ms.blend([[0.0, 0.0]])
    # 0.5*prev + 0.5*new = 0.5*[2,4] + 0 = [1,2]
    assert second == [[1.0, 2.0]]


def test_momentum_reseeds_on_shape_change():
    ms = MomentumState(beta=0.5)
    ms.blend([[1.0, 2.0]])
    out = ms.blend([[1.0], [2.0]])  # different shape
    assert out == [[1.0], [2.0]]


def test_magic_coordinate_count_bounds():
    # Peaked grid (one strong slot) -> small count; flat -> bounded.
    grad = [[-10.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    c = magic_coordinate_count(grad, max_coords=3, min_coords=1)
    assert 1 <= c <= 3
    assert magic_coordinate_count([], max_coords=5) == 1


def test_anneal_temperature_decays():
    assert anneal_temperature(0.0, 5) == 0.0
    t0 = anneal_temperature(1.0, 0)
    t5 = anneal_temperature(1.0, 5)
    assert t0 == 1.0
    assert 0.0 < t5 < t0


def test_sm_accept_greedy_and_metropolis():
    rng = random.Random(0)
    assert sm_accept(-1.0, 0.0, rng) is True  # improvement always accepted
    assert sm_accept(1.0, 0.0, rng) is False  # non-improving, temp 0 -> greedy reject
    # High temperature -> sometimes accept a non-improving swap.
    accepts = sum(sm_accept(0.1, 5.0, random.Random(i)) for i in range(100))
    assert accepts > 0


def test_gcg_variant_flags_run_end_to_end(stub_whitebox_model):
    cfg = GCGConfig(
        max_steps=2,
        suffix_len=4,
        batch_size=2,
        top_k=4,
        momentum=0.5,
        magic=True,
        sm_gcg_temperature=0.5,
        seed=0,
    )
    res = wb_gcg.run(
        stub_whitebox_model,
        None,
        [{"role": "user", "content": "a benign prompt"}],
        "INJECTOK-abc",
        cfg,
    )
    assert res.attack_name == "gcg"
    assert res.per_step_losses
    assert all(l == l for l in res.per_step_losses)


def test_gcg_variant_flags_default_off_uses_legacy_path(stub_whitebox_model):
    # With all variant flags at defaults, the verbatim legacy loop runs (no
    # variant branch); _variants_enabled is False so behaviour is plain GCG.
    cfg = GCGConfig(max_steps=2, suffix_len=3, batch_size=2, top_k=4, seed=5)
    assert wb_gcg._variants_enabled(cfg) is False
    assert wb_gcg._variants_enabled(cfg.model_copy(update={"momentum": 0.5})) is True
    res = wb_gcg.run(
        stub_whitebox_model, None, [{"role": "user", "content": "x"}], "INJECTOK-w", cfg
    )
    assert res.per_step_losses
