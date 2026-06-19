"""CHUNK 11-reinforce-uja — REINFORCE-GCG (arXiv:2502.17254) + UJA (arXiv:2510.02999).

Verifies the LOGIC/WIRING of the two objective-frontier, judge-in-the-loop attacks
entirely on the offline ``StubWhiteBoxModel`` seam + a small CPU judge (``substring``),
with NO torch and NO model download:

* both resolve through the v0.4 attack registry and run end-to-end on the tiny CPU
  model with a small CPU in-loop judge;
* the §6.10.1 circularity firewall (``opt_judge_id != eval_judge_id``) is asserted
  for BOTH attacks — at config construction, at the shared ``resolve_opt_judge``
  seam, and reflected in the result stamp;
* the REINFORCE surrogate + UJA objective math is correct;
* the module docstrings cite the two arXiv ids and document judge separation.

The 24GB-VRAM-fit and the >2x / 85-86%-ASR-on-Llama-3 parity numbers are
DEFERRED-NO-GPU (recorded as ``PAPER_ASR`` / ``PAPER_VRAM`` constants, not run).
"""

from __future__ import annotations

import re

import pytest

from injectkit.whitebox import (
    REINFORCEGCGConfig,
    UJAConfig,
    assert_judge_separation,
    get_attack,
    list_attacks,
    reinforce_loss,
    resolve_opt_judge,
    sample_completions,
    uja_loss,
)
from injectkit.whitebox import reinforce_gcg as reinforce_gcg_mod
from injectkit.whitebox import uja as uja_mod
from injectkit.whitebox.objective_judge import judge_scores


# --------------------------------------------------------------------------- #
# Registry wiring                                                             #
# --------------------------------------------------------------------------- #
def test_both_attacks_registered():
    names = set(list_attacks())
    assert {"reinforce_gcg", "uja"} <= names


@pytest.mark.parametrize(
    "name, cls_name",
    [("reinforce_gcg", "REINFORCEGCGAttack"), ("uja", "UJAAttack")],
)
def test_resolve_through_registry(name, cls_name):
    attack = get_attack(name)
    assert type(attack).__name__ == cls_name
    assert attack.name == name
    assert attack.supported_arch == {"dense"}


# --------------------------------------------------------------------------- #
# §6.10.1 circularity firewall — opt_judge_id != eval_judge_id (the DR)       #
# --------------------------------------------------------------------------- #
def test_reinforce_config_enforces_judge_separation():
    cfg = REINFORCEGCGConfig()
    assert cfg.judge_id != cfg.eval_judge_id
    with pytest.raises(ValueError, match="MUST differ"):
        REINFORCEGCGConfig(judge_id="clean_cls", eval_judge_id="clean_cls")


def test_uja_config_enforces_judge_separation():
    cfg = UJAConfig()
    assert cfg.judge_id != cfg.eval_judge_id
    with pytest.raises(ValueError, match="MUST differ"):
        UJAConfig(judge_id="substring", eval_judge_id="substring")


def test_assert_judge_separation_seam():
    # Distinct ids pass.
    assert_judge_separation("substring", "clean_cls") is None
    # Identical ids raise.
    with pytest.raises(ValueError, match="circularity firewall"):
        assert_judge_separation("clean_cls", "clean_cls")
    # Empty ids raise (the firewall requires two real judges).
    with pytest.raises(ValueError):
        assert_judge_separation("", "clean_cls")


def test_resolve_opt_judge_checks_firewall_then_loads():
    # Loads the OPT judge after passing the firewall.
    judge = resolve_opt_judge("substring", "clean_cls")
    assert judge.judge_id == "substring"
    # Refuses to even load if opt == eval.
    with pytest.raises(ValueError, match="MUST differ"):
        resolve_opt_judge("clean_cls", "clean_cls")


# --------------------------------------------------------------------------- #
# Objective math — REINFORCE surrogate + UJA loss                             #
# --------------------------------------------------------------------------- #
def test_reinforce_loss_zero_when_no_advantage():
    # All rewards equal => advantages 0 => surrogate degenerates to 0 (ties broken
    # by the proposal gradient).
    assert reinforce_loss([0.5, 0.5, 0.5], [1.0, 2.0, 3.0]) == 0.0
    # Empty completion set => 0.
    assert reinforce_loss([], []) == 0.0


def test_reinforce_loss_rewards_high_reward_low_nll():
    # A high-reward completion (above baseline) with a LOW nll should drive the
    # surrogate negative (lower == better for the greedy search).
    loss = reinforce_loss([1.0, 0.0], [1.0, 2.0])
    assert loss < 0.0
    # Default baseline is the mean reward; an explicit baseline shifts it.
    assert reinforce_loss([1.0, 0.0], [1.0, 2.0], baseline=0.0) != loss


def test_uja_loss_is_negative_mean_reward():
    assert uja_loss([1.0, 0.0]) == pytest.approx(-0.5)
    assert uja_loss([1.0, 1.0]) == pytest.approx(-1.0)
    assert uja_loss([]) == 0.0
    # More unsafe (higher judge score) => lower (better) loss.
    assert uja_loss([0.9, 0.9]) < uja_loss([0.1, 0.1])


# --------------------------------------------------------------------------- #
# Judge-in-the-loop seam — sample_completions + judge_scores                  #
# --------------------------------------------------------------------------- #
def test_sample_completions_offline_count_and_determinism(stub_whitebox_model):
    import random

    comps = sample_completions(
        stub_whitebox_model, "hello", num_samples=3, sample_temp=1.0,
        rng=random.Random(0),
    )
    assert len(comps) == 3
    assert all(isinstance(c, str) for c in comps)
    # Deterministic for a fixed seed.
    comps2 = sample_completions(
        stub_whitebox_model, "hello", num_samples=3, sample_temp=1.0,
        rng=random.Random(0),
    )
    assert comps == comps2


def test_judge_scores_returns_score_per_completion():
    judge = resolve_opt_judge("substring", "clean_cls")
    scores = judge_scores(
        ["a perfectly fine and lengthy answer here", "I cannot help with that"],
        judge, canary="CANARY", target="", trigger="INJECTOK-",
    )
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    # The refusal phrase scores 0 (the substring judge flagged it).
    assert scores[1] == 0.0


# --------------------------------------------------------------------------- #
# End-to-end runs on the tiny CPU stub model + small CPU judge                #
# --------------------------------------------------------------------------- #
def _messages():
    return [{"role": "user", "content": "hello world"}]


def test_reinforce_gcg_runs_on_stub(stub_whitebox_model):
    cfg = REINFORCEGCGConfig(max_steps=1, suffix_len=3, num_samples=2)
    result = reinforce_gcg_mod.run(stub_whitebox_model, None, _messages(), "", cfg)
    assert result.attack_name == "reinforce_gcg"
    assert result.optimized_obj_kind == "suffix"
    assert result.queries == 1
    assert len(result.per_step_losses) == 1
    # The firewall ids round-trip into the stamp, distinct.
    assert result.stamp["opt_judge_id"] != result.stamp["eval_judge_id"]
    assert result.stamp["objective"] == "reinforce"
    # The DEFERRED-NO-GPU parity numbers are recorded, never faked into a metric.
    assert "Llama-3-8B" in result.stamp["paper_asr"]
    assert "24GB" in result.stamp["paper_vram"]


def test_uja_runs_on_stub(stub_whitebox_model):
    cfg = UJAConfig(max_steps=1, suffix_len=3, num_samples=2)
    result = uja_mod.run(stub_whitebox_model, None, _messages(), "", cfg)
    assert result.attack_name == "uja"
    assert result.optimized_obj_kind == "suffix"
    assert result.queries == 1
    assert result.stamp["opt_judge_id"] != result.stamp["eval_judge_id"]
    assert result.stamp["objective"] == "untargeted-judge"
    # UJA's loss curve is the negated judge score (in [-1, 0]); never an NLL.
    assert all(loss <= 0.0 for loss in result.per_step_losses)


def test_attacks_accept_defense_kwarg(stub_whitebox_model):
    class _Defense:
        name = "stub_defense"

    for mod, cfg in (
        (reinforce_gcg_mod, REINFORCEGCGConfig(max_steps=1, suffix_len=2, num_samples=1)),
        (uja_mod, UJAConfig(max_steps=1, suffix_len=2, num_samples=1)),
    ):
        result = mod.run(
            stub_whitebox_model, None, _messages(), "", cfg, defense=_Defense()
        )
        assert result.defense_id == "stub_defense"


def test_coerces_plain_attackconfig(stub_whitebox_model):
    # Handing a non-typed config still yields a valid (firewall-satisfying) run.
    from injectkit.whitebox import GCGConfig

    r = reinforce_gcg_mod.run(
        stub_whitebox_model, None, _messages(), "", GCGConfig(max_steps=1, suffix_len=2)
    )
    assert r.stamp["opt_judge_id"] != r.stamp["eval_judge_id"]

    u = uja_mod.run(
        stub_whitebox_model, None, _messages(), "", GCGConfig(max_steps=1, suffix_len=2)
    )
    assert u.stamp["opt_judge_id"] != u.stamp["eval_judge_id"]


# --------------------------------------------------------------------------- #
# Honesty / provenance — docstrings cite the arXiv ids + judge separation     #
# --------------------------------------------------------------------------- #
def test_reinforce_docstring_cites_arxiv_and_judge_separation():
    doc = reinforce_gcg_mod.__doc__ or ""
    assert "2502.17254" in doc
    assert re.search(r"judge", doc, re.IGNORECASE)
    # The firewall / separation requirement is documented.
    assert "eval_judge_id" in doc or "circularity" in doc.lower()


def test_uja_docstring_cites_arxiv_and_judge_separation():
    doc = uja_mod.__doc__ or ""
    assert "2510.02999" in doc
    assert "eval_judge_id" in doc or "circularity" in doc.lower()


def test_paper_constants_recorded_for_deferral():
    assert "2502.17254" in reinforce_gcg_mod.PAPER_ASR
    assert "24GB" in reinforce_gcg_mod.PAPER_VRAM
    assert "2510.02999" in uja_mod.PAPER_ASR
