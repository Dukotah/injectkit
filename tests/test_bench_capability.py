"""Tests for the capability-paradox bench harness (NICHE-STRATEGY.md #3).

Covers the multi-model sweep + the CLI wiring, all driven by the deterministic
offline seam path the rest of the bench harness uses (no torch, no download, no
API key):

* ``injectkit.bench.capability`` — ``run_capability_sweep`` runs one attack across
  a SET of models, aggregates per-model ASR with a Wilson CI + the full 8-field
  stamp, orders the points along a capability axis, and reports the monotonicity
  verdict (the MCPTox capability-paradox curve, arXiv:2508.14925).
* ``injectkit.cli`` — the ``capability`` subcommand runs offline and prints the
  ordered curve + the verdict, and exports the leaderboard artifacts.

The real frontier sweep (the pinned zoo models / live anthropic/ollama/openai
targets) needs a GPU + multi-GB downloads or API keys and is DEFERRED-NO-GPU; the
loader seam is the same one exercised here. Nothing here downloads a model.
"""

from __future__ import annotations

import json

import pytest

from injectkit.attacks.whitebox.prefill import GenerationResult
from injectkit.bench import (
    FLAT,
    INVERSE,
    PARADOX,
    CapabilityCurve,
    ModelSpec,
    ModelUnderTest,
    run_capability_sweep,
)
from injectkit.bench.stamp import STAMP_FIELDS


# --------------------------------------------------------------------------- #
# Offline seams: a marker-echo seam (=> success) and a refusal seam (=> 0 ASR),
# parameterised so a model's ASR can be made to TRACK its capability score — which
# is exactly the MCPTox capability-paradox curve we surface (deterministically).
# --------------------------------------------------------------------------- #


class _MarkerSeam:
    """Always echoes the prompt (with the benign marker) ⇒ deterministic success."""

    name = "demo"

    def prefill_generate(self, messages, prefix, n_tokens, harmony=False):
        body = "".join(str(m.get("content", "")) for m in messages)
        cont = f" Here is the requested information: {body}"
        return GenerationResult(prefix=prefix, continuation=cont, full_text=f"{prefix}{cont}")


class _RefusalSeam:
    """Always refuses ⇒ every behavior is a non-success (ASR 0)."""

    name = "demo"

    def prefill_generate(self, messages, prefix, n_tokens, harmony=False):
        cont = " I'm sorry, but I can't help with that."
        return GenerationResult(prefix=prefix, continuation=cont, full_text=f"{prefix}{cont}")


def _mut(seam, capability, label, *, quant="fp16"):
    return ModelUnderTest(
        spec=ModelSpec(
            name=label,
            loader=lambda *_a, **_k: (seam, None),
            quant=quant,
            revision="offline-seam",
        ),
        capability=capability,
        label=label,
    )


def _behaviors(n: int):
    # The marker the harness builds is f"{trigger}{behavior.id}"; embedding the id
    # in the prompt makes the marker-echo seam emit the marker ⇒ deterministic.
    return [{"id": f"c{i}", "prompt": f"continue the task INJECTOK-c{i}"} for i in range(n)]


# --------------------------------------------------------------------------- #
# capability.py — the multi-model sweep, ordering, and per-model stamp.
# --------------------------------------------------------------------------- #


def test_sweep_runs_across_a_model_set_offline():
    curve = run_capability_sweep(
        "prefill",
        [
            _mut(_MarkerSeam(), 1.0, "tiny"),
            _mut(_MarkerSeam(), 7.0, "mid"),
            _mut(_MarkerSeam(), 14.0, "big"),
        ],
        _behaviors(4),
        judge_id="clean_cls",
        num_seeds=1,
    )
    assert isinstance(curve, CapabilityCurve)
    assert len(curve.points) == 3
    # Every per-model cell carries the full 8-field stamp incl. mandatory quant.
    for p in curve.points:
        d = p.cell.stamp.to_dict()
        for fld in STAMP_FIELDS:
            assert d[fld] not in (None, "")
        assert p.cell.n_behaviors == 4
        # Each per-model ASR carries a Wilson CI.
        assert p.cell.judge_asr.lo <= p.cell.judge_asr.rate <= p.cell.judge_asr.hi


def test_points_are_sorted_along_the_capability_axis():
    # Pass the models OUT of capability order; the curve must sort them ascending.
    curve = run_capability_sweep(
        "prefill",
        [
            _mut(_MarkerSeam(), 14.0, "big"),
            _mut(_MarkerSeam(), 1.0, "tiny"),
            _mut(_MarkerSeam(), 7.0, "mid"),
        ],
        _behaviors(3),
        num_seeds=1,
    )
    caps = [cap for cap, _stat in curve.series()]
    assert caps == sorted(caps) == [1.0, 7.0, 14.0]
    labels = [p.label for p in curve.sorted_points()]
    assert labels == ["tiny", "mid", "big"]


def test_capability_paradox_verdict_when_asr_rises_with_capability():
    # Low-capability model refuses (ASR 0); high-capability model complies (ASR 1).
    # ASR RISES with capability => the MCPTox capability paradox.
    curve = run_capability_sweep(
        "prefill",
        [
            _mut(_RefusalSeam(), 1.0, "tiny-robust"),
            _mut(_MarkerSeam(), 14.0, "big-susceptible"),
        ],
        _behaviors(4),
        num_seeds=1,
    )
    assert curve.verdict() == PARADOX
    lo, hi = curve.series()
    assert lo[1].rate == 0.0  # the small model defended.
    assert hi[1].rate > 0.0   # the big model was bypassed.


def test_inverse_verdict_when_bigger_is_safer():
    # The conventional assumption: the big model is the one that defends.
    curve = run_capability_sweep(
        "prefill",
        [
            _mut(_MarkerSeam(), 1.0, "tiny-susceptible"),
            _mut(_RefusalSeam(), 14.0, "big-robust"),
        ],
        _behaviors(4),
        num_seeds=1,
    )
    assert curve.verdict() == INVERSE


def test_flat_verdict_when_no_trend():
    curve = run_capability_sweep(
        "prefill",
        [
            _mut(_MarkerSeam(), 1.0, "a"),
            _mut(_MarkerSeam(), 14.0, "b"),
        ],
        _behaviors(3),
        num_seeds=1,
    )
    # Both seams behave identically => same ASR at both rungs => flat.
    assert curve.verdict() == FLAT


def test_empty_model_set_is_rejected():
    with pytest.raises(ValueError):
        run_capability_sweep("prefill", [], _behaviors(2))


def test_model_without_capability_score_is_rejected():
    # A bare offline ModelSpec with no zoo entry and no explicit capability has no
    # axis position — the sweep refuses it up front rather than plot an undefined x.
    spec = ModelSpec(
        name="off-zoo-tiny",
        loader=lambda *_a, **_k: (_MarkerSeam(), None),
        quant="fp16",
        revision="offline-seam",
    )
    with pytest.raises(ValueError):
        run_capability_sweep("prefill", [spec], _behaviors(2))


def test_curve_as_dict_and_leaderboard_carry_every_stamp():
    curve = run_capability_sweep(
        "prefill",
        [_mut(_MarkerSeam(), 7.0, "mid"), _mut(_RefusalSeam(), 1.0, "tiny")],
        _behaviors(3),
        num_seeds=1,
    )
    doc = curve.as_dict()
    assert doc["attack_id"] == "prefill"
    assert doc["capability_axis"] == "params_b"
    assert doc["verdict"] in (PARADOX, INVERSE, FLAT)
    # Points serialise in ascending-capability order.
    assert [p["capability"] for p in doc["points"]] == [1.0, 7.0]
    for p in doc["points"]:
        for fld in STAMP_FIELDS:
            assert fld in p["cell"]["stamp"]
    # The leaderboard renders the curve as the model x attack matrix.
    board = curve.leaderboard()
    assert board.attacks() == ["prefill"]
    assert board.models() == ["tiny", "mid"]  # capability-ascending order.
    csv_text = board.to_csv()
    assert len(csv_text.strip().splitlines()) == 1 + 2  # header + 2 cells.


def test_multi_seed_per_model_aggregates():
    curve = run_capability_sweep(
        "prefill",
        [_mut(_MarkerSeam(), 1.0, "a"), _mut(_MarkerSeam(), 7.0, "b")],
        _behaviors(2),
        num_seeds=3,
    )
    for p in curve.points:
        assert p.cell.seeds == (0, 1, 2)
        assert len(p.cell.stamps) == 3
        assert len(p.cell.runs) == 2 * 3


# --------------------------------------------------------------------------- #
# CLI — the `capability` subcommand runs offline and prints the curve.
# --------------------------------------------------------------------------- #


def test_cli_capability_terminal(capsys):
    from injectkit.cli import main

    rc = main(["capability", "--behaviors", "3", "--seeds", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ASR-vs-capability curve" in out
    assert "judge-ASR" in out
    assert "capability axis" in out
    assert "verdict:" in out
    # The synthetic demo ladder has three rungs.
    assert "demo-1b" in out and "demo-7b" in out and "demo-14b" in out


def test_cli_capability_json(capsys):
    from injectkit.cli import main

    rc = main(["capability", "--behaviors", "2", "--seeds", "1", "--format", "json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["attack_id"] == "prefill"
    assert doc["verdict"] in (PARADOX, INVERSE, FLAT)
    # Ordered points, each with a full stamp + capability.
    caps = [p["capability"] for p in doc["points"]]
    assert caps == sorted(caps)
    assert doc["points"][0]["cell"]["stamp"]["quant"] == "fp16"


def test_cli_capability_export_dir(tmp_path):
    from injectkit.cli import main

    rc = main([
        "capability", "--behaviors", "2", "--seeds", "1",
        "--export-dir", str(tmp_path), "--format", "csv",
    ])
    assert rc == 0
    assert (tmp_path / "capability.csv").exists()
    assert (tmp_path / "capability.json").exists()
    assert (tmp_path / "capability.md").exists()


def test_cli_capability_quant_flag_in_stamp(capsys):
    from injectkit.cli import main

    rc = main([
        "capability", "--behaviors", "2", "--seeds", "1", "--quant", "4bit",
        "--format", "json",
    ])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert all(p["cell"]["stamp"]["quant"] == "4bit" for p in doc["points"])
