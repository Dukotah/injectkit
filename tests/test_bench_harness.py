"""Tests for the v0.4 bench harness (CHUNK 7-bench-harness).

Covers the three modules and the CLI wiring:

* ``injectkit.bench.stamp`` — the 8-field reproducibility stamp, quant MANDATORY,
  the order-independent corpus hash, and the cross-seed reproduce key.
* ``injectkit.bench.harness`` — one-command single-cell ASR (attack x model x
  behavior_set x num_seeds x judge_id) on a TINY OFFLINE seam (no torch, no
  download), the Wilson CI, substring-ASR >= judge-ASR, and the "two seeded runs
  reproduce within CI" done-check.
* ``injectkit.bench.leaderboard`` — the model x attack matrix and its CSV + JSON +
  Markdown exports (and the combined ``export`` to disk).
* ``injectkit.cli`` — the ``attack`` subcommand runs offline and prints the full
  stamp.

The 8B / fp16-vs-4bit anchor cells need a GPU + a multi-GB download and are
DEFERRED-NO-GPU; the harness path is proven here on an offline seam + (in the
generation-runner suite) a tiny CPU model. Nothing here downloads a model.
"""

from __future__ import annotations

import json

import pytest

from injectkit import __version__
from injectkit.attacks.whitebox.prefill import GenerationResult
from injectkit.bench import (
    Leaderboard,
    ModelSpec,
    ReproStamp,
    StampError,
    build_stamp,
    corpus_hash,
    run_cell,
    runs_reproduce,
    stamps_reproduce,
    wilson_interval,
)
from injectkit.bench.stamp import STAMP_FIELDS


# --------------------------------------------------------------------------- #
# Offline seams: emit (or withhold) the benign marker so judges score deterministically.
# --------------------------------------------------------------------------- #


class _MarkerSeam:
    """Prefill seam that echoes the prompt (containing the benign marker) ⇒ success."""

    name = "demo"

    def prefill_generate(self, messages, prefix, n_tokens, harmony=False):
        body = "".join(str(m.get("content", "")) for m in messages)
        cont = f" Here is the requested information: {body}"
        return GenerationResult(prefix=prefix, continuation=cont, full_text=f"{prefix}{cont}")


class _RefusalSeam:
    """Prefill seam that always refuses ⇒ every behavior is a non-success."""

    name = "demo"

    def prefill_generate(self, messages, prefix, n_tokens, harmony=False):
        cont = " I'm sorry, but I can't help with that."
        return GenerationResult(prefix=prefix, continuation=cont, full_text=f"{prefix}{cont}")


def _seam_spec(seam, *, quant="fp16", revision="offline-seam"):
    return ModelSpec(
        name="demo",
        loader=lambda *_a, **_k: (seam, None),
        quant=quant,
        revision=revision,
    )


def _behaviors(n: int):
    # The marker the harness builds is f"{trigger}{behavior.id}"; embedding the id in
    # the prompt means the marker-echo seam emits the marker ⇒ deterministic success.
    return [
        {"id": f"c{i}", "prompt": f"continue the task INJECTOK-c{i}"} for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# stamp.py — 8 fields, quant mandatory, corpus hash.
# --------------------------------------------------------------------------- #


def test_stamp_has_all_eight_fields():
    assert STAMP_FIELDS == (
        "version",
        "corpus_hash",
        "model_revision",
        "seed",
        "quant",
        "judge_id",
        "attack_id",
        "backend",
    )
    stamp = build_stamp(
        behaviors=["a", "b"],
        model_revision="org/m@" + "f" * 40,
        seed=0,
        quant="fp16",
        judge_id="clean_cls",
        attack_id="prefill",
        backend="hf",
    )
    d = stamp.to_dict()
    for fld in STAMP_FIELDS:
        assert fld in d and d[fld] not in (None, "")
    assert d["version"] == __version__


def test_stamp_quant_is_mandatory_no_default():
    # quant is keyword-only with NO default on build(): omitting it is a TypeError.
    with pytest.raises(TypeError):
        ReproStamp.build(  # type: ignore[call-arg]
            corpus_hash="x",
            model_revision="m",
            seed=0,
            judge_id="clean_cls",
            attack_id="prefill",
            backend="hf",
        )


def test_stamp_rejects_blank_and_invalid_quant():
    with pytest.raises(StampError):
        ReproStamp.build(
            corpus_hash="x", model_revision="m", seed=0, quant="",
            judge_id="j", attack_id="a", backend="hf",
        )
    with pytest.raises(StampError):
        ReproStamp.build(
            corpus_hash="x", model_revision="m", seed=0, quant="fp32",
            judge_id="j", attack_id="a", backend="hf",
        )


def test_stamp_rejects_missing_mandatory_field():
    with pytest.raises(StampError):
        ReproStamp.build(
            corpus_hash="x", model_revision="", seed=0, quant="fp16",
            judge_id="j", attack_id="a", backend="hf",
        )


def test_stamp_rejects_invalid_backend():
    with pytest.raises(StampError):
        ReproStamp.build(
            corpus_hash="x", model_revision="m", seed=0, quant="fp16",
            judge_id="j", attack_id="a", backend="sglang",
        )


def test_stamp_seed_zero_is_present():
    # seed=0 is a legitimate seed, not "blank".
    stamp = build_stamp(
        behaviors=["a"], model_revision="m", seed=0, quant="4bit",
        judge_id="j", attack_id="a", backend="hf",
    )
    assert stamp.seed == 0
    assert stamp.quant == "4bit"


def test_corpus_hash_is_order_independent_and_content_sensitive():
    h1 = corpus_hash(["a", "b", "c"])
    h2 = corpus_hash(["c", "a", "b"])  # reorder -> same experiment.
    assert h1 == h2
    h3 = corpus_hash(["a", "b", "c", "d"])  # add a behavior -> different.
    assert h3 != h1
    assert len(h1) == 64  # SHA-256 hex.


def test_corpus_hash_handles_mappings():
    h = corpus_hash([{"id": "x", "prompt": "p"}, {"prompt": "q", "id": "y"}])
    assert len(h) == 64


def test_stamps_reproduce_ignores_seed():
    a = build_stamp(behaviors=["a"], model_revision="m", seed=0, quant="fp16",
                    judge_id="j", attack_id="att", backend="hf")
    b = build_stamp(behaviors=["a"], model_revision="m", seed=99, quant="fp16",
                    judge_id="j", attack_id="att", backend="hf")
    c = build_stamp(behaviors=["a"], model_revision="m", seed=0, quant="4bit",
                    judge_id="j", attack_id="att", backend="hf")
    assert stamps_reproduce(a, b)  # only the seed differs.
    assert not stamps_reproduce(a, c)  # quant differs -> different cell.


# --------------------------------------------------------------------------- #
# harness.py — Wilson CI.
# --------------------------------------------------------------------------- #


def test_wilson_interval_basic_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # n == 0 is well-defined (degenerate, not a crash).
    assert wilson_interval(0, 0) == (0.0, 0.0)
    # All successes: hi is 1.0-ish, lo well below 1 (Wilson, not Wald).
    lo2, hi2 = wilson_interval(8, 8)
    assert hi2 == pytest.approx(1.0, abs=1e-9) or hi2 <= 1.0
    assert lo2 < 1.0


# --------------------------------------------------------------------------- #
# harness.py — one-command single-cell ASR with a full stamp (the done-check).
# --------------------------------------------------------------------------- #


def test_single_cell_asr_with_full_stamp_runs_offline():
    cell = run_cell(
        "prefill",
        _seam_spec(_MarkerSeam()),
        _behaviors(4),
        judge_id="clean_cls",
        num_seeds=1,
        backend="hf",
    )
    # Aggregated three signals, each with a CI.
    assert cell.n_behaviors == 4
    assert 0.0 <= cell.judge_asr.rate <= 1.0
    assert cell.judge_asr.lo <= cell.judge_asr.rate <= cell.judge_asr.hi
    assert cell.substring_asr.lo <= cell.substring_asr.hi
    assert cell.strongreject_mean.mean is not None
    # The marker-echo seam should drive a high judge-ASR (deterministic success).
    assert cell.judge_asr.successes >= 1
    # Full 8-field stamp present, quant mandatory.
    d = cell.stamp.to_dict()
    for fld in STAMP_FIELDS:
        assert d[fld] not in (None, "")
    assert d["quant"] == "fp16"
    assert d["attack_id"] == "prefill"
    assert d["judge_id"] == "clean_cls"
    # Budget metadata.
    assert cell.avg_queries >= 1.0
    assert cell.wall_clock_s >= 0.0


def test_substring_asr_at_least_judge_asr():
    # The loosest signal must flag >= the calibrated judge (ROADMAP §8).
    cell = run_cell(
        "prefill", _seam_spec(_MarkerSeam()), _behaviors(5),
        judge_id="clean_cls", num_seeds=1,
    )
    assert cell.substring_asr.rate >= cell.judge_asr.rate - 1e-9


def test_refusal_seam_gives_zero_asr():
    cell = run_cell(
        "prefill", _seam_spec(_RefusalSeam()), _behaviors(3),
        judge_id="clean_cls", num_seeds=1,
    )
    assert cell.judge_asr.successes == 0
    assert cell.judge_asr.rate == 0.0
    # Wilson lo is 0 at 0/n; hi is a positive upper bound (not 0).
    assert cell.judge_asr.lo == 0.0
    assert cell.judge_asr.hi > 0.0


# --------------------------------------------------------------------------- #
# harness.py — two seeded runs reproduce within CI (the headline done-check).
# --------------------------------------------------------------------------- #


def test_two_seeded_runs_reproduce_within_ci():
    behaviors = _behaviors(4)
    a = run_cell("prefill", _seam_spec(_MarkerSeam()), behaviors,
                 judge_id="clean_cls", seeds=[0])
    b = run_cell("prefill", _seam_spec(_MarkerSeam()), behaviors,
                 judge_id="clean_cls", seeds=[1])
    # Same cell modulo seed...
    assert stamps_reproduce(a.stamp, b.stamp)
    assert a.stamp.seed == 0 and b.stamp.seed == 1
    # ...and the ASR reproduces within the CI.
    assert runs_reproduce(a, b)


def test_multi_seed_cell_builds_one_stamp_per_seed():
    cell = run_cell("prefill", _seam_spec(_MarkerSeam()), _behaviors(2),
                    judge_id="clean_cls", num_seeds=3)
    assert cell.seeds == (0, 1, 2)
    assert len(cell.stamps) == 3
    # Per-seed stamps differ only in seed.
    assert stamps_reproduce(cell.stamps[0], cell.stamps[1])
    assert {s.seed for s in cell.stamps} == {0, 1, 2}
    # n runs == behaviors * seeds.
    assert len(cell.runs) == 2 * 3


def test_different_quant_is_a_different_cell():
    behaviors = _behaviors(3)
    a = run_cell("prefill", _seam_spec(_MarkerSeam(), quant="fp16"), behaviors,
                 judge_id="clean_cls", seeds=[0])
    b = run_cell("prefill", _seam_spec(_MarkerSeam(), quant="4bit"), behaviors,
                 judge_id="clean_cls", seeds=[0])
    assert a.quant == "fp16" and b.quant == "4bit"
    assert not stamps_reproduce(a.stamp, b.stamp)  # quant column distinguishes them.


# --------------------------------------------------------------------------- #
# leaderboard.py — model x attack matrix + CSV/JSON/Markdown export.
# --------------------------------------------------------------------------- #


def _two_cell_board():
    board = Leaderboard(title="test board")
    board.add(run_cell("prefill", _seam_spec(_MarkerSeam()), _behaviors(3),
                       judge_id="clean_cls", seeds=[0]))
    board.add(run_cell("prefill", _seam_spec(_RefusalSeam()), _behaviors(3),
                       judge_id="strongreject", seeds=[0]))
    return board


def test_leaderboard_markdown_export():
    board = _two_cell_board()
    md = board.to_markdown()
    assert "# test board" in md
    assert "substring-ASR" in md and "judge-ASR" in md and "StrongREJECT-mean" in md
    assert "| model \\ attack | prefill |" in md
    # Quant column + corpus hash provenance present.
    assert "quant" in md
    assert board.cells[0].stamp.corpus_hash[:12] in md


def test_leaderboard_csv_export_has_quant_and_stamp_columns():
    board = _two_cell_board()
    csv_text = board.to_csv()
    header = csv_text.splitlines()[0].split(",")
    # Mandatory quant column + all 8 stamp columns.
    assert "quant" in header
    for fld in STAMP_FIELDS:
        assert f"stamp_{fld}" in header
    # One data row per cell.
    assert len(csv_text.strip().splitlines()) == 1 + len(board.cells)


def test_leaderboard_json_roundtrips():
    board = _two_cell_board()
    doc = json.loads(board.to_json())
    assert doc["title"] == "test board"
    assert doc["primary_columns"] == ["substring_asr", "judge_asr", "strongreject_mean"]
    assert len(doc["cells"]) == 2
    # Every cell serialises its full stamp.
    for cell in doc["cells"]:
        for fld in STAMP_FIELDS:
            assert fld in cell["stamp"]


def test_leaderboard_export_writes_three_files(tmp_path):
    board = _two_cell_board()
    paths = board.export(tmp_path, stem="lb")
    assert set(paths) == {"csv", "json", "markdown"}
    for p in paths.values():
        assert p.exists() and p.read_text(encoding="utf-8")
    # The written JSON parses.
    json.loads(paths["json"].read_text(encoding="utf-8"))


def test_leaderboard_matrix_axes_and_lookup():
    board = _two_cell_board()
    assert board.models() == ["demo"]
    assert board.attacks() == ["prefill"]
    assert board.cell("demo", "prefill") is not None
    assert board.cell("nope", "prefill") is None


# --------------------------------------------------------------------------- #
# CLI — the `attack` subcommand runs offline and prints the full stamp.
# --------------------------------------------------------------------------- #


def test_cli_attack_terminal(capsys):
    from injectkit.cli import main

    rc = main(["attack", "--behaviors", "3", "--seeds", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    # The three signals + all 8 stamp fields are printed.
    assert "substring-ASR" in out and "judge-ASR" in out and "StrongREJECT-mean" in out
    for fld in STAMP_FIELDS:
        assert fld in out
    assert "quant" in out


def test_cli_attack_json(capsys):
    from injectkit.cli import main

    rc = main(["attack", "--behaviors", "2", "--seeds", "1", "--format", "json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["cells"][0]["stamp"]["quant"] == "fp16"
    assert doc["cells"][0]["attack_id"] == "prefill"


def test_cli_attack_export_dir(tmp_path, capsys):
    from injectkit.cli import main

    rc = main([
        "attack", "--behaviors", "2", "--seeds", "1",
        "--export-dir", str(tmp_path), "--format", "csv",
    ])
    assert rc == 0
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.md").exists()


def test_cli_attack_quant_flag_in_stamp(capsys):
    from injectkit.cli import main

    rc = main(["attack", "--behaviors", "2", "--seeds", "1", "--quant", "4bit",
               "--format", "json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["cells"][0]["stamp"]["quant"] == "4bit"
