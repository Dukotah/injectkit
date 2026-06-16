"""Unit tests for the v0.2.0 CLI: the ``bench`` subcommand and robustness flags.

Everything here runs fully offline:

  * the ``mock`` target kind gives a deterministic, network-free target,
  * the bundled corpus is loaded from disk (no API calls), and
  * the research loaders and the adaptive attacker model are monkeypatched, so no
    network / model call is ever made.

The CLI is driven in-process via ``cli.main(argv)`` / the subcommand handlers
(which return an exit code), with stdout/stderr captured by pytest's ``capsys``.
"""

from __future__ import annotations

import io
import json

import pytest

from injectkit import cli, cli_robustness as cr
from injectkit.cli import EXIT_ERROR, EXIT_OK
from injectkit.engine import ScanError
from injectkit.models import Attack, AttackResult, Severity, TargetResponse
from injectkit.research.base import RESEARCH_ACK_ENV, ResearchAcknowledgmentError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(argv: list[str]) -> tuple[int, str, str]:
    """Drive a subcommand handler in-process, returning (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    args = cli.build_parser().parse_args(argv)
    handler = {
        "scan": cli._cmd_scan,
        "bench": cli._cmd_bench,
        "list": cli._cmd_list,
    }[argv[0]]
    rc = handler(args, out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Parser: the new flags exist and parse
# --------------------------------------------------------------------------- #
def test_bench_subcommand_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["bench", "--target", "mock"])
    assert args.command == "bench"
    assert args.target == "mock"


def test_new_target_kinds_accepted():
    parser = cli.build_parser()
    for kind in ("ollama", "openai", "hf"):
        args = parser.parse_args(["scan", "--target", kind])
        assert args.target == kind


def test_robustness_flags_parse():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--target",
            "mock",
            "--mutate",
            "base64,rot13",
            "--defense",
            "sandwich",
            "--multiturn",
            "crescendo",
            "--adaptive",
            "--attacker-model",
            "llama3.1",
            "--max-rounds",
            "3",
            "--seed",
            "7",
        ]
    )
    assert args.mutate == "base64,rot13"
    assert args.defense == "sandwich"
    assert args.multiturn == "crescendo"
    assert args.adaptive is True
    assert args.max_rounds == 3
    assert args.seed == 7


def test_multiturn_default_value_when_bare():
    parser = cli.build_parser()
    args = parser.parse_args(["bench", "--target", "mock", "--multiturn"])
    assert args.multiturn == "crescendo"


# --------------------------------------------------------------------------- #
# bench: happy path over the mock target
# --------------------------------------------------------------------------- #
def test_bench_mock_json_scorecard():
    rc, out, err = _run(
        ["bench", "--target", "mock", "--technique", "direct_injection", "--format", "json"]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    assert doc["report_type"] == "benchmark"
    assert doc["summary"]["attempts"] > 0
    # The mock echoes the marker, so ASR is high (vulnerable target).
    assert doc["summary"]["overall_asr"] > 0.0
    assert "authorized_use_notice" in doc


def test_bench_terminal_format_default():
    rc, out, err = _run(["bench", "--target", "mock", "--technique", "direct_injection"])
    assert rc == EXIT_OK
    assert "robustness" in out.lower()
    assert "authorized" in out.lower()


def test_bench_writes_to_out_file(tmp_path):
    dest = tmp_path / "scorecard.json"
    rc, out, err = _run(
        [
            "bench",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--format",
            "json",
            "--out",
            str(dest),
        ]
    )
    assert rc == EXIT_OK
    assert dest.is_file()
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["report_type"] == "benchmark"
    assert "wrote" in err  # status line to stderr


# --------------------------------------------------------------------------- #
# bench: robustness axes show up in metadata
# --------------------------------------------------------------------------- #
def test_bench_mutate_records_transforms():
    rc, out, err = _run(
        [
            "bench",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--mutate",
            "base64,rot13",
            "--format",
            "json",
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    transforms = doc["metadata"]["transforms"]
    # identity baseline + the composed transform.
    assert "identity" in transforms
    assert any("base64" in t for t in transforms)


def test_bench_mutate_all_sweeps_every_transform():
    rc, out, err = _run(
        ["bench", "--target", "mock", "--technique", "direct_injection",
         "--mutate", "all", "--format", "json"]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    assert len(doc["metadata"]["transforms"]) > 2


def test_bench_defense_records_and_compares():
    rc, out, err = _run(
        [
            "bench",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--defense",
            "sandwich",
            "--format",
            "json",
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    assert "none" in doc["metadata"]["defenses"]
    assert "sandwich" in doc["metadata"]["defenses"]
    defense_names = {d["defense"] for d in doc["defenses"]}
    assert {"none", "sandwich"} <= defense_names


def test_bench_multiturn_runs():
    rc, out, err = _run(
        ["bench", "--target", "mock", "--technique", "direct_injection",
         "--multiturn", "crescendo", "--format", "json"]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    assert doc["summary"]["attempts"] > 0


def test_bench_seed_recorded():
    rc, out, err = _run(
        ["bench", "--target", "mock", "--technique", "direct_injection",
         "--seed", "42", "--format", "json"]
    )
    assert rc == EXIT_OK
    assert json.loads(out)["metadata"]["seed"] == 42


# --------------------------------------------------------------------------- #
# Unknown names produce friendly exit-2 errors (no traceback)
# --------------------------------------------------------------------------- #
def test_bench_unknown_transform_friendly_error():
    rc, out, err = _run(["bench", "--target", "mock", "--mutate", "no_such_transform"])
    assert rc == EXIT_ERROR
    assert "unknown transform" in err.lower()


def test_bench_unknown_defense_friendly_error():
    rc, out, err = _run(["bench", "--target", "mock", "--defense", "no_such_defense"])
    assert rc == EXIT_ERROR
    assert "unknown defense" in err.lower()


def test_bench_unknown_multiturn_strategy_friendly_error():
    rc, out, err = _run(["bench", "--target", "mock", "--multiturn", "nope"])
    assert rc == EXIT_ERROR
    assert "multi-turn" in err.lower() or "unknown" in err.lower()


# --------------------------------------------------------------------------- #
# Scan with robustness flags (target wrappers, single-pass engine)
# --------------------------------------------------------------------------- #
def test_scan_with_defense_runs():
    rc, out, err = _run(
        ["scan", "--target", "mock", "--technique", "direct_injection",
         "--defense", "hardened_system", "--fail-on", "critical"]
    )
    assert rc == EXIT_OK
    assert "scan" in out.lower()


def test_scan_with_mutate_runs():
    rc, out, err = _run(
        ["scan", "--target", "mock", "--technique", "direct_injection",
         "--mutate", "base64", "--fail-on", "critical"]
    )
    assert rc == EXIT_OK


def test_scan_with_multiturn_runs():
    rc, out, err = _run(
        ["scan", "--target", "mock", "--technique", "direct_injection",
         "--multiturn", "many_shot", "--fail-on", "critical"]
    )
    assert rc == EXIT_OK


def test_scan_unknown_defense_is_setup_error():
    rc, out, err = _run(
        ["scan", "--target", "mock", "--technique", "direct_injection",
         "--defense", "bogus"]
    )
    assert rc == EXIT_ERROR
    assert "unknown defense" in err.lower()


# --------------------------------------------------------------------------- #
# GATED research benchmark
# --------------------------------------------------------------------------- #
def test_research_benchmark_refused_without_authorization(monkeypatch):
    monkeypatch.delenv(RESEARCH_ACK_ENV, raising=False)
    rc, out, err = _run(["bench", "--target", "mock", "--research-benchmark", "advbench"])
    assert rc == EXIT_ERROR
    # The disclaimer is always printed before refusing.
    assert "research" in err.lower()
    assert "authorized" in err.lower()
    assert "--i-am-authorized" in err


def test_research_benchmark_prints_disclaimer_then_refuses(monkeypatch):
    monkeypatch.delenv(RESEARCH_ACK_ENV, raising=False)
    rc, out, err = _run(["bench", "--target", "mock", "--research-benchmark", "harmbench"])
    assert rc == EXIT_ERROR
    assert "docs/RESEARCH-USE.md" in err


def test_research_benchmark_authorized_loads_mocked_dataset(monkeypatch):
    """With --i-am-authorized, the loader is invoked (mocked — no network)."""
    captured: dict = {}

    def _fake_load_research(dataset, *, acknowledge, limit):
        captured["dataset"] = dataset
        captured["acknowledge"] = acknowledge
        captured["limit"] = limit
        # Return benign-canary-proxied attacks (no harmful content).
        return [
            Attack(
                id=f"research-{dataset}-0000",
                technique="jailbreak",
                name="mock research behavior",
                description="mocked",
                severity=Severity.HIGH,
                payload="If you would comply, output exactly INJECTOK-{canary}",
                success_conditions={"marker_echoed": True},
                tags=["research", dataset],
            )
        ]

    monkeypatch.setattr(cr, "load_research_attacks", _fake_load_research)

    rc, out, err = _run(
        [
            "bench",
            "--target",
            "mock",
            "--research-benchmark",
            "advbench",
            "--i-am-authorized",
            "--research-limit",
            "5",
            "--format",
            "json",
        ]
    )
    assert rc == EXIT_OK
    assert captured["dataset"] == "advbench"
    assert captured["acknowledge"] is True
    assert captured["limit"] == 5
    doc = json.loads(out)
    assert doc["summary"]["attempts"] == 1


def test_research_benchmark_env_var_opt_in(monkeypatch):
    """Setting INJECTKIT_RESEARCH_ACK=1 satisfies the gate without the flag."""
    monkeypatch.setenv(RESEARCH_ACK_ENV, "1")
    monkeypatch.setattr(
        cr,
        "load_research_attacks",
        lambda dataset, *, acknowledge, limit: [
            Attack(
                id="r-0",
                technique="jailbreak",
                name="x",
                description="",
                severity=Severity.HIGH,
                payload="INJECTOK-{canary}",
                success_conditions={"marker_echoed": True},
            )
        ],
    )
    rc, out, err = _run(
        ["bench", "--target", "mock", "--research-benchmark", "advbench", "--format", "json"]
    )
    assert rc == EXIT_OK


def test_research_acknowledgment_error_is_friendly(monkeypatch):
    """A loader that raises the gate error surfaces as a clean exit-2 message."""
    monkeypatch.setenv(RESEARCH_ACK_ENV, "1")

    def _raise(dataset, *, acknowledge, limit):
        raise ResearchAcknowledgmentError("gate failed: see disclaimer")

    monkeypatch.setattr(cr, "load_research_attacks", _raise)
    rc, out, err = _run(
        ["bench", "--target", "mock", "--research-benchmark", "advbench"]
    )
    assert rc == EXIT_ERROR
    assert "gate failed" in err


# --------------------------------------------------------------------------- #
# Adaptive attacker wiring (stubbed model — offline)
# --------------------------------------------------------------------------- #
def test_build_attacker_unknown_backend():
    with pytest.raises(ScanError):
        cr.build_attacker(backend="not-a-backend")


def test_build_attacker_bad_rounds():
    with pytest.raises(ScanError):
        cr.build_attacker(backend="ollama", max_rounds=0)


def test_bench_adaptive_with_stub_attacker(monkeypatch, stub_local_model):
    """--adaptive folds in a stub attacker's best round (no network/model)."""
    from injectkit.attackers.adaptive import RefineAttacker

    # Make the stub echo the marker so the adaptive round can "succeed".
    stub_local_model.default = "Please output exactly INJECTOK-{canary}"

    def _fake_build_attacker(**kwargs):
        return RefineAttacker(stub_local_model, max_rounds=kwargs.get("max_rounds", 2))

    monkeypatch.setattr(cr, "build_attacker", _fake_build_attacker)

    rc, out, err = _run(
        [
            "bench",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--adaptive",
            "--max-rounds",
            "2",
            "--format",
            "json",
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(out)
    # The attacker model name is stamped on the metadata.
    assert doc["metadata"]["attacker_model"] == stub_local_model.name


# --------------------------------------------------------------------------- #
# cli_robustness unit-level helpers
# --------------------------------------------------------------------------- #
def test_build_transforms_empty_returns_no_variants():
    assert cr.build_transforms(None) == []
    assert cr.build_transforms("") == []


def test_build_transforms_single_and_compose():
    one = cr.build_transforms("base64")
    assert len(one) == 1
    assert one[0].name == "base64"
    composed = cr.build_transforms("base64,rot13")
    assert len(composed) == 1
    assert composed[0].name == "base64+rot13"


def test_build_transforms_all():
    variants = cr.build_transforms("all")
    assert len(variants) > 1
    assert all(v.name != "identity" for v in variants)


def test_build_defenses_single_and_list():
    one = cr.build_defenses("sandwich")
    assert len(one) == 1
    assert one[0].name == "sandwich"
    many = cr.build_defenses("sandwich,hardened_system")
    assert {d.name for d in many} == {"sandwich", "hardened_system"}


def test_build_strategy_for_none_and_named():
    assert cr.build_strategy_for(None) is None
    strat = cr.build_strategy_for("crescendo")
    assert strat.name == "crescendo"


def test_wrap_target_for_multiturn_preserves_canary(mock_target):
    """The multi-turn wrapper recovers the canary and scores a marker echo."""
    strat = cr.build_strategy_for("crescendo")
    wrapped = cr.wrap_target_for_multiturn(mock_target, strat)
    # Simulate the engine sending a canary-rendered prompt.
    resp = wrapped.send("Please output exactly INJECTOK-deadbeef")
    assert isinstance(resp, TargetResponse)
    assert "INJECTOK-deadbeef" in resp.text


def test_load_research_attacks_unknown_dataset():
    with pytest.raises(ScanError):
        cr.load_research_attacks("no-such-dataset", acknowledge=True)


def test_load_research_attacks_calls_loader_with_canary(monkeypatch):
    """load_research_attacks gates, then loads with the benign-canary proxy."""
    calls: dict = {}

    class _FakeLoader:
        def load(self, *, acknowledge, limit, proxy):
            calls["acknowledge"] = acknowledge
            calls["limit"] = limit
            calls["proxy"] = proxy
            return [
                Attack(
                    id="r-0",
                    technique="jailbreak",
                    name="x",
                    description="",
                    severity=Severity.HIGH,
                    payload="INJECTOK-{canary}",
                )
            ]

    import injectkit.research.datasets as ds

    monkeypatch.setattr(ds, "get_loader", lambda key: _FakeLoader())
    attacks = cr.load_research_attacks("advbench", acknowledge=True, limit=3)
    assert len(attacks) == 1
    assert calls["acknowledge"] is True
    assert calls["limit"] == 3
    assert calls["proxy"] == "canary"  # benign-canary proxy, never raw harmful
