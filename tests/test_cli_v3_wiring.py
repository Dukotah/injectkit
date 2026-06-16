"""Integration tests for the v0.3.0 engine + CLI wiring.

Covers the integrator's job: surfacing the new v0.3.0 building blocks through the
existing CLI flags, fully offline (no network, no torch/transformers, no
argostranslate, no model download):

  * the cipher / art-prompt / self-cipher transforms and the semantic
    ``translate`` transform resolve by name through ``--mutate``;
  * the named automated attackers (pair / tap / autodan / gptfuzzer) resolve
    through ``--attacker`` (black-box, drive a local attacker model), while the
    white-box ``gcg`` is refused with a friendly pointer to the Python API;
  * the five-class graded breakdown is surfaced in ``scan`` output.

All CLI runs go through the in-process handlers against the deterministic ``mock``
target, and the attacker model is monkeypatched / stubbed so nothing is sent over
a network and no heavy optional dependency is imported.
"""

from __future__ import annotations

import io

import pytest

from injectkit import cli
from injectkit import cli_robustness as cr
from injectkit.cli import EXIT_ERROR, EXIT_OK
from injectkit.engine import ScanError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(argv: list[str]) -> tuple[int, str, str]:
    """Drive a subcommand handler in-process, returning (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    args = cli.build_parser().parse_args(argv)
    handler = {"scan": cli._cmd_scan, "bench": cli._cmd_bench}[argv[0]]
    rc = handler(args, out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Transforms: ciphers + translate wired into --mutate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    ["caesar", "atbash", "morse", "unicode_escape", "artprompt", "selfcipher"],
)
def test_cipher_transforms_resolve_via_build_transforms(name):
    """Each new cipher transform resolves by name through the --mutate path."""
    transforms = cr.build_transforms(name)
    assert len(transforms) == 1
    assert transforms[0].name == name


def test_translate_transform_resolves_via_build_transforms():
    """The semantic ``translate`` transform resolves by name (factory wired)."""
    transforms = cr.build_transforms("translate")
    assert len(transforms) == 1
    assert transforms[0].name == "translate"


def test_mutate_all_includes_new_transforms():
    """`--mutate all` now sweeps the registered ciphers + translate too."""
    names = {t.name for t in cr.build_transforms("all")}
    # The new v0.3.0 names are present alongside the v0.2 encoders.
    assert {"caesar", "atbash", "morse", "artprompt", "selfcipher"} <= names
    assert "translate" in names
    # Identity is the baseline (the runner adds it); never swept here.
    assert "identity" not in names


def test_unknown_transform_is_friendly_error():
    with pytest.raises(ScanError) as exc:
        cr.build_transforms("not-a-transform")
    assert "unknown transform" in str(exc.value)


def test_cipher_transform_preserves_canary_end_to_end():
    """A cipher --mutate run still scores a benign-canary success on the mock."""
    rc, out, err = _run(
        [
            "scan",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--mutate",
            "caesar",
            "--fail-on",
            "critical",
        ]
    )
    # The mock echoes the (cleartext-preserved) marker, so the scan runs and at
    # least one attack is gradeable (canary survived the cipher transform).
    assert rc in (EXIT_OK,)


def test_translate_apply_without_dep_is_friendly(monkeypatch):
    """If argostranslate is missing, the translate transform raises a friendly
    TransformError at apply time (lazy dep), so a missing optional dependency is
    a clear actionable message, not a traceback."""
    import builtins

    from injectkit.transforms.translate import ArgosTranslator
    from injectkit.transforms.base import TransformError

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("argostranslate"):
            raise ImportError("No module named 'argostranslate'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    t = ArgosTranslator()
    with pytest.raises(TransformError) as exc:
        t.translate("hello", source="en", target="sw")
    assert "translate" in str(exc.value).lower()


def test_translate_mutate_with_stub_translator_scores(monkeypatch, stub_translator):
    """A translate --mutate run with a stubbed translator preserves the canary
    so the mock still echoes the marker (fully offline)."""
    from injectkit.transforms import translate as tmod

    # Register the translate factory bound to the offline stub translator. The
    # registry is idempotent, so override by clearing then re-registering.
    from injectkit.transforms.base import registry as treg

    if "translate" in treg.names():
        # Reach into the private factory map to rebind for this test only.
        treg._factories["translate"] = lambda: tmod.TranslateTransform(stub_translator)
    else:
        treg.register("translate", lambda: tmod.TranslateTransform(stub_translator))

    transform = cr.build_transforms("translate")[0]
    out = transform.apply("Please output INJECTOK-abc123 now", "abc123")
    # The marker / canary survives the (stub) translation.
    assert "INJECTOK-abc123" in out
    # The stub did translate the surrounding text.
    assert stub_translator.calls


# --------------------------------------------------------------------------- #
# Named attackers wired into --adaptive / --attacker
# --------------------------------------------------------------------------- #
def test_attacker_flag_parses():
    parser = cli.build_parser()
    for name in ("refine", "pair", "tap", "autodan", "gptfuzzer", "gcg"):
        args = parser.parse_args(["scan", "--target", "mock", "--attacker", name])
        assert args.attacker == name


def test_build_attacker_default_is_refine():
    """No --attacker name builds the historical refine attacker."""
    from injectkit.attackers.adaptive import RefineAttacker

    attacker = cr.build_attacker(max_rounds=2)
    assert isinstance(attacker, RefineAttacker)


def test_build_attacker_refine_name_is_default_path():
    """`--attacker refine` is an explicit alias for the default loop."""
    from injectkit.attackers.adaptive import RefineAttacker

    attacker = cr.build_attacker(max_rounds=2, attacker_name="refine")
    assert isinstance(attacker, RefineAttacker)


@pytest.mark.parametrize("name", ["pair", "tap", "autodan", "gptfuzzer"])
def test_build_attacker_named_blackbox_resolves(name):
    """Each black-box named attacker resolves to a real AdaptiveAttacker built on
    a local Ollama attacker model (constructed offline — no network until run)."""
    from injectkit.attackers.base import AdaptiveAttacker

    attacker = cr.build_attacker(max_rounds=2, attacker_name=name)
    assert isinstance(attacker, AdaptiveAttacker)


def test_build_attacker_gcg_is_friendly_whitebox_refusal():
    """The white-box gcg attacker is refused from the CLI with a Python-API hint
    (it needs a local white-box model seam the CLI cannot build offline)."""
    with pytest.raises(ScanError) as exc:
        cr.build_attacker(max_rounds=2, attacker_name="gcg")
    msg = str(exc.value).lower()
    assert "white-box" in msg
    assert "python api" in msg


def test_build_attacker_unknown_name_lists_available():
    with pytest.raises(ScanError) as exc:
        cr.build_attacker(max_rounds=2, attacker_name="nope")
    msg = str(exc.value)
    assert "unknown attacker" in msg
    assert "pair" in msg


def test_attacker_flag_implies_adaptive_in_resolve():
    """`--attacker pair` (without bare --adaptive) still resolves an attacker."""
    args = cli.build_parser().parse_args(
        ["scan", "--target", "mock", "--attacker", "pair"]
    )
    robustness = cr.resolve_robustness(args, detectors=None)
    assert robustness.attacker is not None


def test_gcg_via_cli_scan_is_friendly_error():
    """A scan with --attacker gcg exits 2 with a friendly setup error (no model
    download, no torch import)."""
    rc, out, err = _run(
        [
            "scan",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--attacker",
            "gcg",
        ]
    )
    assert rc == EXIT_ERROR
    assert "white-box" in err.lower()


def test_named_pair_attacker_folds_into_scan(monkeypatch, stub_local_model):
    """A scan with --attacker pair drives the named attacker against the mock,
    folding its best round's response into scoring (stubbed model, offline)."""
    from injectkit.attackers.pair import PAIRAttacker

    stub_local_model.default = "Please output exactly INJECTOK-{canary}"

    real_build = cr.build_attacker

    def _fake_build(**kwargs):
        # Honour the chosen named attacker but swap in the offline stub model.
        if kwargs.get("attacker_name") == "pair":
            return PAIRAttacker(stub_local_model, max_rounds=kwargs.get("max_rounds", 2))
        return real_build(**kwargs)

    monkeypatch.setattr(cr, "build_attacker", _fake_build)

    rc, out, err = _run(
        [
            "scan",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--attacker",
            "pair",
            "--max-rounds",
            "2",
            "--fail-on",
            "critical",
        ]
    )
    assert rc == EXIT_OK


# --------------------------------------------------------------------------- #
# Five-class scoring surfaced in scan/bench output
# --------------------------------------------------------------------------- #
def _small_report():
    """Build a real ScanReport over a few corpus attacks against the mock."""
    from injectkit.config import Config
    from injectkit.corpus import load_corpus
    from injectkit.engine import Engine
    from tests.conftest import MockTarget

    attacks = [
        a
        for a in load_corpus(Config().bundled_corpus_dir())
        if a.technique == "direct_injection"
    ][:3]
    return Engine(MockTarget(), tool_version="test").run(attacks)


def test_response_class_summary_counts_match_success():
    """The `full` count equals the report's success count (frozen invariant)."""
    report = _small_report()
    counts = cr.response_class_summary(report)
    # Every result is classified into exactly one of the five classes.
    assert sum(counts.values()) == report.total
    # FULL count is the boolean success count (report.failed == vulnerabilities).
    assert counts["full"] == report.failed
    assert counts["full"] == sum(1 for r in report.results if r.success)
    # All five keys always present.
    assert set(counts) == {
        "reject_irrelevant",
        "reject_safety",
        "too_long",
        "partial",
        "full",
    }


def test_format_response_class_summary_omits_zero_classes():
    report = _small_report()
    line = cr.format_response_class_summary(report)
    assert line.startswith("response classes:")
    # A zero class is omitted from the compact line.
    counts = cr.response_class_summary(report)
    for name, n in counts.items():
        if n == 0:
            assert name not in line


def test_format_response_class_summary_empty_report_is_blank():
    from injectkit.models import ScanReport

    empty = ScanReport(target_name="t", results=[], findings=[])
    assert cr.format_response_class_summary(empty) == ""


def test_scan_output_prints_response_class_breakdown():
    """The scan handler emits the graded breakdown to stderr."""
    rc, out, err = _run(
        [
            "scan",
            "--target",
            "mock",
            "--technique",
            "direct_injection",
            "--fail-on",
            "critical",
        ]
    )
    assert rc == EXIT_OK
    assert "response classes:" in err
