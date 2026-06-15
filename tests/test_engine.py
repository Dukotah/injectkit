"""Unit + integration tests for the scan orchestrator (injectkit.engine).

These run fully offline against the in-repo :class:`MockTarget` (see
conftest.py); no network and no API key are touched. They exercise:

  * the end-to-end scan loop against a vulnerable and a clean target,
  * canary generation/uniqueness and per-attack rendering of system/context,
  * robustness: a target that raises and a detector that raises must NOT abort
    the scan — they become an errored response / non-success verdict,
  * report aggregation (findings, counts, timing, model metadata),
  * the ``run_scan`` convenience wrapper and the empty-corpus guard.
"""

from __future__ import annotations

import re
import time

import pytest

from injectkit.engine import (
    Engine,
    ScanError,
    make_canary,
    run_scan,
)
from injectkit.models import (
    Attack,
    AttackResult,
    DetectorVerdict,
    ScanReport,
    Severity,
    TargetResponse,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class RaisingTarget:
    """A Target whose send() always raises — to prove the engine isolates it."""

    name = "raising"

    def send(self, prompt, system=None, context=None):  # noqa: D401, ANN001
        raise RuntimeError("boom: simulated transport failure")


class RaisingDetector:
    """A Detector whose evaluate() always raises — must not crash a scan."""

    name = "raising-detector"

    def evaluate(self, attack, response, canary):  # noqa: ANN001
        raise ValueError("detector exploded")


class RecordingDetector:
    """A Detector that records every (attack, response, canary) it sees."""

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, str]] = []

    def evaluate(self, attack, response, canary):  # noqa: ANN001
        self.seen.append((attack.id, response.text, canary))
        return DetectorVerdict(detector=self.name, success=False, confidence=0.0)


# --------------------------------------------------------------------------- #
# Canary
# --------------------------------------------------------------------------- #
def test_make_canary_is_unique_and_prefixed():
    """Canaries are unique per call and carry the 'ik' prefix, word-safe chars."""
    canaries = {make_canary() for _ in range(200)}
    assert len(canaries) == 200  # all unique
    for c in canaries:
        assert c.startswith("ik")
        assert re.fullmatch(r"[A-Za-z0-9]+", c)


# --------------------------------------------------------------------------- #
# End-to-end happy paths
# --------------------------------------------------------------------------- #
def test_scan_vulnerable_target_produces_finding(mock_target, sample_attack):
    """A vulnerable target that echoes the marker yields a success + finding."""
    report = run_scan(mock_target, [sample_attack])

    assert isinstance(report, ScanReport)
    assert report.total == 1
    assert report.failed == 1  # one vulnerability found
    assert report.passed == 0
    assert len(report.findings) == 1

    result = report.results[0]
    assert result.success is True
    assert result.detected is True
    # direct_injection floors at LOW but the attack declared HIGH -> stays HIGH.
    assert result.severity == Severity.HIGH
    assert result.confidence > 0.0

    finding = report.findings[0]
    assert finding.attack_id == sample_attack.id
    assert finding.technique == "direct_injection"


def test_scan_clean_target_no_findings(clean_target, sample_attack):
    """A clean (always-refusing) target defends everything: zero findings."""
    report = run_scan(clean_target, [sample_attack])

    assert report.total == 1
    assert report.failed == 0
    assert report.passed == 1
    assert report.findings == []
    assert report.highest_severity is None

    result = report.results[0]
    assert result.success is False
    assert result.response.refused is True


def test_marker_carries_the_per_attack_canary(mock_target, sample_attack):
    """The rendered payload's marker contains the live canary, and it echoes back."""
    report = run_scan(mock_target, [sample_attack])
    result = report.results[0]

    # The mock echoes the marker line; it must contain INJECTOK-<canary>.
    assert f"INJECTOK-{result.canary}" in result.response.text
    # The target actually received the rendered payload (canary substituted).
    assert mock_target.calls[0]["prompt"] == sample_attack.render(result.canary)
    assert "{canary}" not in mock_target.calls[0]["prompt"]


def test_system_prompt_is_rendered_and_passed_through():
    """A {canary} in the attack's system prompt is rendered and sent to target."""
    recorder = RecordingDetector()

    class SysTarget:
        name = "sys"

        def __init__(self):
            self.last_system = None

        def send(self, prompt, system=None, context=None):  # noqa: ANN001
            self.last_system = system
            return TargetResponse(text="ok", model="sys")

    attack = Attack(
        id="sys-1",
        technique="system_prompt_leak",
        name="leak",
        description="d",
        severity=Severity.HIGH,
        payload="leak it: INJECTOK-{canary}",
        success_conditions={"system_prompt_leaked": True},
        system="SECRET-CONFIG-{canary}",
    )
    target = SysTarget()
    engine = Engine(target, [recorder])
    report = engine.run([attack])

    canary = report.results[0].canary
    # The {canary} placeholder in the system prompt was rendered before sending.
    assert target.last_system == f"SECRET-CONFIG-{canary}"
    assert "{canary}" not in target.last_system


# --------------------------------------------------------------------------- #
# Robustness: one bad attack/detector must not abort the scan
# --------------------------------------------------------------------------- #
def test_target_that_raises_becomes_errored_response(sample_attack):
    """A target.send that raises is captured into an errored response, no crash."""
    report = run_scan(RaisingTarget(), [sample_attack])

    assert report.total == 1
    result = report.results[0]
    assert result.success is False  # an error is never a success
    assert result.response.error is not None
    assert "RuntimeError" in result.response.error


def test_detector_that_raises_is_isolated(mock_target, sample_attack):
    """A detector that raises yields a non-success verdict; scan still completes."""
    report = run_scan(
        mock_target, [sample_attack], detectors=[RaisingDetector()]
    )

    assert report.total == 1
    verdict = report.results[0].verdicts[0]
    assert verdict.success is False
    assert "ValueError" in verdict.rationale
    # No finding, because the only detector failed.
    assert report.findings == []


def test_target_returning_non_response_is_isolated(sample_attack):
    """A target that returns a non-TargetResponse becomes an errored response.

    The Target protocol is community-pluggable; an adapter that violates the
    contract by returning a raw dict/None must not crash the scan or be fed to a
    detector that reads its attributes.
    """

    class WrongTypeTarget:
        name = "wrongtype"

        def send(self, prompt, system=None, context=None):  # noqa: ANN001
            return {"text": "INJECTOK-not-a-response"}  # protocol violation

    report = run_scan(WrongTypeTarget(), [sample_attack])

    assert report.total == 1
    result = report.results[0]
    assert result.success is False
    assert result.response.error is not None
    assert "TargetResponse" in result.response.error
    assert report.findings == []


def test_detector_returning_non_verdict_is_isolated(mock_target, sample_attack):
    """A detector returning a non-DetectorVerdict yields a non-success verdict.

    Mirrors the raise guard: a community detector that returns garbage (a str,
    None, etc.) must not crash the scan nor corrupt scoring/reporting.
    """

    class WrongTypeDetector:
        name = "wrongtype-detector"

        def evaluate(self, attack, response, canary):  # noqa: ANN001
            return "not a verdict"  # protocol violation

    report = run_scan(mock_target, [sample_attack], detectors=[WrongTypeDetector()])

    assert report.total == 1
    verdict = report.results[0].verdicts[0]
    assert verdict.success is False
    assert "DetectorVerdict" in verdict.rationale
    assert report.results[0].success is False
    assert report.findings == []


def test_one_bad_attack_does_not_abort_remaining(mock_target, sample_attack):
    """Mixing a normal attack with a target failure still scans every attack.

    Uses a target that raises only for a specific marker so we can confirm the
    scan continues past the failure and the good attack still succeeds.
    """

    class FlakyTarget:
        name = "flaky"

        def send(self, prompt, system=None, context=None):  # noqa: ANN001
            if "boom" in prompt:
                raise RuntimeError("flaky failure")
            # Echo the marker so the heuristic detector flags success.
            m = re.search(r"INJECTOK-[A-Za-z0-9]+", prompt)
            return TargetResponse(text=m.group(0) if m else "", model="flaky")

    bad = Attack(
        id="bad",
        technique="direct_injection",
        name="bad",
        description="d",
        severity=Severity.HIGH,
        payload="boom INJECTOK-{canary}",
        success_conditions={"marker_echoed": True},
    )
    report = run_scan(FlakyTarget(), [bad, sample_attack])

    assert report.total == 2
    by_id = {r.attack.id: r for r in report.results}
    assert by_id["bad"].response.error is not None
    assert by_id["bad"].success is False
    assert by_id[sample_attack.id].success is True
    assert report.failed == 1


# --------------------------------------------------------------------------- #
# Report aggregation
# --------------------------------------------------------------------------- #
def test_report_metadata_and_timing(mock_target, sample_attack):
    """Report carries target name/model, version, ordered results, and timing."""
    before = time.time()
    report = run_scan(mock_target, [sample_attack], tool_version="9.9.9")
    after = time.time()

    assert report.target_name == "mock"
    assert report.target_model == "mock"  # mock stamps its name as model
    assert report.tool_version == "9.9.9"
    assert before <= report.started_at <= report.finished_at <= after
    assert report.duration_s >= 0.0
    assert report.results[0].duration_s >= 0.0


def test_results_preserve_corpus_order(mock_target):
    """Results come back in the same order the attacks were supplied."""
    attacks = [
        Attack(
            id=f"a{i}",
            technique="direct_injection",
            name=f"a{i}",
            description="d",
            severity=Severity.LOW,
            payload="benign question",
            success_conditions={"marker_echoed": True},
        )
        for i in range(5)
    ]
    report = run_scan(mock_target, attacks)
    assert [r.attack.id for r in report.results] == [f"a{i}" for i in range(5)]


def test_on_result_callback_fires_per_attack(mock_target, sample_attack):
    """The on_result progress callback is invoked once per completed attack."""
    seen: list[AttackResult] = []
    engine = Engine(mock_target, on_result=seen.append)
    engine.run([sample_attack, sample_attack])
    assert len(seen) == 2
    assert all(isinstance(r, AttackResult) for r in seen)


def test_injectable_canary_factory_is_used(mock_target, sample_attack):
    """A custom canary factory makes runs deterministic for tests."""
    engine = Engine(mock_target, canary_factory=lambda: "ikFIXEDCANARY")
    report = engine.run([sample_attack])
    assert report.results[0].canary == "ikFIXEDCANARY"
    assert "INJECTOK-ikFIXEDCANARY" in report.results[0].response.text


# --------------------------------------------------------------------------- #
# Guards / config
# --------------------------------------------------------------------------- #
def test_empty_corpus_raises_scan_error(mock_target):
    """An empty attack list is a setup error, not a silent empty report."""
    with pytest.raises(ScanError):
        run_scan(mock_target, [])


def test_engine_requires_a_detector(mock_target):
    """Constructing with an explicit empty detector list is rejected."""
    with pytest.raises(ScanError):
        Engine(mock_target, detectors=[])


def test_default_detector_is_offline_heuristic(mock_target, sample_attack):
    """With no detectors supplied, the engine uses the offline heuristic only."""
    engine = Engine(mock_target)
    assert len(engine.detectors) == 1
    assert engine.detectors[0].name == "heuristics"
    report = engine.run([sample_attack])
    assert report.results[0].verdicts[0].detector == "heuristics"


def test_run_scan_loads_bundled_corpus_path(mock_target):
    """run accepts attacks loaded from the bundled corpus directory end-to-end."""
    from injectkit.config import Config
    from injectkit.corpus import load_corpus

    attacks = load_corpus(Config().bundled_corpus_dir())
    assert attacks, "bundled corpus should contain attacks"
    report = run_scan(mock_target, attacks)
    assert report.total == len(attacks)
    # Every attack produced a scored result with a verdict.
    assert all(r.verdicts for r in report.results)
