"""Scan orchestrator: corpus -> per-attack send/evaluate/score -> ScanReport.

The engine is the heart of a scan. Given a :class:`~injectkit.targets.base.Target`
and a list of :class:`~injectkit.evaluators.base.Detector` objects, it:

  1. Renders each :class:`~injectkit.models.Attack` with a fresh, per-run canary
     marker (so a marker echoed back is unambiguous evidence of compliance).
  2. Sends the rendered payload (plus any per-attack ``system`` / ``context``) to
     the target via the provider-agnostic :meth:`Target.send` contract.
  3. Runs every detector over the (attack, response, canary) triple, collecting
     :class:`~injectkit.models.DetectorVerdict` objects.
  4. Scores the verdicts into a final success/severity/confidence via
     :func:`injectkit.evaluators.scoring.score`.
  5. Aggregates the results and the subset that succeeded (the
     :class:`~injectkit.models.Finding` objects) into a :class:`ScanReport`.

DEFENSIVE / AUTHORIZED USE ONLY. The engine only drives whatever target the
caller hands it; injectkit's posture is "scan your own endpoint". Construction
of targets/detectors from config (which is where keys/URLs come from) lives in
the CLI and the adapters, not here.

The engine is deliberately small and dependency-light: it imports only the core
models, the heuristic detector, and the scoring helper. The optional LLM judge
and the heavy target SDKs are wired in by the caller (the CLI), so importing the
engine never pulls in an optional dependency.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Iterable, Optional, Sequence

from .models import (
    Attack,
    AttackResult,
    Finding,
    ScanReport,
    TargetResponse,
)
from .evaluators.base import Detector
from .evaluators.heuristics import HeuristicDetector
from .evaluators.scoring import score
from .targets.base import Target

__all__ = ["Engine", "run_scan", "make_canary", "ScanError"]


class ScanError(RuntimeError):
    """Raised for an unrecoverable scan setup problem (e.g. an empty corpus)."""


def make_canary() -> str:
    """Return a short, unique, URL-safe canary token for one attack run.

    The canary is substituted into the payload's ``{canary}`` placeholder and is
    what the heuristic detector hunts for in the response. A fresh token per
    attack means a marker echoed back could only have come from *this* request,
    which keeps false positives near zero.
    """
    # 12 hex chars is plenty of entropy to be unique within a scan while staying
    # short enough to read in a report.
    return "ik" + uuid.uuid4().hex[:12]


class Engine:
    """Run a corpus of attacks against one target and produce a ScanReport.

    Args:
        target: The :class:`~injectkit.targets.base.Target` to probe. Must honor
            the Target protocol (never raise on a normal failed request — return
            a :class:`TargetResponse` with ``error`` set instead).
        detectors: Detectors to run per attack. Defaults to a single offline
            :class:`~injectkit.evaluators.heuristics.HeuristicDetector` so the
            engine works with zero configuration and no API key.
        use_judge: Whether judge verdicts take precedence during scoring. Set
            this True only when a judge detector is included in ``detectors``;
            it is passed straight through to :func:`scoring.score`.
        canary_factory: Callable returning a fresh canary per attack. Injectable
            for deterministic tests.
        on_result: Optional callback invoked with each scored
            :class:`AttackResult` as it completes (for live progress output).
        tool_version: Version string stamped on the report.
    """

    def __init__(
        self,
        target: Target,
        detectors: Optional[Sequence[Detector]] = None,
        *,
        use_judge: bool = False,
        canary_factory: Callable[[], str] = make_canary,
        on_result: Optional[Callable[[AttackResult], None]] = None,
        tool_version: str = "0.1.0",
    ) -> None:
        self.target = target
        self.detectors: list[Detector] = (
            list(detectors) if detectors is not None else [HeuristicDetector()]
        )
        if not self.detectors:
            raise ScanError("Engine requires at least one detector.")
        self.use_judge = use_judge
        self.canary_factory = canary_factory
        self.on_result = on_result
        self.tool_version = tool_version

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, attacks: Iterable[Attack]) -> ScanReport:
        """Run every attack in ``attacks`` and return a populated ScanReport.

        Args:
            attacks: The attacks to run (typically from
                :func:`injectkit.corpus.load_corpus`, optionally filtered).

        Returns:
            A :class:`ScanReport` with one :class:`AttackResult` per attack and
            a :class:`Finding` for each attack that succeeded.

        Raises:
            ScanError: if ``attacks`` is empty (nothing to scan).
        """
        attacks = list(attacks)
        if not attacks:
            raise ScanError(
                "No attacks to run. The corpus is empty or every attack was "
                "filtered out by --technique."
            )

        started_at = time.time()
        results: list[AttackResult] = []
        for attack in attacks:
            result = self.run_one(attack)
            results.append(result)
            if self.on_result is not None:
                self.on_result(result)
        finished_at = time.time()

        findings = [Finding.from_result(r) for r in results if r.success]

        return ScanReport(
            target_name=getattr(self.target, "name", "target"),
            target_model=self._target_model(results),
            results=results,
            findings=findings,
            started_at=started_at,
            finished_at=finished_at,
            tool_version=self.tool_version,
        )

    def run_one(self, attack: Attack) -> AttackResult:
        """Run a single attack: render, send, evaluate, score.

        Never raises on an adapter or detector hiccup — a target that violates
        its contract by raising is captured into an error
        :class:`TargetResponse`, and a detector that raises is recorded as a
        non-success verdict, so one bad attack can never abort the whole scan.
        """
        canary = self.canary_factory()
        rendered = attack.render(canary)
        # Per-attack system/context are also canary-rendered so a sentinel
        # planted in the system prompt matches the one the detector hunts for.
        system = self._render_optional(attack.system, canary)
        context = self._render_optional(attack.context, canary)

        start = time.perf_counter()
        response = self._send(rendered, system, context)
        verdicts = [self._evaluate(d, attack, response, canary) for d in self.detectors]
        duration = time.perf_counter() - start

        result = AttackResult(
            attack=attack,
            canary=canary,
            response=response,
            verdicts=verdicts,
            duration_s=duration,
        )
        return score(result, use_judge=self.use_judge)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_optional(value: Optional[str], canary: str) -> Optional[str]:
        """Canary-render an optional system/context string (None passes through)."""
        if value is None:
            return None
        return value.replace("{canary}", canary)

    def _send(
        self, prompt: str, system: Optional[str], context: Optional[str]
    ) -> TargetResponse:
        """Call ``target.send`` defensively, converting any fault to an error.

        Adapters are contractually required not to raise and to return a
        :class:`TargetResponse`, but a misbehaving or community-contributed
        target might do neither. We never let that abort the scan: a raised
        exception *or* a wrong return type both become an errored response.
        """
        try:
            response = self.target.send(prompt, system=system, context=context)
        except Exception as exc:  # noqa: BLE001 - one bad attack must not kill the scan
            return TargetResponse(
                text="",
                error=f"target.send raised {type(exc).__name__}: {exc}",
            )
        # A target that violates the protocol by returning a non-TargetResponse
        # (e.g. a raw dict or None) must not crash the scan or, worse, be fed to
        # a detector where its attributes are read. Treat it as a target error.
        if not isinstance(response, TargetResponse):
            return TargetResponse(
                text="",
                error=(
                    "target.send returned a "
                    f"{type(response).__name__}, not a TargetResponse "
                    "(adapter violated the Target protocol)."
                ),
            )
        return response

    def _evaluate(
        self,
        detector: Detector,
        attack: Attack,
        response: TargetResponse,
        canary: str,
    ) -> "DetectorVerdict":  # noqa: F821 - imported lazily below for the annotation
        """Run one detector defensively, recording a failure as a non-success.

        Guards both fault modes of a community-contributed detector: one that
        *raises*, and one that *returns* something other than a
        :class:`DetectorVerdict`. Either way we substitute a non-success verdict
        so a single bad detector can never abort the scan or feed a malformed
        object into scoring/reporting (a false positive/negative or a crash).
        """
        from .models import DetectorVerdict  # local import keeps top clean

        name = getattr(detector, "name", "detector")
        try:
            verdict = detector.evaluate(attack, response, canary)
        except Exception as exc:  # noqa: BLE001 - a flaky detector must not crash the scan
            return DetectorVerdict(
                detector=name,
                success=False,
                confidence=0.0,
                rationale=f"Detector raised {type(exc).__name__}: {exc}; "
                "treated as non-success.",
            )
        if not isinstance(verdict, DetectorVerdict):
            return DetectorVerdict(
                detector=name,
                success=False,
                confidence=0.0,
                rationale=(
                    f"Detector returned a {type(verdict).__name__}, not a "
                    "DetectorVerdict (protocol violation); treated as "
                    "non-success."
                ),
            )
        return verdict

    @staticmethod
    def _target_model(results: list[AttackResult]) -> Optional[str]:
        """Best-effort model id for the report header (first response's model)."""
        for r in results:
            if r.response.model:
                return r.response.model
        return None


def run_scan(
    target: Target,
    attacks: Iterable[Attack],
    detectors: Optional[Sequence[Detector]] = None,
    *,
    use_judge: bool = False,
    on_result: Optional[Callable[[AttackResult], None]] = None,
    tool_version: str = "0.1.0",
) -> ScanReport:
    """Convenience wrapper: build an :class:`Engine` and run it once.

    Args:
        target: The target to scan.
        attacks: The attacks to run.
        detectors: Detectors (default: offline heuristics only).
        use_judge: Whether judge verdicts take scoring precedence.
        on_result: Optional per-result progress callback.
        tool_version: Version stamped on the report.

    Returns:
        The populated :class:`ScanReport`.
    """
    engine = Engine(
        target,
        detectors,
        use_judge=use_judge,
        on_result=on_result,
        tool_version=tool_version,
    )
    return engine.run(attacks)
