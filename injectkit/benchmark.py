"""Benchmark / ASR report data model — attack-success-rate, reproducibly.

A :class:`ScanReport` answers "did my target fall for these specific attacks?".
A :class:`BenchmarkResult` answers the research question: "what is the
attack-success-rate (ASR) of this target, broken down by technique/family, with
and without each defense, and is the run reproducible?".

ASR is the fraction of attacks that succeeded:

    ASR = successes / attempts

(attempts excludes errored attacks — a target that never answered is not a
defended pass, mirroring :attr:`ScanReport.errored`). The benchmark rolls ASR up
per technique, per defense, and overall, and stamps reproducible run metadata
(corpus hash, transforms, seed, attacker model, tool version).

DEFENSIVE / AUTHORIZED USE ONLY. ASR here is measured against the benign canary
proxy: "success" means the injection bypassed the target's instructions to emit
the benign marker, not that harmful content was produced.

This module is pure data + rollups: no network, no heavy imports, no side
effects. Builders (the benchmark runner, reporters) construct these from scored
results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .models import AttackResult, Severity

__all__ = [
    "ASRCell",
    "BenchmarkRunMetadata",
    "BenchmarkResult",
    "compute_asr",
]


def compute_asr(successes: int, attempts: int) -> float:
    """Return ASR = successes / attempts, or 0.0 when there were no attempts.

    ``attempts`` should already exclude errored attacks (see module docstring).
    """
    if attempts <= 0:
        return 0.0
    return successes / attempts


@dataclass
class ASRCell:
    """One attack-success-rate measurement for a (group, defense) combination.

    A "group" is a slice of the corpus — a single technique, an attack family, or
    "overall". ``attempts`` excludes errored attacks. ``asr`` is the computed
    rate. ``highest_severity`` is the worst severity among the successes in this
    cell, for at-a-glance triage.
    """

    #: The slice this cell measures: a technique/family name or "overall".
    group: str
    #: The defense in effect for this measurement ("none" = undefended baseline).
    defense: str
    attempts: int
    successes: int
    errored: int = 0
    highest_severity: Optional[Severity] = None

    @property
    def asr(self) -> float:
        """Attack-success-rate for this cell (successes / attempts)."""
        return compute_asr(self.successes, self.attempts)

    @classmethod
    def from_results(
        cls,
        group: str,
        defense: str,
        results: list[AttackResult],
    ) -> "ASRCell":
        """Build a cell by tallying a list of scored :class:`AttackResult`.

        Errored results (``response.error`` set) are counted in ``errored`` and
        excluded from ``attempts``, so a partially-unreachable target does not
        deflate its ASR with phantom passes.
        """
        errored = sum(1 for r in results if r.response.error is not None)
        attempts = len(results) - errored
        successes = sum(1 for r in results if r.success)
        sev: Optional[Severity] = None
        for r in results:
            if r.success and (sev is None or r.severity.rank > sev.rank):
                sev = r.severity
        return cls(
            group=group,
            defense=defense,
            attempts=attempts,
            successes=successes,
            errored=errored,
            highest_severity=sev,
        )


@dataclass
class BenchmarkRunMetadata:
    """Reproducibility metadata stamped on every benchmark run.

    Records exactly what produced the numbers so a run can be reproduced and
    cited: the tool version, the target, the corpus fingerprint, the transforms
    and defenses exercised, the RNG seed, and (when adaptive attacks ran) the
    attacker model. ``corpus_hash`` is a stable digest of the attack set so two
    runs over the same corpus are comparable.
    """

    tool_version: str
    target_name: str
    target_model: Optional[str] = None
    #: Stable digest of the corpus used (e.g. sha256 of sorted attack ids+payloads).
    corpus_hash: Optional[str] = None
    #: Transform names exercised (e.g. ["identity", "base64", "rot13"]).
    transforms: list[str] = field(default_factory=list)
    #: Defense names exercised (e.g. ["none", "spotlight"]).
    defenses: list[str] = field(default_factory=list)
    #: RNG seed for any seeded transform/attacker, for exact reproduction.
    seed: Optional[int] = None
    #: Attacker model name if an adaptive run contributed (else None).
    attacker_model: Optional[str] = None
    #: Whether the LLM judge participated in scoring.
    used_judge: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def duration_s(self) -> float:
        """Wall-clock benchmark duration in seconds (0.0 if unfinished)."""
        if self.finished_at is None:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)


@dataclass
class BenchmarkResult:
    """A full ASR benchmark: per-group, per-defense cells plus run metadata.

    ``cells`` holds every measured (group, defense) :class:`ASRCell`. Convenience
    accessors compute the overall ASR, the per-technique breakdown, and the
    defense delta (how much a defense reduced ASR versus the "none" baseline) so
    reporters do not re-derive them.
    """

    metadata: BenchmarkRunMetadata
    cells: list[ASRCell] = field(default_factory=list)

    # ---- rollups -------------------------------------------------------- #

    def overall(self, defense: str = "none") -> Optional[ASRCell]:
        """The "overall" cell for ``defense``, or None if not present."""
        for c in self.cells:
            if c.group == "overall" and c.defense == defense:
                return c
        return None

    def overall_asr(self, defense: str = "none") -> float:
        """The headline overall ASR for ``defense`` (0.0 if no overall cell)."""
        cell = self.overall(defense)
        return cell.asr if cell is not None else 0.0

    def by_technique(self, defense: str = "none") -> dict[str, ASRCell]:
        """Per-technique cells for ``defense``, keyed by technique name."""
        return {
            c.group: c
            for c in self.cells
            if c.defense == defense and c.group != "overall"
        }

    def defenses(self) -> list[str]:
        """Sorted distinct defense names present in the cells."""
        return sorted({c.defense for c in self.cells})

    def defense_delta(self, defense: str) -> Optional[float]:
        """Overall ASR reduction from "none" to ``defense`` (baseline - defended).

        A positive delta means the defense lowered ASR (it helped); negative
        means it raised it. Returns None if either overall cell is missing.
        """
        base = self.overall("none")
        defended = self.overall(defense)
        if base is None or defended is None:
            return None
        return base.asr - defended.asr
