"""Evaluators: detectors and scoring that decide whether an attack succeeded.

A :class:`~injectkit.evaluators.base.Detector` inspects an (attack, response)
pair and emits a :class:`~injectkit.models.DetectorVerdict`. The scoring module
combines verdicts into a final success/severity/confidence decision.
"""

from __future__ import annotations

from .base import Detector
from .heuristics import (
    DEFAULT_TRIGGER,
    VALID_CONDITION_KEYS,
    HeuristicDetector,
)
from .scoring import ScoreOutcome, score, score_verdicts

__all__ = [
    "Detector",
    "HeuristicDetector",
    "DEFAULT_TRIGGER",
    "VALID_CONDITION_KEYS",
    "score",
    "score_verdicts",
    "ScoreOutcome",
]
