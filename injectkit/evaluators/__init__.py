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

# v0.3.0 five-class response framework. The boolean ``success`` stays derivable
# as ``ResponseClass.is_success`` (True only for FULL), and FULL is defined to
# coincide with the engine's heuristic boolean success so existing
# scoring/reports are unchanged.
from .response_class import (  # noqa: F401
    LENGTH_STOP_REASONS,
    ResponseClass,
    classify_response,
    classify_result,
)

__all__ = [
    "Detector",
    "HeuristicDetector",
    "DEFAULT_TRIGGER",
    "VALID_CONDITION_KEYS",
    "score",
    "score_verdicts",
    "ScoreOutcome",
    # v0.3.0 five-class response framework
    "ResponseClass",
    "classify_response",
    "classify_result",
    "LENGTH_STOP_REASONS",
]
