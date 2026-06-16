"""Adaptive attackers — propose/refine loops that optimise attack STRUCTURE.

An :class:`~injectkit.attackers.base.AdaptiveAttacker` iteratively rewrites an
attack to bypass a target's instructions, scored each round by a judge/detector
(standard automated red-teaming / ASR methodology). It is **local-model-first**:
the default attacker model is a local model (Ollama / an HF model / a stub),
needs no API key, and is lazy-imported.

ETHICS — NON-NEGOTIABLE: the attacker optimises attack STRUCTURE to defeat
instructions, measured by a benign canary proxy. It is NOT a harmful-output
generator. It only ever tries to make the target emit the benign success marker
it was told to withhold. Tests use a scripted stub model and make no network or
model calls.
"""

from __future__ import annotations

from .base import (
    AdaptiveAttacker,
    AttackerError,
    AttackerModel,
    AttackerResult,
    AttackerTranscriptStep,
)
from .adaptive import (
    OllamaAttackerModel,
    RefineAttacker,
    RefinePromptBuilder,
    ensure_canary,
    extract_payload,
)

__all__ = [
    "AttackerModel",
    "AdaptiveAttacker",
    "AttackerResult",
    "AttackerTranscriptStep",
    "AttackerError",
    "RefineAttacker",
    "RefinePromptBuilder",
    "OllamaAttackerModel",
    "ensure_canary",
    "extract_payload",
]
