"""Defenses — mitigations the engine can apply to measure ASR with/without them.

A :class:`~injectkit.defenses.base.Defense` wraps a target's system prompt and/or
filters its input and output, so the benchmark can report attack-success-rate
*with* a defense versus *without* it (the headline robustness number). Examples a
builder will implement: spotlighting / delimiter-fencing of untrusted input,
a hardened system-prompt prefix, an input classifier that flags injection
attempts, and an output filter that catches leaked markers.

v0.3.1 adds two canonical benchmark-grade lightweight defenses:

* :class:`~injectkit.defenses.smoothllm.SmoothLLMDefense` (``"smoothllm"``) —
  randomized smoothing via N char-level perturbations + majority vote
  (Robey et al., arXiv:2310.03684).
* :class:`~injectkit.defenses.perplexity_filter.PerplexityFilterDefense`
  (``"perplexity_filter"``) — reject inputs whose character-bigram perplexity
  exceeds a threshold (Alon & Kamfonas, arXiv:2308.14132; Jain et al.,
  arXiv:2309.00614).

DEFENSIVE / AUTHORIZED USE ONLY. Defenses are mitigations you evaluate on your
own target — this is the "does my guardrail actually help?" measurement.
"""

from __future__ import annotations

from .base import (
    Defense,
    DefenseRegistry,
    NullDefense,
    get_defense,
    list_defenses,
    register_defense,
    registry,
)
from .mitigations import (
    HardenedSystemDefense,
    InputSanitizerDefense,
    OutputFilterDefense,
    SandwichDefense,
    register_builtin_defenses,
)
# Import new defenses to trigger their self-registration.
from .smoothllm import SmoothLLMDefense, _SmoothLLMTarget, apply_perturbation
from .perplexity_filter import (
    CharBigramModel,
    PerplexityFilterDefense,
    REFERENCE_CORPUS,
)

__all__ = [
    "Defense",
    "NullDefense",
    "DefenseRegistry",
    "get_defense",
    "list_defenses",
    "register_defense",
    "registry",
    # Built-in mitigations (v0.1–v0.2)
    "HardenedSystemDefense",
    "InputSanitizerDefense",
    "OutputFilterDefense",
    "SandwichDefense",
    "register_builtin_defenses",
    # v0.3.1 lightweight defenses
    "SmoothLLMDefense",
    "_SmoothLLMTarget",
    "apply_perturbation",
    "CharBigramModel",
    "PerplexityFilterDefense",
    "REFERENCE_CORPUS",
]
