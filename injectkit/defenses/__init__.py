"""Defenses — mitigations the engine can apply to measure ASR with/without them.

A :class:`~injectkit.defenses.base.Defense` wraps a target's system prompt and/or
filters its input and output, so the benchmark can report attack-success-rate
*with* a defense versus *without* it (the headline robustness number). Examples a
builder will implement: spotlighting / delimiter-fencing of untrusted input,
a hardened system-prompt prefix, an input classifier that flags injection
attempts, and an output filter that catches leaked markers.

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

__all__ = [
    "Defense",
    "NullDefense",
    "DefenseRegistry",
    "get_defense",
    "list_defenses",
    "register_defense",
    "registry",
    # Built-in mitigations
    "HardenedSystemDefense",
    "InputSanitizerDefense",
    "OutputFilterDefense",
    "SandwichDefense",
    "register_builtin_defenses",
]
