"""White-box attack families that live under the v0.3 ``attacks/`` namespace.

CHUNK 5-prefill-attack (ROADMAP §6.x). The shipped package is flat-layout and the
v0.4 white-box interface (:class:`injectkit.whitebox.base.Attack`, the registry,
the typed configs, the model zoo) lives in :mod:`injectkit.whitebox`. The chunk
spec names ``injectkit/attacks/whitebox/prefill.py`` as the home of the first-class
prefill attack, so this subpackage hosts it and re-exports the public surface.

The prefill attack is a registered :class:`injectkit.whitebox.base.Attack`
subclass — it resolves through the *same* white-box registry as GCG
(``injectkit.whitebox.registry``), so the existing registry + bench/harness wiring
produces a leaderboard row for ``prefill`` with no new plumbing. Importing
``injectkit.attacks.whitebox.prefill`` (or accessing any name re-exported here)
registers ``prefill`` on the white-box registry.

Circular-import note: :mod:`injectkit.whitebox`'s package ``__init__`` imports this
subpackage's ``prefill`` submodule for its ``@register`` side effect, and ``prefill``
in turn imports ``Attack``/``PrefillConfig`` from ``injectkit.whitebox``'s
submodules. To keep both import orders working, this package re-exports the prefill
symbols LAZILY (PEP 562 ``__getattr__``) rather than eagerly at import time — so
importing the package never pulls a half-initialised ``prefill`` module.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

__all__ = [
    "PrefillAttack",
    "PrefillConfig",
    "PrefillTrial",
    "GenerationResult",
    "PREFILL_INVENTORY",
    "GENERIC_PREFILL",
    "GPT_OSS_PREFILL_FAMILY",
    "candidate_prefills_for",
    "family_of",
    "run",
]


def __getattr__(name: str):
    """Lazily resolve a prefill symbol (PEP 562), dodging the import cycle."""
    if name in __all__:
        from . import prefill

        return getattr(prefill, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
