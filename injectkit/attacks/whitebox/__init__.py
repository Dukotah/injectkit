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

#: Symbols re-exported lazily from the prefill submodule (CHUNK 5).
_PREFILL_SYMBOLS = frozenset(
    {
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
    }
)

#: Symbols re-exported lazily from the judge_loop submodule (v0.5 judge-in-loop).
_JUDGE_LOOP_SYMBOLS = frozenset(
    {
        "ReinforceGCGAttack",
        "UJAAttack",
        "JudgeReward",
        "assert_opt_judge_distinct",
        "OptJudgeCircularityError",
        "reinforce_gcg_run",
        "uja_run",
    }
)

__all__ = sorted(_PREFILL_SYMBOLS | _JUDGE_LOOP_SYMBOLS)


def __getattr__(name: str):
    """Lazily resolve a prefill / judge_loop symbol (PEP 562), dodging the cycle."""
    if name in _PREFILL_SYMBOLS:
        from . import prefill

        return getattr(prefill, name)
    if name in _JUDGE_LOOP_SYMBOLS:
        from . import judge_loop

        return getattr(judge_loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
