"""Attack strategies — generators that emit single-shot or multi-turn attacks.

v0.1.0 ran a static corpus of single-shot :class:`~injectkit.models.Attack`
cases. v0.2.0 generalises this: an :class:`~injectkit.attacks.base.AttackStrategy`
*produces* attack steps, which may be a single prompt or a scripted sequence of
turns (crescendo, role-play escalation, many-shot delivered as real turns). A
plain corpus attack is just the simplest strategy (one step).

DEFENSIVE / AUTHORIZED USE ONLY. Strategies default to benign-canary proxy
payloads — they measure whether structure bypasses instructions, not whether the
model can be made to emit harmful content.
"""

from __future__ import annotations

from .base import (
    AttackStep,
    AttackStrategy,
    MultiTurnAttack,
    SingleShotStrategy,
    StrategyError,
    attack_to_strategy,
)
from .multiturn import (
    MULTI_TURN_STRATEGIES,
    ContextOverflowStrategy,
    CrescendoReplyReferencingStrategy,
    CrescendoStrategy,
    ManyShotStrategy,
    PersonaPrimingStrategy,
    build_strategy,
)

__all__ = [
    "AttackStep",
    "MultiTurnAttack",
    "AttackStrategy",
    "SingleShotStrategy",
    "StrategyError",
    "attack_to_strategy",
    "CrescendoStrategy",
    "CrescendoReplyReferencingStrategy",
    "ManyShotStrategy",
    "ContextOverflowStrategy",
    "PersonaPrimingStrategy",
    "MULTI_TURN_STRATEGIES",
    "build_strategy",
]
