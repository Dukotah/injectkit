"""v0.4 white-box attack interface — the shared :class:`Attack` ABC + registry.

The seam ROADMAP §6.0 freezes: one typed contract every white-box attack family
implements (``run(model, tokenizer, messages, target, cfg, defense=None) ->
AttackResult``), a name registry resolving attacks by key, and typed Pydantic
configs. This subpackage is additive to the shipped v0.3 ``attackers/`` and
``attacks/`` packages and does not modify them.

Importing this package registers the built-in attacks (currently ``gcg``), so
``injectkit.whitebox.registry.get_attack("gcg")`` resolves after
``import injectkit.whitebox``.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from .base import ArchitectureError, Attack, AttackResult
from .config import AttackConfig, GCGConfig
from .registry import (
    AttackRegistry,
    get_attack,
    get_attack_class,
    list_attacks,
    register,
    registry,
)
from .zoo import (
    ZOO_PATH,
    ZooEntry,
    ZooError,
    check_attack_supported,
    get_entry,
    list_models,
    load_by_revision,
    load_model,
    load_zoo,
)

# Import the concrete attacks for their @register side effect (wires "gcg").
from . import gcg  # noqa: E402,F401  (import-time registration)
from .gcg import GCGAttack

__all__ = [
    "Attack",
    "AttackResult",
    "ArchitectureError",
    "AttackConfig",
    "GCGConfig",
    "AttackRegistry",
    "registry",
    "register",
    "get_attack",
    "get_attack_class",
    "list_attacks",
    "GCGAttack",
    # Model zoo (CHUNK 2-model-zoo).
    "ZOO_PATH",
    "ZooEntry",
    "ZooError",
    "check_attack_supported",
    "get_entry",
    "list_models",
    "load_by_revision",
    "load_model",
    "load_zoo",
]
