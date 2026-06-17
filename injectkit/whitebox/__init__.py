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
from .config import AttackConfig, GCGConfig, PrefillConfig
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

# Wire the prefill attack ("prefill"). It lives under attacks/whitebox/ (the chunk
# spec's path) but registers on THIS registry, so it resolves like gcg.
#
# Circular-import care: prefill.py imports `Attack`/`AttackResult`/`PrefillConfig`
# from THIS package's submodules, so importing prefill *first* re-enters this
# __init__ while prefill is only partially initialised (its `PrefillAttack` class
# not yet defined). We therefore import the prefill SUBMODULE here purely for its
# @register side effect (the `attacks.whitebox` package __init__ keeps its own
# symbols lazy so this submodule import does not pull a half-built `PrefillAttack`),
# and expose the `PrefillAttack` symbol LAZILY via the module-level __getattr__
# below (PEP 562) so it resolves only once prefill has finished initialising. This
# keeps both import orders (whitebox-first and prefill-first) green.
from ..attacks.whitebox import prefill as _prefill  # noqa: E402,F401  (registration)


def __getattr__(name: str):  # PEP 562 — lazy re-export to dodge the import cycle.
    if name == "PrefillAttack":
        return _prefill.PrefillAttack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .gcg_hard import (
    AttackBuffer,
    ProbeSamplingConfig,
    PromptSlices,
    filter_ids,
    locate_optim_slice,
    round_trips,
    sample_candidates,
    token_gradients_onehot,
)
from .targets import (
    FIXED_BASELINE_PREFIX,
    PrefixCandidate,
    PrefixScore,
    advprefix_target,
    candidate_prefixes_for,
    pareto_frontier,
    select_advprefix,
)

__all__ = [
    "Attack",
    "AttackResult",
    "ArchitectureError",
    "AttackConfig",
    "GCGConfig",
    "PrefillConfig",
    "AttackRegistry",
    "registry",
    "register",
    "get_attack",
    "get_attack_class",
    "list_attacks",
    "GCGAttack",
    # Prefill attack (CHUNK 5-prefill-attack; arXiv:2602.14689).
    "PrefillAttack",
    # nanoGCG-parity hardening (CHUNK 3-gcg-advprefix).
    "PromptSlices",
    "locate_optim_slice",
    "filter_ids",
    "round_trips",
    "token_gradients_onehot",
    "sample_candidates",
    "AttackBuffer",
    "ProbeSamplingConfig",
    # AdvPrefix target source (CHUNK 3-gcg-advprefix).
    "advprefix_target",
    "candidate_prefixes_for",
    "select_advprefix",
    "pareto_frontier",
    "PrefixCandidate",
    "PrefixScore",
    "FIXED_BASELINE_PREFIX",
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
