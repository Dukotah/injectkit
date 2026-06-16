"""Named adaptive-attacker registry — pair / tap / autodan / gptfuzzer / gcg.

v0.2.0 shipped one concrete adaptive attacker (``refine``). The research survey
(``docs/RESEARCH.md`` → "Named automated attackers" and "Gradient suffixes")
calls for admitting the documented automated red-teamers by name:

    pair       PAIR — single-rewrite propose/refine (arXiv:2310.08419).
    tap        TAP — tree-of-attacks with pruning (arXiv:2312.02119).
    autodan    AutoDAN — genetic/hierarchical stealthy-prompt search (2310.04451).
    gptfuzzer  GPTFUZZER — mutation-fuzzing of jailbreak templates (2309.10253).
    gcg        GCG — white-box gradient suffix (HF-only; AmpleGCG 2404.07921).

This module freezes the **registry contract** the CLI resolves
``--attacker <name>`` against, parallel to the transform / defense registries.
``pair``/``tap``/``autodan``/``gptfuzzer`` are BLACK-BOX
:class:`~injectkit.attackers.base.AdaptiveAttacker`s (they drive a local attacker
model + the crescendo-style strategies); ``gcg`` is the WHITE-BOX
:class:`~injectkit.attackers.whitebox_base.WhiteBoxGCGAttacker` (HF-only, lazy
torch). All optimise toward the BENIGN canary marker — never harmful content.

The registry stores **factory specs** (a builder callable + a one-line doc),
NOT instances, so an attacker needing a model/config is parameterised at resolve
time. Each concrete attacker module registers its factory here at import time
(``pair``/``tap``/``autodan``/``gptfuzzer``/``gcg`` are all wired); the spec's
``available`` flag stays False only for a name that was declared but never wired,
in which case resolving raises a friendly error.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .base import AdaptiveAttacker, AttackerError

__all__ = [
    "AttackerSpec",
    "AttackerRegistry",
    "registry",
    "register_attacker",
    "get_attacker",
    "list_attackers",
    "NAMED_ATTACKERS",
]


#: ``--attacker`` factory signature. The CLI/engine passes the runtime pieces an
#: adaptive attacker needs (an attacker model and/or a white-box model, plus
#: free-form options like ``max_rounds``); the factory returns a ready
#: :class:`~injectkit.attackers.base.AdaptiveAttacker`. Kept permissive
#: (``**options``) so each attacker takes only what it needs.
AttackerFactory = Callable[..., AdaptiveAttacker]


@dataclass
class AttackerSpec:
    """A registered named attacker: its factory, kind, doc, and availability.

    Stored in the registry instead of an instance so attackers that need a model
    or config are built at resolve time.
    """

    #: Stable ``--attacker`` key (e.g. "pair", "gcg").
    name: str
    #: ``"black_box"`` (model-driven prompt rewriting) or ``"white_box"``
    #: (gradient suffix; HF-only). The CLI uses this to decide which runtime
    #: pieces to hand the factory.
    kind: str
    #: One-line description + primary citation (surfaced by ``injectkit list``).
    doc: str
    #: The builder callable. ``None`` only for a declared-but-unwired spec.
    factory: Optional[AttackerFactory] = None

    @property
    def available(self) -> bool:
        """True once a concrete factory has been registered for this attacker."""
        return self.factory is not None

    def build(self, **options: object) -> AdaptiveAttacker:
        """Instantiate the attacker via its factory.

        All five built-in attackers register a factory at import time, so this
        normally just calls it. The guard remains a safety net for a name that
        was :meth:`AttackerRegistry.declare`-d but never wired with a factory: it
        raises a friendly :class:`AttackerError` rather than a raw ``TypeError``.
        """
        if self.factory is None:
            raise AttackerError(
                f"attacker {self.name!r} ({self.kind}) is declared but has no "
                "registered factory."
            )
        return self.factory(**options)


class AttackerRegistry:
    """Name -> :class:`AttackerSpec` registry the CLI resolves ``--attacker`` on.

    Parallels :class:`~injectkit.transforms.base.TransformRegistry` and the
    defense registry. Pre-seeded with the five named specs; each concrete
    attacker module registers a real factory under its name at import time.
    """

    def __init__(self) -> None:
        self._specs: dict[str, AttackerSpec] = {}

    def declare(self, spec: AttackerSpec) -> None:
        """Add a (possibly factory-less) spec; overwrites a same-name placeholder.

        Declaring a known name with a real factory replaces the placeholder spec,
        which is how a builder marks an attacker available without changing call
        sites. Declaring a brand-new name simply registers it.
        """
        self._specs[spec.name] = spec

    def register(self, name: str, factory: AttackerFactory) -> None:
        """Attach a concrete ``factory`` to the named spec (marks it available).

        Raises:
            KeyError: if ``name`` was never declared (use :meth:`declare` first
                for a brand-new attacker).
        """
        if name not in self._specs:
            raise KeyError(
                f"unknown attacker {name!r}; declare its AttackerSpec first. "
                f"Known: {sorted(self._specs)}"
            )
        spec = self._specs[name]
        self._specs[name] = AttackerSpec(
            name=spec.name, kind=spec.kind, doc=spec.doc, factory=factory
        )

    def get(self, name: str, **options: object) -> AdaptiveAttacker:
        """Resolve and build the attacker registered under ``name``.

        Raises:
            KeyError: if ``name`` is unknown.
            AttackerError: if the attacker is declared but not yet implemented.
        """
        if name not in self._specs:
            raise KeyError(
                f"unknown attacker {name!r}; available: {sorted(self._specs)}"
            )
        return self._specs[name].build(**options)

    def spec(self, name: str) -> AttackerSpec:
        """Return the :class:`AttackerSpec` for ``name`` (raises KeyError if absent)."""
        return self._specs[name]

    def names(self) -> list[str]:
        """All declared attacker names, sorted."""
        return sorted(self._specs)

    def available_names(self) -> list[str]:
        """Only the attacker names that have a concrete factory wired, sorted."""
        return sorted(n for n, s in self._specs.items() if s.available)


#: The five named attackers, declared up front (citations from docs/RESEARCH.md).
#: Each concrete attacker module wires its factory via :func:`register_attacker`
#: at import time.
NAMED_ATTACKERS: tuple[AttackerSpec, ...] = (
    AttackerSpec(
        name="pair",
        kind="black_box",
        doc="PAIR: attacker-model single-rewrite propose/refine (arXiv:2310.08419).",
    ),
    AttackerSpec(
        name="tap",
        kind="black_box",
        doc="TAP: tree-of-attacks with pruning, branch+prune search (arXiv:2312.02119).",
    ),
    AttackerSpec(
        name="autodan",
        kind="black_box",
        doc="AutoDAN: genetic/hierarchical stealthy-prompt search (arXiv:2310.04451).",
    ),
    AttackerSpec(
        name="gptfuzzer",
        kind="black_box",
        doc="GPTFUZZER: mutation-fuzzing of jailbreak templates (arXiv:2309.10253).",
    ),
    AttackerSpec(
        name="gcg",
        kind="white_box",
        doc="GCG: white-box gradient suffix; HF-only, benign target (arXiv:2404.07921).",
    ),
)


#: The process-wide default attacker registry, pre-seeded with the named specs.
registry = AttackerRegistry()
for _spec in NAMED_ATTACKERS:
    registry.declare(_spec)


def register_attacker(name: str, factory: AttackerFactory) -> None:
    """Wire a concrete attacker factory onto the default :data:`registry`.

    Each concrete attacker module calls this at import time, marking the named
    spec available so ``--attacker <name>`` resolves to a real instance.
    """
    registry.register(name, factory)


def get_attacker(name: str, **options: object) -> AdaptiveAttacker:
    """Resolve a named attacker from the default :data:`registry`."""
    return registry.get(name, **options)


def list_attackers() -> list[str]:
    """All declared attacker names on the default :data:`registry` (sorted)."""
    return registry.names()
