"""The v0.4 white-box attack registry — ``@register`` decorator + lookup.

ROADMAP §5/§6.0: every attack family is a registered :class:`~injectkit.whitebox.base.Attack`
subclass resolvable by name (HarmBench ``RedTeamingMethod`` / EasyJailbreak-recipe
shape, generalising injectkit's existing transform/defense/attacker registries).

Usage:

    from injectkit.whitebox.registry import register, get_attack, list_attacks

    @register("gcg")
    class GCGAttack(Attack):
        ...

    attack = get_attack("gcg")            # -> a fresh GCGAttack() instance
    cls = get_attack_class("gcg")         # -> the GCGAttack class itself
    names = list_attacks()                # -> ["gcg", ...] sorted

The registry stores the **classes** (not instances) so attacks are constructed at
resolve time; :func:`get_attack` returns a fresh, zero-arg instance (every v0.4
attack is configured at :meth:`Attack.run` time via the typed ``cfg``, so the
constructor takes nothing — the model/config are run arguments, ROADMAP §6.0).

This registry is deliberately separate from the v0.3
:mod:`injectkit.attackers.registry` (the ``--attacker`` factory registry for the
black-box adaptive attackers + the legacy GCG wrapper): the v0.4 ``Attack`` ABC
is a different contract (typed ``run(model, tokenizer, messages, target, cfg,
defense)``), and keeping them separate is what lets v0.4 land without disturbing
the shipped v0.3 CLI wiring.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

from .base import Attack

__all__ = [
    "AttackRegistry",
    "registry",
    "register",
    "get_attack",
    "get_attack_class",
    "list_attacks",
]


_A = TypeVar("_A", bound=Attack)


class AttackRegistry:
    """Name → :class:`~injectkit.whitebox.base.Attack` *class* registry.

    Stores attack classes keyed by their registry name. Resolution constructs a
    fresh instance, so attacks are never shared mutable singletons. Re-registering
    the same name raises unless ``override=True`` — a typo or a double-import that
    would silently shadow an attack is a bug, not a no-op.
    """

    def __init__(self) -> None:
        self._classes: dict[str, Type[Attack]] = {}

    def register(
        self, name: str, cls: Type[_A], *, override: bool = False
    ) -> Type[_A]:
        """Register attack class ``cls`` under ``name`` and return it unchanged.

        Args:
            name: The registry key. Must be non-empty.
            cls: An :class:`~injectkit.whitebox.base.Attack` subclass.
            override: Allow replacing an already-registered name (default False).

        Raises:
            ValueError: if ``name`` is empty, ``cls`` is not an ``Attack``
                subclass, or ``name`` is already registered and ``override`` is
                False.
        """
        if not name:
            raise ValueError("attack registry name must be a non-empty string.")
        if not (isinstance(cls, type) and issubclass(cls, Attack)):
            raise ValueError(
                f"cannot register {cls!r} as attack {name!r}: not an Attack subclass."
            )
        if name in self._classes and not override:
            raise ValueError(
                f"attack {name!r} is already registered "
                f"({self._classes[name].__name__}); pass override=True to replace."
            )
        self._classes[name] = cls
        return cls

    def get_class(self, name: str) -> Type[Attack]:
        """Return the registered class for ``name`` (raises KeyError if absent)."""
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(
                f"unknown attack {name!r}; registered: {sorted(self._classes)}"
            ) from None

    def get(self, name: str) -> Attack:
        """Resolve ``name`` and return a fresh, zero-arg attack instance."""
        return self.get_class(name)()

    def names(self) -> list[str]:
        """All registered attack names, sorted."""
        return sorted(self._classes)


#: The process-wide default white-box attack registry.
registry = AttackRegistry()


def register(
    name: str, *, override: bool = False
) -> Callable[[Type[_A]], Type[_A]]:
    """Class decorator registering an :class:`~injectkit.whitebox.base.Attack`.

    Sets the class's :attr:`~injectkit.whitebox.base.Attack.name` to ``name`` (so
    the registry key and the attack's self-reported name never drift) and wires it
    onto the default :data:`registry`.

        @register("gcg")
        class GCGAttack(Attack):
            ...

    Args:
        name: The registry key (also assigned to ``cls.name``).
        override: Allow replacing an already-registered name (default False).
    """

    def _decorator(cls: Type[_A]) -> Type[_A]:
        cls.name = name
        return registry.register(name, cls, override=override)

    return _decorator


def get_attack(name: str) -> Attack:
    """Resolve ``name`` to a fresh attack instance from the default registry."""
    return registry.get(name)


def get_attack_class(name: str) -> Type[Attack]:
    """Resolve ``name`` to the registered attack class from the default registry."""
    return registry.get_class(name)


def list_attacks() -> list[str]:
    """All registered attack names on the default registry (sorted)."""
    return registry.names()
