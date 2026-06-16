"""The Transform protocol and registry — the contract every transform implements.

A :class:`Transform` rewrites a rendered attack payload into an obfuscated or
restructured variant, while preserving the per-run canary so the heuristic
detector can still recognise a benign-proxy success. Examples a builder will
implement against this contract: base64 / rot13 / leetspeak encoders, unicode
homoglyph or zero-width injectors, role-play wrappers, translation framers, and
payload-splitting "many-part" transforms.

Design contract (every Transform MUST honour):

* **Canary-preserving.** ``apply(payload, canary)`` must return a string from
  which the success marker / canary is still recoverable by the target when the
  injection lands. A transform that mangles the canary beyond recovery breaks
  the benign-proxy measurement and is a bug. The canary is passed explicitly so
  a transform that *encodes* the payload (e.g. base64) can carve the canary out,
  leave it in cleartext, or re-encode it deterministically.
* **Pure & deterministic.** No network, no clock, no RNG by default. If a
  transform needs randomness (e.g. random homoglyph substitution) it must accept
  an explicit ``seed`` in ``__init__`` so runs are reproducible (benchmark
  metadata records the seed).
* **Total.** Never raise on ordinary string input; on an input it cannot handle,
  return the payload unchanged rather than raising. A genuinely unrecoverable
  internal error may raise :class:`TransformError`, which the engine treats as a
  skipped transform.
* **Composable.** Transforms compose left-to-right via :class:`Compose`; each is
  applied to the output of the previous. ``name`` should be stable and unique so
  composed names (``"base64+roleplay"``) are readable in reports.

DEFENSIVE / AUTHORIZED USE ONLY. Transforms measure robustness of a target you
own or are authorised to test. They are not framed for evading third-party
production defences.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "Transform",
    "TransformError",
    "Compose",
    "Identity",
    "TransformRegistry",
    "registry",
    "register_transform",
    "get_transform",
    "list_transforms",
]


class TransformError(RuntimeError):
    """Raised when a transform cannot be applied (engine treats as skipped)."""


@runtime_checkable
class Transform(Protocol):
    """Rewrites an attack payload while keeping the canary recoverable.

    Implementations expose a stable ``name`` and an ``apply`` method. They are
    expected to be pure and deterministic (see module docstring).
    """

    #: Short, stable, unique identifier used in reports and composition names
    #: (e.g. "base64", "rot13", "homoglyph", "roleplay_wrap").
    name: str

    def apply(self, payload: str, canary: str) -> str:
        """Return an obfuscated/restructured variant of ``payload``.

        Args:
            payload: The already-canary-rendered attack payload to transform.
            canary: The per-run canary token embedded in ``payload``. Passed so
                encoders can keep the success marker recoverable (e.g. leave the
                canary in cleartext, or encode it the same way the target will
                decode it).

        Returns:
            The transformed payload string. The canary / success marker must
            remain recoverable when the injection succeeds, so the existing
            heuristic detector still scores a benign-proxy hit.

        Raises:
            TransformError: only on a genuinely unrecoverable internal error;
                ordinary unsupported input should return ``payload`` unchanged.
        """
        ...


class Identity:
    """The no-op transform: returns the payload unchanged. Useful as a baseline.

    The benchmark always includes Identity so every other transform's ASR can be
    compared against the untransformed attack-success-rate.
    """

    name = "identity"

    def apply(self, payload: str, canary: str) -> str:
        """Return ``payload`` unchanged."""
        return payload


class Compose:
    """Compose several transforms left-to-right into one Transform.

    ``Compose(a, b)`` applies ``a`` then ``b`` (``b.apply(a.apply(payload))``).
    The composed ``name`` joins the parts with ``"+"`` for readable reports.
    """

    def __init__(self, *transforms: Transform) -> None:
        self.transforms: list[Transform] = list(transforms)
        self.name = "+".join(t.name for t in self.transforms) or "identity"

    def apply(self, payload: str, canary: str) -> str:
        """Apply each wrapped transform in order, threading the output."""
        out = payload
        for t in self.transforms:
            out = t.apply(out, canary)
        return out


class TransformRegistry:
    """Name -> Transform-factory registry the CLI and benchmark resolve against.

    Builders register a zero-arg (or default-arg) factory under a stable name so
    ``--transform base64,rot13`` on the CLI resolves to concrete instances. The
    registry stores *factories* (callables returning a Transform) rather than
    instances so transforms that take a seed/config can be parameterised later.
    """

    def __init__(self) -> None:
        self._factories: dict[str, "type[Transform] | callable"] = {}

    def register(self, name: str, factory: "type[Transform] | callable") -> None:
        """Register ``factory`` (a class or callable returning a Transform).

        Raises:
            ValueError: if ``name`` is already registered (no silent shadowing).
        """
        if name in self._factories:
            raise ValueError(f"transform {name!r} is already registered")
        self._factories[name] = factory

    def get(self, name: str) -> Transform:
        """Instantiate and return the transform registered under ``name``.

        Raises:
            KeyError: if ``name`` is not registered.
        """
        if name not in self._factories:
            raise KeyError(
                f"unknown transform {name!r}; available: {sorted(self._factories)}"
            )
        return self._factories[name]()

    def names(self) -> list[str]:
        """Return the sorted list of registered transform names."""
        return sorted(self._factories)


#: The process-wide default registry. Builders register their transforms here at
#: import time; the CLI/benchmark resolve ``--transform`` names against it.
registry = TransformRegistry()
# The identity baseline is always available.
registry.register("identity", Identity)


def register_transform(name: str, factory: "type[Transform] | callable") -> None:
    """Register a transform factory on the default :data:`registry`."""
    registry.register(name, factory)


def get_transform(name: str) -> Transform:
    """Resolve a transform by name from the default :data:`registry`."""
    return registry.get(name)


def list_transforms() -> list[str]:
    """List transform names registered on the default :data:`registry`."""
    return registry.names()
