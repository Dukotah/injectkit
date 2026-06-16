"""Payload transforms — composable obfuscations applied to attack payloads.

A :class:`~injectkit.transforms.base.Transform` rewrites an attack payload
(keeping the canary intact so detection still works) to test whether a wrapper
defeats a target's input filters. Transforms are the building block the
adaptive attacker and the corpus expansion both lean on.

DEFENSIVE / AUTHORIZED USE ONLY. Transforms exist to measure robustness of a
target you own against obfuscated injections — they are not detection-evasion
tooling aimed at third-party systems. The canary stays recoverable so a success
remains a benign-proxy success, never harmful content.
"""

from __future__ import annotations

from .base import (
    Compose,
    Identity,
    Transform,
    TransformError,
    TransformRegistry,
    get_transform,
    list_transforms,
    register_transform,
    registry,
)
from .encoders import (
    Base64Transform,
    HexTransform,
    LeetspeakTransform,
    PayloadSplitting,
    ReversedText,
    Rot13Transform,
    UnicodeHomoglyph,
    ZeroWidthInsertion,
    register_builtin_transforms,
)

__all__ = [
    "Transform",
    "TransformError",
    "TransformRegistry",
    "Compose",
    "Identity",
    "get_transform",
    "list_transforms",
    "register_transform",
    "registry",
    # Encoder/obfuscation transforms
    "Base64Transform",
    "HexTransform",
    "LeetspeakTransform",
    "PayloadSplitting",
    "ReversedText",
    "Rot13Transform",
    "UnicodeHomoglyph",
    "ZeroWidthInsertion",
    "register_builtin_transforms",
]
