"""The Detector protocol — the contract every detector implements.

A Detector looks at one attack and the target's response to that attack and
decides whether the injection succeeded, returning a
:class:`~injectkit.models.DetectorVerdict`. Detectors are cheap and offline by
default (the heuristics module: marker/canary echo, refusal detection,
system-prompt-leak markers, regex success conditions). The optional LLM judge
is also a Detector — it just happens to call out to a model.

Detectors receive the per-run canary so they can check for marker echoes
without re-deriving it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Attack, DetectorVerdict, TargetResponse

__all__ = ["Detector"]


@runtime_checkable
class Detector(Protocol):
    """Evaluates whether an injection attack succeeded.

    Implementations should be pure (no side effects) and must not raise on
    ordinary inputs — return a verdict with ``success=False`` if they cannot
    determine success.
    """

    #: Short identifier used in verdicts and reports (e.g. "heuristics").
    name: str

    def evaluate(
        self,
        attack: Attack,
        response: TargetResponse,
        canary: str,
    ) -> DetectorVerdict:
        """Judge one (attack, response) pair.

        Args:
            attack: The attack that was sent, including its
                ``success_conditions`` rules.
            response: The target's normalized response.
            canary: The per-run unique marker substituted into the payload, so
                the detector can look for ``INJECTOK-<canary>`` style echoes.

        Returns:
            A :class:`DetectorVerdict` with ``detector`` set to ``self.name``.
        """
        ...
