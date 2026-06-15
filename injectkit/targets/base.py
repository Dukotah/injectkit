"""The Target protocol — the contract every target adapter implements.

A Target is anything injectkit can send a prompt to and read a reply from: an
HTTP chat endpoint, the Anthropic Messages API, an MCP server/agent, or the
built-in MockTarget used in tests. Adapters normalize their provider's response
into a :class:`~injectkit.models.TargetResponse` so the engine and detectors
are provider-agnostic.

Adapters MUST lazy-import their heavy SDK (anthropic, mcp, httpx) inside their
own module so that importing injectkit's core never pulls in optional deps.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..models import TargetResponse

__all__ = ["Target"]


@runtime_checkable
class Target(Protocol):
    """A destination injectkit can probe.

    Implementations should be safe to call repeatedly (the engine sends one
    request per attack) and should never raise on a normal failed request —
    instead, return a :class:`TargetResponse` with ``error`` set so the scan
    can continue and report the failure.
    """

    #: Human-readable name shown in reports (e.g. "anthropic:claude-opus-4-8").
    name: str

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Send one attack prompt and return a normalized response.

        Args:
            prompt: The rendered attack payload (canary already substituted).
            system: Optional system prompt to apply for this request. When the
                attack carries its own ``system`` (e.g. a fake secret to leak),
                the engine passes it here. ``None`` means use the target's
                default/configured system prompt.
            context: Optional extra context to inject as untrusted data — used
                for indirect-injection attacks that simulate a retrieved
                document or tool output. Adapters that have no notion of
                separate context may prepend it to the prompt.

        Returns:
            A :class:`TargetResponse`. On a refusal, set ``refused=True``
            (a refusal means the target defended successfully). On a transport
            or API error, set ``error`` to a short message rather than raising.
        """
        ...
