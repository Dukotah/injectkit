"""Multi-turn target contracts — ConversationalTarget plus a single-shot adapter.

v0.1.0 targets are single-shot: one prompt in, one :class:`TargetResponse` out
(the :class:`~injectkit.targets.base.Target` protocol). v0.2.0 adds multi-turn
attacks (crescendo, role-play escalation, many-shot delivered as real turns).
Those need a target that can carry a conversation, so this module defines:

* :class:`ChatMessage` — one normalized turn (role + content).
* :class:`ConversationalTarget` — a Protocol with
  ``chat(messages, system) -> TargetResponse``.
* :class:`SingleShotChatAdapter` — wraps any existing single-shot
  :class:`Target` so it satisfies :class:`ConversationalTarget`. It flattens the
  message list into one prompt (a transcript) and forwards to ``Target.send``,
  so every v0.1.0 adapter (http, anthropic, mcp, mock) works in multi-turn flows
  with zero changes. Native multi-turn adapters (e.g. a real Anthropic
  conversation) can implement :class:`ConversationalTarget` directly for true
  turn-by-turn state.

DEFENSIVE / AUTHORIZED USE ONLY. Same posture as the single-shot Target: probe
only endpoints you own or are authorised to test.

Adapters MUST lazy-import any heavy SDK inside their own module (never here), so
importing this contract never pulls in an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence, runtime_checkable

from ..models import TargetResponse
from .base import Target

__all__ = [
    "ChatRole",
    "ChatMessage",
    "ConversationalTarget",
    "SingleShotChatAdapter",
    "as_conversational",
]

#: The roles a chat turn may carry. "system" is usually passed separately via
#: the ``system`` argument, but is allowed in the list for adapters that prefer
#: an inline system turn.
ChatRole = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    """One turn in a multi-turn conversation.

    ``role`` is one of "user", "assistant", "system". ``content`` is the turn
    text (already canary-rendered by the time it reaches a target). This is the
    normalized shape attack strategies emit and conversational targets consume,
    independent of any provider's message schema.
    """

    role: ChatRole
    content: str


@runtime_checkable
class ConversationalTarget(Protocol):
    """A target that can hold a multi-turn conversation.

    Like :class:`~injectkit.targets.base.Target`, implementations must never
    raise on a normal failed request — return a :class:`TargetResponse` with
    ``error`` set instead. ``chat`` returns the response to the *final* turn in
    ``messages`` (the latest user turn); prior turns are conversation history.
    """

    #: Human-readable name shown in reports.
    name: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        system: Optional[str] = None,
    ) -> TargetResponse:
        """Send a conversation and return the response to the latest user turn.

        Args:
            messages: The full ordered conversation so far. The last element is
                the new user turn to be answered; earlier elements are history
                (which may include prior assistant turns the strategy scripted
                or genuinely received).
            system: Optional system prompt to apply for the whole conversation.
                ``None`` means use the target's default/configured system prompt.

        Returns:
            A :class:`TargetResponse` for the final turn. On a refusal set
            ``refused=True``; on a transport/API error set ``error`` rather than
            raising.
        """
        ...


class SingleShotChatAdapter:
    """Adapt a single-shot :class:`Target` to the :class:`ConversationalTarget`.

    Wraps an existing v0.1.0 target so multi-turn attack strategies can drive it
    without the target understanding turns. The conversation is flattened into a
    single transcript prompt (``"User: ...\\nAssistant: ...\\nUser: ..."``) and
    sent via ``Target.send``; the system prompt is forwarded through.

    This keeps every existing adapter usable in multi-turn flows. It is a
    faithful approximation for stateless HTTP/echo endpoints; for true
    turn-by-turn state (where the model genuinely remembers earlier turns)
    prefer a native :class:`ConversationalTarget` implementation.

    Args:
        target: The single-shot target to wrap.
        role_labels: Mapping of role -> transcript label. Defaults to
            ``{"user": "User", "assistant": "Assistant", "system": "System"}``.
        name: Optional override for the display name (defaults to the wrapped
            target's ``name``).
    """

    def __init__(
        self,
        target: Target,
        *,
        role_labels: Optional[dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.target = target
        self.name = name or getattr(target, "name", "target")
        self.role_labels = role_labels or {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }

    def chat(
        self,
        messages: Sequence[ChatMessage],
        system: Optional[str] = None,
    ) -> TargetResponse:
        """Flatten ``messages`` into a transcript and forward to ``Target.send``.

        The final user turn is sent as the prompt with the prior turns rendered
        as a transcript prefix, so a stateless target still sees the full
        conversation context. The wrapped target's normalized
        :class:`TargetResponse` is returned unchanged.
        """
        prompt = self._render_transcript(messages)
        return self.target.send(prompt, system=system, context=None)

    def _render_transcript(self, messages: Sequence[ChatMessage]) -> str:
        """Render the message list into a single ``Role: content`` transcript."""
        lines: list[str] = []
        for m in messages:
            label = self.role_labels.get(m.role, m.role.capitalize())
            lines.append(f"{label}: {m.content}")
        # Cue the model to continue as the assistant.
        assistant_label = self.role_labels.get("assistant", "Assistant")
        lines.append(f"{assistant_label}:")
        return "\n".join(lines)


def as_conversational(target: Target) -> ConversationalTarget:
    """Return ``target`` as a ConversationalTarget.

    If ``target`` already satisfies :class:`ConversationalTarget` (has a
    ``chat`` method) it is returned unchanged; otherwise it is wrapped in a
    :class:`SingleShotChatAdapter`. This is the single helper the engine/strategy
    code should call so it never has to branch on target capability.
    """
    if hasattr(target, "chat") and callable(getattr(target, "chat")):
        return target  # type: ignore[return-value]
    return SingleShotChatAdapter(target)
