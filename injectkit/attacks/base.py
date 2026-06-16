"""The AttackStrategy protocol and the multi-turn attack data model.

An :class:`AttackStrategy` turns one :class:`~injectkit.models.Attack` (and the
per-run canary) into an ordered sequence of :class:`AttackStep` turns to deliver
to a :class:`~injectkit.targets.conversational.ConversationalTarget`. This single
abstraction covers:

* **Single-shot** attacks (the v0.1.0 corpus): one step, one user turn — handled
  by :class:`SingleShotStrategy`, the default.
* **Multi-turn** attacks: crescendo (gradually escalating innocuous->target),
  role-play priming, or many-shot framing delivered as genuine alternating
  turns. These are described by a :class:`MultiTurnAttack` (a reusable template)
  or generated on the fly by a strategy.

The engine (:meth:`injectkit.engine.Engine.run_strategy`) drives a strategy like
this:

    strategy = attack_to_strategy(attack)
    steps = strategy.build(attack, canary)
    # send steps[0..n-1] as scripted history / probes, score the final response

Strategies that also expose the adaptive reply-referencing hooks (``next_turn`` +
``final_step``, e.g.
:class:`~injectkit.attacks.multiturn.CrescendoReplyReferencingStrategy`) are
instead driven turn-by-turn by the engine so each escalation can quote the
target's real prior reply; see
:meth:`injectkit.engine.Engine._deliver_adaptive`.

DEFENSIVE / AUTHORIZED USE ONLY. Every built-in strategy keeps the benign canary
recoverable so success stays a benign-proxy signal, never harmful content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..models import Attack, Severity
from ..targets.conversational import ChatMessage

__all__ = [
    "AttackStep",
    "MultiTurnAttack",
    "AttackStrategy",
    "SingleShotStrategy",
    "StrategyError",
    "attack_to_strategy",
]


class StrategyError(RuntimeError):
    """Raised when a strategy cannot build a valid attack sequence."""


@dataclass
class AttackStep:
    """One turn the strategy wants delivered to the conversational target.

    ``message`` is the chat turn (role + content, canary already rendered).
    ``scored`` marks the step whose *response* the engine should score for
    success — typically the final user turn. Earlier "setup" turns (priming,
    crescendo lead-ins) are unscored and only build context. ``expect_response``
    is True when the engine should actually call the target for this step and
    capture its reply (used as conversation history for later turns); a scripted
    assistant turn the strategy injects verbatim has ``expect_response=False``.
    """

    message: ChatMessage
    #: True if the target's response to this step is the one to evaluate.
    scored: bool = True
    #: True if the engine should send this turn and capture a real response.
    #: False for strategy-scripted assistant turns inserted as fake history.
    expect_response: bool = True
    #: Optional human-readable note for the transcript/report (e.g. "crescendo
    #: lead-in 1/3").
    note: str = ""


@dataclass
class MultiTurnAttack:
    """A reusable multi-turn attack template (crescendo, role-play, etc.).

    This is the multi-turn analogue of a corpus :class:`~injectkit.models.Attack`.
    ``turns`` are the ordered user turns to deliver (each may contain a
    ``{canary}`` placeholder rendered per run). ``scored_turn_index`` selects
    which turn's response is evaluated (default: the last). ``system`` and
    ``success_conditions`` mirror the single-shot Attack so the same detectors
    score the final response.

    A loader for multi-turn corpus YAML (a future builder) constructs these; the
    engine converts each into :class:`AttackStep` objects via a strategy.
    """

    id: str
    technique: str
    name: str
    description: str
    severity: Severity
    #: Ordered user-turn payloads; each may carry a ``{canary}`` placeholder.
    turns: list[str]
    success_conditions: dict[str, object] = field(default_factory=dict)
    system: Optional[str] = None
    #: Which turn's response to score; negative indexes from the end (-1 = last).
    scored_turn_index: int = -1
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_file: Optional[str] = None

    def to_attack(self) -> Attack:
        """Project the *scored* turn into a single-shot :class:`Attack`.

        Lets multi-turn cases reuse the existing detector/scoring/Finding path:
        the scored turn's payload, the shared ``success_conditions``, severity,
        ``system``, technique and metadata become a standard Attack, which the
        engine evaluates against the final response. The earlier turns are still
        delivered as conversation context by the strategy; this projection only
        feeds the detector the data it needs to grade the outcome.
        """
        idx = self.scored_turn_index
        payload = self.turns[idx] if self.turns else ""
        return Attack(
            id=self.id,
            technique=self.technique,
            name=self.name,
            description=self.description,
            severity=self.severity,
            payload=payload,
            success_conditions=dict(self.success_conditions),
            references=list(self.references),
            tags=list(self.tags),
            system=self.system,
            context=None,
            source_file=self.source_file,
        )


@runtime_checkable
class AttackStrategy(Protocol):
    """Produces the ordered turns to deliver for one attack.

    A strategy is pure and deterministic given ``(attack, canary)`` — no network,
    no RNG unless seeded in ``__init__``. It returns the steps; the engine owns
    actually sending them to the conversational target and scoring the result.
    """

    #: Stable identifier shown in reports (e.g. "single_shot", "crescendo").
    name: str

    def build(self, attack: Attack, canary: str) -> list[AttackStep]:
        """Return the ordered :class:`AttackStep` list for this attack run.

        Args:
            attack: The attack to deliver. For multi-turn strategies that need
                the full turn list, the engine passes the
                :meth:`MultiTurnAttack.to_attack` projection plus the strategy is
                constructed with the originating :class:`MultiTurnAttack`; simple
                strategies use only ``attack.payload``.
            canary: The per-run canary to render into every turn's ``{canary}``.

        Returns:
            One or more :class:`AttackStep` objects. Exactly one step should have
            ``scored=True`` (the engine scores that step's response).
        """
        ...


class SingleShotStrategy:
    """The default strategy: deliver the attack as a single scored user turn.

    Wraps a v0.1.0 corpus :class:`~injectkit.models.Attack` so the multi-turn
    engine path can run it unchanged — one step, scored, expecting a response.
    """

    name = "single_shot"

    def build(self, attack: Attack, canary: str) -> list[AttackStep]:
        """Return a single scored user turn rendered with ``canary``."""
        return [
            AttackStep(
                message=ChatMessage(role="user", content=attack.render(canary)),
                scored=True,
                expect_response=True,
                note="single-shot",
            )
        ]


def attack_to_strategy(attack: Attack) -> AttackStrategy:
    """Return the default :class:`SingleShotStrategy` for a corpus attack.

    The engine calls this to obtain a strategy for a plain single-shot attack.
    Multi-turn cases supply their own strategy explicitly (constructed from a
    :class:`MultiTurnAttack`), so this helper only covers the simple default.
    """
    return SingleShotStrategy()
