"""The AdaptiveAttacker protocol and its local-model-first data model.

An :class:`AdaptiveAttacker` runs a propose/refine loop: an *attacker model*
proposes a candidate attack payload, the engine sends it to the target, a
detector/judge scores the response, and the attacker model refines the payload
using that feedback — repeating until it succeeds or a budget is exhausted. This
is the standard automated red-teaming (PAIR/TAP-style) ASR methodology.

ETHICS — NON-NEGOTIABLE (every implementation MUST honour):

* **Structure, not harm.** The attacker optimises attack *structure* to bypass
  the target's instructions, judged by the benign canary proxy. The objective is
  "make the target emit the benign success marker it was told to withhold", never
  "produce harmful content". The seed attack already carries the benign canary;
  the attacker rewrites the framing around it, keeping the canary recoverable.
* **Local-model-first / offline.** The default attacker model is local (Ollama or
  a local HF model) and needs no API key. The model client is lazy-imported with
  a friendly error if its optional dependency is missing. Unit tests pass a
  scripted :class:`AttackerModel` stub and make no network/model calls.
* **Bounded.** Every run has an explicit ``max_rounds`` budget so it terminates.

DEFENSIVE / AUTHORIZED USE ONLY. Run the adaptive attacker only against targets
you own or are authorised to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..models import Attack, AttackResult

__all__ = [
    "AttackerModel",
    "AttackerTranscriptStep",
    "AttackerResult",
    "AdaptiveAttacker",
    "AttackerError",
]


class AttackerError(RuntimeError):
    """Raised when the adaptive attacker cannot run (e.g. model unavailable)."""


@runtime_checkable
class AttackerModel(Protocol):
    """The local model that proposes and refines attack payloads.

    This is the generation side of the loop, kept separate from the *target*
    (the thing under test) and the *judge* (the thing that scores). The default
    implementation wraps a local model (Ollama / HF) and lazy-imports its SDK;
    tests provide a scripted stub.

    Implementations must be safe to call repeatedly and should never raise on an
    ordinary prompt — return a best-effort string (or empty) instead.
    """

    #: Stable identifier for the attacker model (e.g. "ollama:llama3.1").
    name: str

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        """Return the model's completion for ``prompt``.

        Args:
            prompt: The instruction asking the model to propose or refine an
                attack payload (the :class:`AdaptiveAttacker` builds this from the
                seed attack and the latest target feedback).
            system: Optional system prompt steering the attacker model.

        Returns:
            The proposed/refined attack payload text. The implementation should
            keep the benign ``{canary}`` / success-marker placeholder intact so
            the result is still a benign-proxy attack.
        """
        ...


@dataclass
class AttackerTranscriptStep:
    """One round of the propose/refine loop, recorded for the report.

    Captures the candidate payload the attacker model proposed that round and
    the engine's scored :class:`~injectkit.models.AttackResult` for it, so the
    full optimisation trajectory is reproducible and auditable.
    """

    round: int
    #: The candidate payload the attacker model proposed this round.
    candidate_payload: str
    #: The scored result of sending that candidate to the target.
    result: AttackResult
    #: Optional note (e.g. the attacker model's stated rationale for the refine).
    rationale: str = ""


@dataclass
class AttackerResult:
    """The outcome of a full adaptive attack run against one seed attack.

    ``best_result`` is the highest-scoring round (a success if one was found,
    else the closest attempt). ``succeeded`` is True if any round succeeded.
    ``transcript`` is every round in order. ``rounds_used`` and ``seed_attack``
    plus the attacker-model name make the run reproducible in benchmark metadata.
    """

    seed_attack: Attack
    succeeded: bool
    best_result: AttackResult
    transcript: list[AttackerTranscriptStep] = field(default_factory=list)
    rounds_used: int = 0
    #: Name of the attacker model that drove this run (for metadata).
    attacker_model: str = ""

    @property
    def best_payload(self) -> str:
        """The candidate payload of the best round (most successful attempt)."""
        if self.transcript:
            # The transcript step whose result matches best_result, else the last.
            for step in self.transcript:
                if step.result is self.best_result:
                    return step.candidate_payload
            return self.transcript[-1].candidate_payload
        return self.best_result.attack.render(self.best_result.canary)


@runtime_checkable
class AdaptiveAttacker(Protocol):
    """Runs a bounded propose/refine loop to bypass a target's instructions.

    The orchestration contract the engine/CLI builder implements against:

        attacker = SomeAdaptiveAttacker(model=local_model, max_rounds=5)
        result = attacker.run(seed_attack, target, detectors)

    where ``target`` is a Target / ConversationalTarget and ``detectors`` score
    each round (typically the heuristic detector, optionally the judge). The
    attacker must respect its ``max_rounds`` budget and stop early on success.
    """

    #: Stable identifier for this attacker strategy (e.g. "pair", "refine").
    name: str

    #: Hard cap on propose/refine rounds, so every run terminates.
    max_rounds: int

    def run(
        self,
        seed_attack: Attack,
        target: object,
        detectors: object,
    ) -> AttackerResult:
        """Optimise ``seed_attack`` against ``target`` and return the result.

        Args:
            seed_attack: The starting benign-canary attack to refine. Its canary
                placeholder and success conditions are preserved across rounds so
                scoring stays a benign-proxy measurement.
            target: The target under test (a
                :class:`~injectkit.targets.base.Target` or
                :class:`~injectkit.targets.conversational.ConversationalTarget`).
            detectors: The detector(s) used to score each round's response
                (a sequence honouring the
                :class:`~injectkit.evaluators.base.Detector` protocol).

        Returns:
            An :class:`AttackerResult` with the best round, success flag, and the
            full transcript. Must terminate within ``max_rounds`` rounds.

        Raises:
            AttackerError: only on an unrecoverable setup problem (e.g. the
                attacker model's optional dependency is missing). Per-round
                target/judge faults are captured into the transcript, never
                raised.
        """
        ...
