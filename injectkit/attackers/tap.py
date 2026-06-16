"""TAP — Tree-of-Attacks with Pruning, a branch-and-prune adaptive attacker.

This module ships the concrete
:class:`~injectkit.attackers.base.AdaptiveAttacker` named ``"tap"``. TAP
(Mehrotra et al., "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically",
arXiv:2312.02119; see ``docs/RESEARCH.md`` → "Named automated attackers") extends
the single-rewrite PAIR loop (arXiv:2310.08419) into a *tree-of-thought* search:

* **Branch.** Each surviving attack prompt is a tree node. At every depth the
  attacker model proposes ``branching_factor`` distinct refined children from each
  leaf, exploring several framings in parallel rather than one.
* **Prune off-topic (phase 1).** Before spending a (costly) target query, each
  child is screened for *on-topic-ness* — does it still pursue the benign
  objective (keep the per-run marker recoverable)? Off-topic children are pruned
  cheaply, exactly as TAP prunes branches an evaluator deems unrelated to the
  goal.
* **Query + score.** The on-topic survivors are sent to the target and scored by
  the supplied detector(s) via the shared scoring core, producing a benign-proxy
  success/confidence per node.
* **Prune low-score (phase 2).** Only the top ``width`` highest-scoring nodes
  survive to seed the next depth, keeping the frontier bounded (TAP's width cap).
* **Bounded.** The search stops on the first success or when ``max_depth`` is
  reached, so every run terminates within an explicit budget.

ETHICS — NON-NEGOTIABLE (this module enforces all of these):

* **Structure, not harm.** The tree optimises attack *structure* to bypass the
  target's instructions, judged solely by the benign canary proxy. Every node's
  objective is "make the target emit the marker ``<trigger><canary>`` it was told
  to withhold" — never "produce harmful content". The off-topic pruner is itself a
  *benign-objective* check: a candidate is "on topic" iff it still asks for the
  benign marker (:func:`~injectkit.attackers.adaptive.ensure_canary` re-anchors
  any child that drifts), so the whole search stays a benign-proxy measurement.
* **Local-model-first / offline.** The default attacker model is the local
  :class:`~injectkit.attackers.adaptive.OllamaAttackerModel` (no API key); its
  optional dependency is lazy-imported by that class. Unit tests pass a scripted
  ``AttackerModel`` stub and make no network/model calls.
* **Bounded.** ``max_depth`` × ``branching_factor`` × ``width`` are all explicit
  caps, and the loop stops early on success, so the search always terminates.

DEFENSIVE / AUTHORIZED USE ONLY. Run TAP only against targets you own or are
explicitly authorised to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..engine import make_canary
from ..evaluators.base import Detector
from ..evaluators.heuristics import DEFAULT_TRIGGER, HeuristicDetector
from ..models import Attack, AttackResult, TargetResponse
from ..targets.conversational import ConversationalTarget, as_conversational
from .adaptive import RefineAttacker, RefinePromptBuilder, ensure_canary, extract_payload
from .base import (
    AttackerError,
    AttackerModel,
    AttackerResult,
    AttackerTranscriptStep,
)

__all__ = [
    "is_on_topic",
    "TAPNode",
    "TAPAttacker",
    "make_tap_attacker",
]


# --------------------------------------------------------------------------- #
# Off-topic pruning — the benign-objective screen (TAP phase 1)
# --------------------------------------------------------------------------- #


def is_on_topic(candidate: str, canary: str, *, trigger: str = DEFAULT_TRIGGER) -> bool:
    """Return whether ``candidate`` still pursues the benign marker objective.

    TAP prunes, *before* querying the target, any branch an evaluator judges
    off-topic / unrelated to the attack goal. Here the goal is strictly the
    benign-canary proxy, so "on topic" means the candidate still asks the target
    to emit the exact per-run marker ``<trigger><canary>``. A child that dropped
    the marker (and so could only be scored as a free-form, potentially harmful
    request) is off-topic and pruned — keeping the search a benign-proxy
    measurement end to end.

    Note that :func:`~injectkit.attackers.adaptive.ensure_canary` re-anchors a
    dropped marker before this check runs in the loop, so in practice this prunes
    only genuinely empty/degenerate candidates; it is kept explicit so the
    branch-and-prune structure (and its safety rationale) is faithful and
    testable.

    Args:
        candidate: The refined child payload proposed by the attacker model
            (after marker re-anchoring).
        canary: The per-run canary token for this node.
        trigger: The marker prefix the detector looks for (default ``INJECTOK-``).

    Returns:
        ``True`` if the rendered marker ``<trigger><canary>`` is recoverable in
        ``candidate``, ``False`` otherwise (prune it).
    """
    marker = f"{trigger}{canary}"
    rendered = (candidate or "").replace("{canary}", canary)
    return marker in rendered


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #


@dataclass
class TAPNode:
    """One node in the attack tree: a scored candidate plus its lineage.

    The frontier at each depth is a list of :class:`TAPNode`. ``result`` holds the
    benign-proxy score for the node's candidate (after it was queried against the
    target); ``score`` is the sortable key the width-pruner ranks on (a success
    outranks any non-success, ties broken by confidence). ``parent`` / ``depth``
    record the lineage so the winning path is reconstructable in the transcript.
    """

    #: The (marker-anchored) candidate payload this node sent to the target.
    candidate: str
    #: The per-run canary used to score this node.
    canary: str
    #: Tree depth (root children are depth 1).
    depth: int
    #: The scored result of querying the target with ``candidate`` (None until queried).
    result: Optional[AttackResult] = None
    #: The parent node this child was branched from (None for depth-1 nodes).
    parent: Optional["TAPNode"] = field(default=None, repr=False)

    @property
    def succeeded(self) -> bool:
        """True if this node's queried result is a benign-proxy success."""
        return bool(self.result is not None and self.result.success)

    @property
    def score(self) -> tuple[int, float]:
        """Sort key for width-pruning: (success-flag, confidence).

        A success (1, …) always outranks a non-success (0, …); within the same
        success bucket, higher detector confidence wins. Larger is better, so the
        pruner keeps the top ``width`` by ``score`` descending.
        """
        if self.result is None:
            return (0, 0.0)
        return (1 if self.result.success else 0, self.result.confidence)


# --------------------------------------------------------------------------- #
# The TAP attacker
# --------------------------------------------------------------------------- #


class TAPAttacker(RefineAttacker):
    """Tree-of-Attacks with Pruning — a bounded branch-and-prune adaptive attacker.

    Implements the :class:`~injectkit.attackers.base.AdaptiveAttacker` protocol by
    extending :class:`~injectkit.attackers.adaptive.RefineAttacker` (reusing its
    target-send / detector-scoring / round-attack helpers) with a tree search:

    1. **Seed.** Build a root frontier by asking the attacker model for
       ``branching_factor`` initial framings of the seed attack.
    2. **For each depth (up to ``max_depth``):**

       a. *Branch* — from every node in the current frontier, propose
          ``branching_factor`` refined children using that node's target reaction
          as feedback (PAIR-style refine prompt).
       b. *Prune off-topic* — drop children that no longer pursue the benign
          marker objective (:func:`is_on_topic`); marker re-anchoring runs first
          so only degenerate candidates are dropped.
       c. *Query + score* — send each surviving child to the target and score it.
       d. *Stop on success* — if any child succeeded, return immediately.
       e. *Prune to width* — keep only the top ``width`` children by score as the
          next frontier.

    The optimisation objective is strictly the benign canary proxy: a node
    "succeeds" iff the target emits the per-run marker. No harmful content is ever
    requested or rewarded, and the off-topic pruner enforces that invariant.

    Args:
        model: The local attacker model proposing candidates (the
            :class:`~injectkit.attackers.base.AttackerModel` protocol). Tests pass
            a scripted stub; production passes
            :class:`~injectkit.attackers.adaptive.OllamaAttackerModel`.
        max_depth: Hard cap on tree depth (propose/refine rounds); must be >= 1.
            Mapped onto the protocol's ``max_rounds`` so the run is bounded and
            benchmark metadata is uniform across attackers.
        branching_factor: Children proposed per node per depth (must be >= 1).
        width: Maximum frontier size kept after each depth's score-pruning (the
            beam width; must be >= 1).
        detectors: Optional detectors to score each node. Defaults to a single
            offline :class:`~injectkit.evaluators.heuristics.HeuristicDetector`.
            The detectors passed to :meth:`run` override this.
        use_judge: Whether judge verdicts take precedence when scoring (passed
            through to the shared scoring core).
        prompt_builder: Strategy object building the attacker-model prompts.
            Defaults to a :class:`~injectkit.attackers.adaptive.RefinePromptBuilder`.
        canary_factory: Callable returning a fresh canary per node. Injectable for
            deterministic tests.
        trigger: The success-marker prefix; kept in sync with the prompt builder.
        name: Stable identifier for this attacker strategy (default ``"tap"``).
    """

    def __init__(
        self,
        model: AttackerModel,
        *,
        max_depth: int = 3,
        branching_factor: int = 3,
        width: int = 5,
        detectors: Optional[Sequence[Detector]] = None,
        use_judge: bool = False,
        prompt_builder: Optional[RefinePromptBuilder] = None,
        canary_factory=make_canary,
        trigger: str = DEFAULT_TRIGGER,
        name: str = "tap",
    ) -> None:
        if max_depth < 1:
            raise AttackerError("max_depth must be >= 1.")
        if branching_factor < 1:
            raise AttackerError("branching_factor must be >= 1.")
        if width < 1:
            raise AttackerError("width must be >= 1.")
        # max_rounds (the protocol budget) is the tree depth.
        super().__init__(
            model,
            max_rounds=max_depth,
            detectors=detectors,
            use_judge=use_judge,
            prompt_builder=prompt_builder,
            canary_factory=canary_factory,
            trigger=trigger,
            name=name,
        )
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.width = width

    # ------------------------------------------------------------------ public

    def run(
        self,
        seed_attack: Attack,
        target: object,
        detectors: object,
    ) -> AttackerResult:
        """Run the bounded tree-of-attacks search against ``target``.

        Args:
            seed_attack: The benign-canary attack to refine. Its
                ``success_conditions`` and ``system`` are carried into every node
                so scoring stays a benign-proxy measurement.
            target: A :class:`~injectkit.targets.base.Target` or
                :class:`~injectkit.targets.conversational.ConversationalTarget`,
                adapted via
                :func:`~injectkit.targets.conversational.as_conversational`.
            detectors: The detector(s) to score each node (a sequence honouring
                the :class:`~injectkit.evaluators.base.Detector` protocol). If
                falsy, this attacker's configured detectors are used.

        Returns:
            An :class:`AttackerResult` whose ``transcript`` records every queried
            node in visit order (one :class:`AttackerTranscriptStep` each, its
            ``round`` set to the node's tree depth), ``best_result`` is the
            highest-scoring node, and ``succeeded`` is True iff any node emitted
            the benign marker. Always terminates within ``max_depth`` depths.

        Raises:
            AttackerError: only on unrecoverable setup (e.g. the attacker model's
                optional dependency is missing). Per-node target/detector faults
                are captured into the transcript, never raised.
        """
        active_detectors = self._coerce_detectors(detectors)
        conv: ConversationalTarget = as_conversational(target)  # type: ignore[arg-type]
        system_prompt = self.prompt_builder.system_prompt()

        transcript: list[AttackerTranscriptStep] = []
        best: Optional[AttackResult] = None

        # Depth 1: seed the root frontier with initial framings (no feedback yet).
        nodes = self._expand_root(
            seed_attack, conv, system_prompt, active_detectors, transcript
        )
        best = self._update_best(nodes, best)
        winner = self._first_success(nodes)
        # Width-prune the root frontier before any deeper expansion (the beam cap
        # applies at every depth, including the root branching).
        frontier = self._prune_to_width(nodes)
        depth = 1

        # Depths 2..max_depth: branch each leaf, prune off-topic, query, prune width.
        while winner is None and depth < self.max_depth and frontier:
            depth += 1
            children = self._expand_frontier(
                seed_attack,
                frontier,
                depth,
                conv,
                system_prompt,
                active_detectors,
                transcript,
            )
            if not children:
                # Every branch pruned/degenerate — nothing left to explore.
                break
            best = self._update_best(children, best)
            winner = self._first_success(children)
            if winner is not None:
                break
            # Width-prune: keep only the top ``width`` nodes for the next depth.
            frontier = self._prune_to_width(children)

        succeeded = winner is not None
        if winner is not None:
            best = winner.result
        elif best is None:
            # Every node was pruned off-topic before any query — fall back to the
            # last recorded (pruned) result so the report still has a best round.
            best = transcript[-1].result if transcript else self._pruned_result(
                seed_attack, "", self.canary_factory()
            )

        # transcript is never empty: depth 1 always proposes >= 1 node.
        assert best is not None  # for type-checkers; always set above
        return AttackerResult(
            seed_attack=seed_attack,
            succeeded=succeeded,
            best_result=best,
            transcript=transcript,
            rounds_used=depth,
            attacker_model=getattr(self.model, "name", "unknown"),
        )

    # ----------------------------------------------------------------- helpers

    def _expand_root(
        self,
        seed_attack: Attack,
        conv: ConversationalTarget,
        system_prompt: str,
        detectors: list[Detector],
        transcript: list[AttackerTranscriptStep],
    ) -> list[TAPNode]:
        """Build, screen, query, and score the depth-1 root frontier.

        Each child is an *initial* framing of the seed attack (no target feedback
        exists yet), so the initial proposal prompt is used. Off-topic children
        are pruned before any target query, exactly as at deeper levels.
        """
        nodes: list[TAPNode] = []
        for _ in range(self.branching_factor):
            canary = self.canary_factory()
            instruction = self.prompt_builder.initial_prompt(seed_attack, canary)
            node = self._propose_and_query(
                seed_attack,
                instruction,
                system_prompt,
                canary,
                depth=1,
                parent=None,
                conv=conv,
                detectors=detectors,
                transcript=transcript,
            )
            if node is not None:
                nodes.append(node)
        return nodes

    def _expand_frontier(
        self,
        seed_attack: Attack,
        frontier: list[TAPNode],
        depth: int,
        conv: ConversationalTarget,
        system_prompt: str,
        detectors: list[Detector],
        transcript: list[AttackerTranscriptStep],
    ) -> list[TAPNode]:
        """Branch every node in ``frontier`` into refined, screened, scored children.

        For each parent node, ``branching_factor`` children are proposed using a
        PAIR-style refine prompt seeded with that parent's target reaction, then
        screened for on-topic-ness and (if they survive) queried + scored.
        """
        children: list[TAPNode] = []
        for parent in frontier:
            assert parent.result is not None  # only queried nodes enter the frontier
            previous_response = parent.result.response
            for _ in range(self.branching_factor):
                canary = self.canary_factory()
                instruction = self.prompt_builder.refine_prompt(
                    seed_attack, canary, parent.candidate, previous_response
                )
                node = self._propose_and_query(
                    seed_attack,
                    instruction,
                    system_prompt,
                    canary,
                    depth=depth,
                    parent=parent,
                    conv=conv,
                    detectors=detectors,
                    transcript=transcript,
                )
                if node is not None:
                    children.append(node)
        return children

    def _propose_and_query(
        self,
        seed_attack: Attack,
        instruction: str,
        system_prompt: str,
        canary: str,
        *,
        depth: int,
        parent: Optional[TAPNode],
        conv: ConversationalTarget,
        detectors: list[Detector],
        transcript: list[AttackerTranscriptStep],
    ) -> Optional[TAPNode]:
        """Propose one child, prune if off-topic, else query + score it.

        The pipeline for a single node:

        1. Ask the attacker model for a candidate (defensively — a flaky round
           cannot abort the search).
        2. Re-anchor the benign marker
           (:func:`~injectkit.attackers.adaptive.ensure_canary`) so the candidate
           stays a benign-proxy attack.
        3. *Off-topic prune* — if the marker is still not recoverable, drop the
           node (return ``None``) without spending a target query.
        4. Query the target and score the response; wrap it in a
           :class:`TAPNode` and record a transcript step.

        Returns:
            The queried :class:`TAPNode`, or ``None`` if the child was pruned as
            off-topic.
        """
        raw, rationale = self._generate(instruction, system_prompt)
        candidate = ensure_canary(extract_payload(raw), canary, trigger=self.trigger)

        # Phase-1 pruning: a candidate that still does not carry the benign marker
        # is off-topic for the benign-proxy objective — prune before querying.
        if not is_on_topic(candidate, canary, trigger=self.trigger):
            transcript.append(
                AttackerTranscriptStep(
                    round=depth,
                    candidate_payload=candidate,
                    result=self._pruned_result(seed_attack, candidate, canary),
                    rationale=f"{rationale}; pruned off-topic (marker not recoverable)",
                )
            )
            return None

        response = self._send(conv, seed_attack, candidate, canary)
        result = self._score(seed_attack, candidate, canary, response, detectors)
        transcript.append(
            AttackerTranscriptStep(
                round=depth,
                candidate_payload=candidate,
                result=result,
                rationale=rationale,
            )
        )
        return TAPNode(
            candidate=candidate,
            canary=canary,
            depth=depth,
            result=result,
            parent=parent,
        )

    def _pruned_result(
        self, seed_attack: Attack, candidate: str, canary: str
    ) -> AttackResult:
        """A non-success :class:`AttackResult` recording an off-topic-pruned node.

        Pruned nodes are never queried, so this carries an error
        :class:`~injectkit.models.TargetResponse` (the candidate never reached the
        target) and an empty verdict list — it exists only so the transcript is a
        faithful, auditable record of every branch the search considered.
        """
        scored_attack = self._round_attack(seed_attack, candidate)
        return AttackResult(
            attack=scored_attack,
            canary=canary,
            response=TargetResponse(
                text="", error="pruned off-topic before query (benign marker dropped)"
            ),
            verdicts=[],
            success=False,
        )

    def _prune_to_width(self, nodes: list[TAPNode]) -> list[TAPNode]:
        """Phase-2 pruning: keep the top ``width`` nodes by score (beam width).

        Sorts ``nodes`` by :attr:`TAPNode.score` descending (success beats
        non-success, ties by confidence) and returns the leading ``width`` as the
        next depth's frontier. Stable for the tail so equal-score nodes keep their
        proposal order, making the search deterministic given a scripted model.
        """
        ranked = sorted(nodes, key=lambda n: n.score, reverse=True)
        return ranked[: self.width]

    @staticmethod
    def _first_success(nodes: list[TAPNode]) -> Optional[TAPNode]:
        """Return the first node in ``nodes`` that succeeded, or None."""
        for node in nodes:
            if node.succeeded:
                return node
        return None

    def _update_best(
        self, nodes: list[TAPNode], best: Optional[AttackResult]
    ) -> Optional[AttackResult]:
        """Fold ``nodes`` into the running best :class:`AttackResult`.

        Reuses :meth:`RefineAttacker._is_better` (success outranks non-success,
        ties by confidence) so the best round selection matches the single-rewrite
        attacker exactly.
        """
        for node in nodes:
            if node.result is None:
                continue
            if self._is_better(node.result, best):
                best = node.result
        return best


# --------------------------------------------------------------------------- #
# Registry factory — wire `tap` onto the named-attacker registry
# --------------------------------------------------------------------------- #


def make_tap_attacker(
    model: Optional[AttackerModel] = None,
    **options: object,
) -> TAPAttacker:
    """Factory for the named-attacker registry's ``tap`` spec.

    Builds a ready :class:`TAPAttacker` from the runtime pieces the CLI/engine
    hand a black-box attacker factory. A local attacker ``model`` is required (TAP
    is model-driven); the remaining ``options`` (e.g. ``max_depth``,
    ``branching_factor``, ``width``, ``detectors``, ``trigger``) are forwarded to
    the constructor.

    Args:
        model: The local attacker model. Required — TAP cannot run without one.
        **options: Forwarded :class:`TAPAttacker` constructor keyword arguments.

    Returns:
        A configured :class:`TAPAttacker`.

    Raises:
        AttackerError: if no attacker ``model`` was supplied.
    """
    if model is None:
        raise AttackerError(
            "the 'tap' attacker needs a local attacker model "
            "(pass model=OllamaAttackerModel(...) or a stub). TAP is "
            "model-driven and cannot run without one."
        )
    return TAPAttacker(model, **options)  # type: ignore[arg-type]


# Wire the concrete `tap` factory onto the named-attacker registry at import
# time, marking the pre-declared spec available. Idempotent: re-importing simply
# re-registers the same factory over the placeholder.
from .registry import register_attacker as _register_attacker  # noqa: E402

_register_attacker("tap", make_tap_attacker)
