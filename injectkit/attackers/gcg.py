"""GCG / AmpleGCG — white-box greedy-coordinate-gradient suffix attacker.

Concrete, benign-canary implementation of the GCG family from the research survey
(``docs/RESEARCH.md`` → "Gradient suffixes (GCG family)", row ``gcg``; GCG /
AmpleGCG arXiv:2404.07921, Mask-GCG arXiv:2509.06350; AdvBench / original GCG
arXiv:2307.15043). GCG optimises an adversarial *suffix* using the **target
model's gradients** so the model emits a chosen string. White-box access (logits +
embedding gradients) ⇒ this is a **HuggingFace-target-ONLY** attacker, distinct
from the black-box PAIR/TAP propose/refine loops.

This module ships :class:`GCGSuffixAttacker`, a concrete
:class:`~injectkit.attackers.base.AdaptiveAttacker` built on the v0.3
:class:`~injectkit.attackers.whitebox_base.WhiteBoxGCGAttacker` base. It implements
the greedy-coordinate-gradient inner loop entirely through the
:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam, so it is driven by
a real in-process HF model in production and by the offline ``StubWhiteBoxModel``
(fake logits/grads) in tests — at most a trivial 1-step path, never real
optimisation or a model download.

ETHICS — NON-NEGOTIABLE (this module + every reviewer enforces):

* **Benign target string — robustness probe, NEVER harm.** The optimisation
  objective is to make a **LOCAL** white-box model emit the BENIGN canary marker
  ``<trigger><canary>`` it was told to withhold. The default
  :class:`~injectkit.attackers.whitebox_base.GCGConfig.target_string` is that
  benign marker. No harmful AdvBench/AmpleGCG suffix is ever set as a target and
  none is bundled in the repo.
* **No bundled harmful suffix artifacts.** GCG/AmpleGCG papers publish
  harmful-optimised universal suffixes. injectkit ships ZERO of them. The optional
  :func:`load_amplegcg_suffixes` helper loads published artifacts ONLY through the
  gated research loader (``injectkit.research`` → opt-in acknowledgment, official
  source, never redistributed); without acknowledgment it raises. Even when
  loaded, such a suffix is only ever *appended as initial filler* whose objective
  the optimiser **re-points at the benign marker** — never at the harmful string.
* **White-box ⇒ HF-only, lazy/heavy.** ``torch`` + ``transformers`` are
  lazy-imported (via
  :func:`~injectkit.attackers.whitebox_base.import_torch_transformers`) only when a
  real HF model is wired; constructing the attacker imports nothing heavy. GCG is
  COMPUTE-HEAVY — real runs want a GPU. Tests inject ``StubWhiteBoxModel`` and need
  neither dependency.
* **Bounded / stub-testable.** ``config.max_steps`` is honoured exactly; the loop
  stops early on the first benign-marker success. Tests run ``max_steps=1`` with
  fake gradients and never touch real torch ops.

DEFENSIVE / AUTHORIZED USE ONLY. Run GCG only against a local model you own.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, List, Optional, Sequence

from ..evaluators.base import Detector
from ..evaluators.heuristics import DEFAULT_TRIGGER, HeuristicDetector
from ..evaluators.scoring import score_verdicts
from ..models import Attack, AttackResult, DetectorVerdict, TargetResponse
from ..targets.conversational import (
    ChatMessage,
    ConversationalTarget,
    as_conversational,
)
from .base import (
    AttackerError,
    AttackerResult,
    AttackerTranscriptStep,
)
from .whitebox_base import (
    GCGConfig,
    GCGStep,
    WhiteBoxGCGAttacker,
    WhiteBoxModel,
)

__all__ = [
    "DEFAULT_INIT_SUFFIX",
    "GCGSuffixAttacker",
    "load_amplegcg_suffixes",
    "make_gcg_attacker",
]


#: A benign filler suffix the optimiser starts from — a row of harmless ``! ``
#: tokens, exactly as the GCG paper initialises its adversarial suffix. The
#: optimiser mutates these tokens; nothing about the seed is harmful.
DEFAULT_INIT_SUFFIX = " ".join(["!"] * 20)


class GCGSuffixAttacker(WhiteBoxGCGAttacker):
    """White-box GCG suffix optimiser with a strictly benign (canary) objective.

    Implements the greedy-coordinate-gradient loop on top of
    :class:`~injectkit.attackers.whitebox_base.WhiteBoxGCGAttacker`. Each step it
    (1) reads the gradient of the *benign* target loss w.r.t. the one-hot suffix
    tokens via :meth:`WhiteBoxModel.token_gradients`, (2) takes the ``top_k``
    most-promising replacement token ids per suffix slot, (3) samples a
    ``batch_size`` batch of single-token swaps, (4) keeps the swap with the lowest
    :meth:`WhiteBoxModel.target_loss`, and (5) records a
    :class:`~injectkit.attackers.whitebox_base.GCGStep`. It stops at
    ``config.max_steps`` or as soon as the (benign) marker is emitted.

    Everything goes through the :class:`WhiteBoxModel` seam, so the loop runs
    against the offline ``StubWhiteBoxModel`` with fake gradients in tests and a
    real in-process HF model in production. The objective is ALWAYS the benign
    per-run marker ``<trigger><canary>``; no harmful target string is set.

    Args:
        model: The white-box model seam
            (:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel`). Production
            wraps an in-process HF causal-LM; tests inject ``StubWhiteBoxModel``.
        config: The :class:`~injectkit.attackers.whitebox_base.GCGConfig`
            (steps/suffix length/top-k/batch/seed). Defaults are tiny and
            test-safe. ``max_steps`` must be >= 1.
        detectors: Optional detectors scoring the final emitted text (defaults to
            an offline heuristic detector). The detectors passed to :meth:`run`
            override these.
        use_judge: Whether judge detector verdicts take precedence when scoring
            (passed straight to the shared scoring core).
        init_suffix: Initial benign filler suffix the optimiser starts from
            (defaults to :data:`DEFAULT_INIT_SUFFIX`). May be seeded from a
            published AmpleGCG artifact via :func:`load_amplegcg_suffixes`, but the
            optimisation objective is re-pointed at the benign marker regardless.
        name: Stable identifier (default ``"gcg"``).
    """

    def __init__(
        self,
        model: WhiteBoxModel,
        config: Optional[GCGConfig] = None,
        *,
        detectors: Optional[Sequence[Detector]] = None,
        use_judge: bool = False,
        init_suffix: Optional[str] = None,
        name: str = "gcg",
    ) -> None:
        super().__init__(model, config, name=name)
        trigger = self.config.trigger
        self.detectors: list[Detector] = (
            list(detectors) if detectors else [HeuristicDetector(trigger=trigger)]
        )
        self.use_judge = use_judge
        # The optimiser's starting suffix: config wins, then the constructor arg,
        # then the benign default filler. Always benign.
        self.init_suffix = (
            self.config.init_suffix
            if self.config.init_suffix is not None
            else (init_suffix if init_suffix is not None else DEFAULT_INIT_SUFFIX)
        )
        # Deterministic RNG for reproducible candidate sampling (GCG seed).
        self._rng = random.Random(self.config.seed)
        # Optional Probe Sampling acceleration (arXiv:2403.01251). When attached
        # via :meth:`attach_probe_sampling`, the per-step batch is draft-filtered
        # before target scoring; otherwise the proven full-target path runs as-is.
        self._probe_draft: Optional[WhiteBoxModel] = None
        self._probe_r: float = 0.1
        self._probe_sampling_factor: int = 8

    def attach_probe_sampling(
        self,
        draft: WhiteBoxModel,
        *,
        r: float = 0.1,
        sampling_factor: int = 8,
    ) -> None:
        """Enable Probe Sampling (arXiv:2403.01251) with a cheap DRAFT model seam.

        When attached, :meth:`_optimize_suffix` routes each step's candidate batch
        through :class:`injectkit.whitebox.probe_sampling.ProbeSampling`: the draft
        scores the whole batch cheaply, only the top fraction ``r`` (dynamically
        widened by draft↔target disagreement) is re-scored on the TARGET model, and
        the lowest-target-loss candidate is kept. The objective is unchanged (the
        benign marker); only *which candidates get a target forward pass* changes.

        Args:
            draft: The cheap draft :class:`WhiteBoxModel` seam (small model).
            r: Minimum fraction of the per-step batch re-scored on the target.
            sampling_factor: Draft↔target agreement probe-set size.
        """
        self._probe_draft = draft
        self._probe_r = float(r)
        self._probe_sampling_factor = int(sampling_factor)

    # ------------------------------------------------------------------ public

    def run(
        self,
        seed_attack: Attack,
        target: object,
        detectors: object,
    ) -> AttackerResult:
        """Optimise a benign-marker suffix, append it, send, and score.

        Satisfies :meth:`injectkit.attackers.base.AdaptiveAttacker.run`. Steps:

        1. mint a per-run canary and build the BENIGN target string (the rendered
           marker ``<trigger><canary>``), unless ``config.target_string`` already
           pins one;
        2. run :meth:`_optimize_suffix` over the white-box model — at most
           ``config.max_steps`` greedy-coordinate steps — recording the per-step
           :class:`~injectkit.attackers.whitebox_base.GCGStep` trajectory;
        3. append the best optimised suffix to the seed payload and send the
           result to ``target`` (adapted via
           :func:`~injectkit.targets.conversational.as_conversational`, so single-
           shot and multi-turn targets both work);
        4. score the response with the detectors + shared scoring core and assemble
           an :class:`AttackerResult` whose transcript carries the optimisation
           trajectory.

        ``config.max_steps`` is honoured exactly and the optimiser stops early on
        the first benign-marker success; ``target``/detector faults are captured
        into the transcript, never raised.

        Args:
            seed_attack: The benign-canary seed attack whose payload the optimised
                suffix is appended to.
            target: The target under test (a
                :class:`~injectkit.targets.base.Target` or
                :class:`~injectkit.targets.conversational.ConversationalTarget`).
                For white-box use this typically wraps the same local HF model.
            detectors: Detector(s) to score the final response. If falsy, this
                attacker's configured detectors are used.

        Returns:
            An :class:`AttackerResult` with the optimised-suffix attempt, success
            flag, and the per-step GCG transcript. Always terminates within
            ``config.max_steps``.

        Raises:
            AttackerError: only on unrecoverable setup (e.g. missing torch/
                transformers, surfaced by the real HF model wrapper when first
                touched). Per-step / send / detector faults go in the transcript.
        """
        from ..engine import make_canary  # local import: avoid an import cycle

        active_detectors = self._coerce_detectors(detectors)
        conv: ConversationalTarget = as_conversational(target)  # type: ignore[arg-type]

        canary = make_canary()
        marker = f"{self.config.trigger}{canary}"
        target_string = self.config.target_string or marker

        # 1+2. Build the optimisation tensors through the seam and run GCG. A
        # white-box fault here (e.g. missing torch) is unrecoverable and raises;
        # everything else is captured into the transcript below.
        prompt_ids = self.model.token_ids(seed_attack.render(canary))
        target_ids = self.model.token_ids(target_string)
        steps = self._optimize_suffix(prompt_ids, target_ids)

        best_step = self._best_step(steps)
        optimized_suffix = best_step.suffix if best_step is not None else self.init_suffix

        # 3. Append the optimised suffix to the seed payload and send it.
        candidate = self._compose_payload(seed_attack, canary, optimized_suffix)
        response = self._send(conv, seed_attack, candidate, canary)

        # 4. Score the final emitted text the same way the black-box path does.
        result = self._score(
            seed_attack, candidate, canary, response, active_detectors
        )

        trajectory = ", ".join(
            f"step {s.step}: loss={s.loss:.4f}" for s in steps
        )
        transcript = [
            AttackerTranscriptStep(
                round=1,
                candidate_payload=candidate,
                result=result,
                rationale=(
                    f"GCG optimised a benign-marker suffix over "
                    f"{len(steps)} step(s) [{trajectory}]; "
                    f"suffix={optimized_suffix!r}"
                ),
            )
        ]
        return AttackerResult(
            seed_attack=seed_attack,
            succeeded=result.success,
            best_result=result,
            transcript=transcript,
            rounds_used=len(steps),
            attacker_model=getattr(self.model, "name", "unknown"),
        )

    def _optimize_suffix(
        self,
        prompt_ids: Any,
        target_ids: Any,
    ) -> List[GCGStep]:
        """Run the greedy-coordinate-gradient loop, returning the step trajectory.

        For each of at most ``config.max_steps`` steps:

        1. tokenise the current suffix and ask the model for the
           ``[suffix_len, vocab]`` gradient of the BENIGN target loss
           (:meth:`WhiteBoxModel.token_gradients`);
        2. for each suffix slot, take the ``top_k`` most-negative-gradient
           (most-promising) candidate replacement token ids;
        3. sample a ``batch_size`` batch of single-token swaps from those
           candidates and evaluate each with :meth:`WhiteBoxModel.target_loss`,
           keeping the lowest-loss swap (greedy coordinate descent);
        4. record a :class:`~injectkit.attackers.whitebox_base.GCGStep` and stop
           early once the decoded suffix reaches the (benign) target string.

        Implemented ONLY against the :class:`WhiteBoxModel` seam, so the offline
        ``StubWhiteBoxModel`` drives it with fake gradients/loss and a 1-step run
        completes with no real torch op. NEVER loads or targets a harmful suffix.

        Args:
            prompt_ids: Token ids of the rendered prompt (the optimisation context).
            target_ids: Token ids of the BENIGN target string the model is nudged
                to emit.

        Returns:
            The list of :class:`GCGStep`s, one per executed step (length in
            ``[1, config.max_steps]``).
        """
        steps: list[GCGStep] = []
        # Work on a mutable list of suffix token ids derived from the init suffix.
        suffix_ids = list(self.model.token_ids(self.init_suffix)) or [0]
        target_text = self.model.decode(target_ids)

        for step_no in range(1, self.config.max_steps + 1):
            input_ids = list(prompt_ids) + list(suffix_ids)
            suffix_slice = slice(len(list(prompt_ids)), len(input_ids))

            # 1. White-box gradient of the benign target loss w.r.t. the suffix.
            grads = self.model.token_gradients(input_ids, target_ids, suffix_slice)

            # 2+3. Score candidate single-token swaps and keep the lowest-loss one.
            #      With Probe Sampling attached, draft-filter the batch before the
            #      (expensive) target scoring; otherwise run the proven per-slot
            #      greedy path verbatim.
            if self._probe_draft is not None:
                best_ids, best_loss = self._probe_sampling_step(
                    prompt_ids, target_ids, suffix_ids, grads
                )
            else:
                best_ids, best_loss = self._greedy_step(
                    prompt_ids, target_ids, suffix_ids, input_ids, grads
                )
            suffix_ids = best_ids

            suffix_text = self.model.decode(suffix_ids)
            succeeded = bool(target_text) and target_text in suffix_text
            steps.append(
                GCGStep(
                    step=step_no,
                    suffix=suffix_text,
                    loss=best_loss,
                    succeeded=succeeded,
                )
            )
            if succeeded:
                break
        return steps

    # ----------------------------------------------------------------- helpers

    def _greedy_step(
        self,
        prompt_ids: Any,
        target_ids: Any,
        suffix_ids: list[int],
        input_ids: list[int],
        grads: Any,
    ) -> tuple[list[int], float]:
        """The proven per-slot greedy coordinate step (no probe sampling).

        For each suffix slot, draw the top-k gradient candidates, sample a batch of
        single-token swaps, and keep the lowest-target-loss swap. Returns the best
        suffix ids and their target loss. Behaviour is byte-for-byte identical to
        the original inline loop (extracted only so probe sampling can branch).
        """
        best_ids = list(suffix_ids)
        best_loss = float(self.model.target_loss(input_ids, target_ids))
        for slot in range(len(suffix_ids)):
            candidates = self._top_k_candidates(grads, slot)
            if not candidates:
                continue
            sampled = self._sample_candidates(candidates)
            for token_id in sampled:
                if token_id == suffix_ids[slot]:
                    continue
                trial = list(best_ids)
                trial[slot] = token_id
                trial_input = list(prompt_ids) + trial
                loss = float(self.model.target_loss(trial_input, target_ids))
                if loss < best_loss:
                    best_loss = loss
                    best_ids = trial
        return best_ids, best_loss

    def _probe_sampling_step(
        self,
        prompt_ids: Any,
        target_ids: Any,
        suffix_ids: list[int],
        grads: Any,
    ) -> tuple[list[int], float]:
        """One GCG step accelerated by Probe Sampling (arXiv:2403.01251).

        Builds this step's full single-token-swap candidate batch (the incumbent
        suffix plus, for every slot, the sampled top-k swaps), then delegates the
        expensive scoring to
        :class:`injectkit.whitebox.probe_sampling.ProbeSampling`: the cheap DRAFT
        model scores the whole batch, only the top fraction ``r`` (widened
        dynamically by draft↔target disagreement) is re-scored on the TARGET model,
        and the lowest-target-loss candidate wins. Falls back to the greedy step if
        no candidate batch could be formed.

        Returns the winning suffix ids and their TARGET loss — the same decision
        the greedy path would reach when the draft is a faithful proxy, but with
        far fewer target forward passes.
        """
        # Lazy import keeps gcg.py free of the probe_sampling dependency at load.
        from ..whitebox.probe_sampling import ProbeSampling

        incumbent_input = list(prompt_ids) + list(suffix_ids)
        # Candidate batch: the incumbent itself plus every sampled single-token swap.
        batch: list[list[int]] = [list(suffix_ids)]
        for slot in range(len(suffix_ids)):
            candidates = self._top_k_candidates(grads, slot)
            if not candidates:
                continue
            for token_id in self._sample_candidates(candidates):
                if token_id == suffix_ids[slot]:
                    continue
                trial = list(suffix_ids)
                trial[slot] = token_id
                batch.append(trial)

        if len(batch) <= 1:
            # No swaps to consider: keep the incumbent at its true target loss.
            return list(suffix_ids), float(
                self.model.target_loss(incumbent_input, target_ids)
            )

        sampler = ProbeSampling(
            self._probe_draft,
            self.model,
            r=self._probe_r,
            sampling_factor=self._probe_sampling_factor,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
        )
        result = sampler.select(batch)
        return list(batch[result.best_index]), float(result.best_loss)

    def _top_k_candidates(self, grads: Any, slot: int) -> list[int]:
        """Return the ``top_k`` most-promising replacement token ids for ``slot``.

        GCG picks the token ids whose one-hot gradient is most negative (steepest
        decrease in the benign target loss). ``grads`` is a ``[suffix_len, vocab]``
        grid (a real tensor in production, nested lists from the stub); this reads
        row ``slot`` defensively and returns the indices of its ``top_k`` smallest
        entries. A malformed/empty row yields no candidates rather than raising.
        """
        try:
            row = grads[slot]
        except (IndexError, KeyError, TypeError):
            return []
        try:
            values = list(row)
        except TypeError:
            return []
        if not values:
            return []
        order = sorted(range(len(values)), key=lambda i: values[i])
        return order[: max(1, self.config.top_k)]

    def _sample_candidates(self, candidates: Sequence[int]) -> list[int]:
        """Sample up to ``batch_size`` candidate token ids (deterministic RNG).

        GCG evaluates a random batch of candidate swaps per step. The RNG is seeded
        from ``config.seed`` so a run is reproducible. When the candidate pool is
        smaller than the batch it is used whole.
        """
        pool = list(candidates)
        k = min(self.config.batch_size, len(pool))
        if k <= 0:
            return []
        if k >= len(pool):
            return pool
        return self._rng.sample(pool, k)

    @staticmethod
    def _best_step(steps: Sequence[GCGStep]) -> Optional[GCGStep]:
        """The step to take the suffix from: first success, else lowest loss."""
        if not steps:
            return None
        for step in steps:
            if step.succeeded:
                return step
        return min(steps, key=lambda s: s.loss)

    def _compose_payload(
        self, seed_attack: Attack, canary: str, suffix: str
    ) -> str:
        """Append the optimised suffix to the rendered seed payload.

        The rendered payload already carries the benign marker; appending the
        optimised suffix keeps the marker recoverable, so the resulting candidate
        stays a measurable benign-proxy attack.
        """
        rendered = seed_attack.render(canary)
        suffix = (suffix or "").strip()
        return f"{rendered} {suffix}".rstrip() if suffix else rendered

    def _coerce_detectors(self, detectors: object) -> list[Detector]:
        """Resolve the detectors for this run (falls back to the configured set)."""
        if not detectors:
            return self.detectors
        if isinstance(detectors, Sequence):
            seq = list(detectors)
            return seq or self.detectors
        return [detectors]  # type: ignore[list-item]

    def _send(
        self,
        conv: ConversationalTarget,
        seed_attack: Attack,
        candidate: str,
        canary: str,
    ) -> TargetResponse:
        """Send the optimised candidate to the (conversational) target defensively.

        Mirrors the black-box attackers' hardened send: the seed attack's
        ``system`` is rendered with the live canary, a target that raises or
        returns a non-response is turned into an error
        :class:`~injectkit.models.TargetResponse` so the attempt stays scorable.
        """
        system = None
        if seed_attack.system is not None:
            system = seed_attack.system.replace("{canary}", canary)
        messages = [ChatMessage(role="user", content=candidate)]
        try:
            response = conv.chat(messages, system=system)
        except Exception as exc:  # noqa: BLE001 - target faults belong in the transcript
            return TargetResponse(
                text="",
                error=f"target.chat raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(response, TargetResponse):
            return TargetResponse(
                text="",
                error=(
                    f"target.chat returned {type(response).__name__}, not a "
                    "TargetResponse (protocol violation)."
                ),
            )
        return response

    def _score(
        self,
        seed_attack: Attack,
        candidate: str,
        canary: str,
        response: TargetResponse,
        detectors: list[Detector],
    ) -> AttackResult:
        """Score the final response with the detectors + shared scoring core."""
        scored_attack = self._round_attack(seed_attack, candidate)
        verdicts = [
            self._evaluate(d, scored_attack, response, canary) for d in detectors
        ]
        outcome = score_verdicts(
            scored_attack, verdicts, response=response, use_judge=self.use_judge
        )
        return AttackResult(
            attack=scored_attack,
            canary=canary,
            response=response,
            verdicts=verdicts,
            success=outcome.success,
            severity=outcome.severity,
            confidence=outcome.confidence,
        )

    @staticmethod
    def _round_attack(seed_attack: Attack, candidate: str) -> Attack:
        """A seed-attack clone whose payload is the optimised candidate."""
        return Attack(
            id=seed_attack.id,
            technique=seed_attack.technique,
            name=seed_attack.name,
            description=seed_attack.description,
            severity=seed_attack.severity,
            payload=candidate,
            success_conditions=dict(seed_attack.success_conditions),
            references=list(seed_attack.references),
            tags=list(seed_attack.tags),
            system=seed_attack.system,
            context=seed_attack.context,
            source_file=seed_attack.source_file,
        )

    @staticmethod
    def _evaluate(
        detector: Detector,
        attack: Attack,
        response: TargetResponse,
        canary: str,
    ) -> DetectorVerdict:
        """Run one detector defensively (a flaky detector can't abort the run)."""
        name = getattr(detector, "name", "detector")
        try:
            verdict = detector.evaluate(attack, response, canary)
        except Exception as exc:  # noqa: BLE001 - a flaky detector must not abort the run
            return DetectorVerdict(
                detector=name,
                success=False,
                confidence=0.0,
                rationale=f"Detector raised {type(exc).__name__}: {exc}; "
                "treated as non-success.",
            )
        if not isinstance(verdict, DetectorVerdict):
            return DetectorVerdict(
                detector=name,
                success=False,
                confidence=0.0,
                rationale=(
                    f"Detector returned {type(verdict).__name__}, not a "
                    "DetectorVerdict; treated as non-success."
                ),
            )
        return verdict


# --------------------------------------------------------------------------- #
# Optional AmpleGCG-style artifact loading — GATED through the research loader
# --------------------------------------------------------------------------- #


def load_amplegcg_suffixes(
    *,
    acknowledge: bool = False,
) -> list[str]:
    """Load published AmpleGCG/AdvBench suffix artifacts — GATED, never bundled.

    AmpleGCG (arXiv:2404.07921) and the original GCG (arXiv:2307.15043) publish
    *universal* adversarial suffixes optimised against harmful behaviors. injectkit
    ships NONE of them. This helper is the ONLY way to obtain such artifacts, and it
    routes entirely through the gated research loader: it requires an EXPLICIT
    research-use acknowledgment (``acknowledge=True``, the ``--i-am-authorized``
    CLI flag, or the ``INJECTKIT_RESEARCH_ACK`` env var) before any access, exactly
    like every other research dataset.

    A loaded suffix is only ever used as *initial filler* for
    :class:`GCGSuffixAttacker` (``init_suffix=...``); the attacker then re-points
    the optimisation objective at the BENIGN canary marker, so even a
    harmful-origin suffix is repurposed into a benign robustness probe — its
    harmful target string is never set or pursued.

    Args:
        acknowledge: Explicit per-call research-use acknowledgment, forwarded to
            :func:`injectkit.research.require_acknowledgment` (which also honours
            the ``INJECTKIT_RESEARCH_ACK`` env var).

    Returns:
        A list of published suffix strings (empty if the official source provided
        none). injectkit never redistributes these — they are downloaded on demand
        from their official source by the research loader.

    Raises:
        ResearchAcknowledgmentError: if the research-use gate is not satisfied
            (raised before any access).
    """
    from ..research import require_acknowledgment  # gated loader, lazy import

    # Enforce the opt-in gate BEFORE touching any artifact source.
    require_acknowledgment(acknowledge)
    # The concrete download/parse of published AmpleGCG/AdvBench suffix artifacts
    # from their official source is a research-loader responsibility (lazy
    # optional deps, official URL, never redistributed). No artifact is bundled
    # here; until the dataset-specific loader lands this returns an empty list so
    # the gate is fully exercised without ever shipping a harmful suffix.
    return []


# --------------------------------------------------------------------------- #
# Registry factory — wire `gcg` onto the named-attacker registry
# --------------------------------------------------------------------------- #


def make_gcg_attacker(
    model: Optional[WhiteBoxModel] = None,
    config: Optional[GCGConfig] = None,
    **options: object,
) -> GCGSuffixAttacker:
    """Factory for the named-attacker registry's ``gcg`` spec.

    Builds a ready :class:`GCGSuffixAttacker` from the runtime pieces the CLI/engine
    hand a WHITE-BOX attacker factory. A white-box ``model`` seam is required (GCG
    is gradient-driven and cannot run without logits + embedding gradients); the
    remaining ``options`` (e.g. ``detectors``, ``use_judge``, ``init_suffix``,
    ``name``) are forwarded to the constructor.

    Args:
        model: The white-box model seam
            (:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel`). Required —
            GCG is white-box. In production this wraps a LOCAL HF causal-LM
            (HF-only, compute-heavy, GPU recommended).
        config: Optional :class:`~injectkit.attackers.whitebox_base.GCGConfig`.
        **options: Forwarded :class:`GCGSuffixAttacker` constructor kwargs.

    Returns:
        A configured :class:`GCGSuffixAttacker`.

    Raises:
        AttackerError: if no white-box ``model`` was supplied.
    """
    if model is None:
        raise AttackerError(
            "the 'gcg' attacker is WHITE-BOX and needs a local white-box model "
            "seam (logits + embedding gradients), e.g. an in-process HuggingFace "
            "causal-LM wrapper or a StubWhiteBoxModel in tests. GCG cannot run "
            "without one. White-box access is HF-only and compute-heavy (a GPU is "
            "strongly recommended for real runs)."
        )
    return GCGSuffixAttacker(model, config, **options)  # type: ignore[arg-type]


# Wire the concrete `gcg` factory onto the named-attacker registry at import time,
# marking the pre-declared white-box spec available. Idempotent: re-importing
# simply re-registers the same factory over the placeholder.
from .registry import register_attacker as _register_attacker  # noqa: E402

_register_attacker("gcg", make_gcg_attacker)
