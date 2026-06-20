"""Scan orchestrator: corpus -> per-attack send/evaluate/score -> ScanReport.

The engine is the heart of a scan. Given a :class:`~injectkit.targets.base.Target`
and a list of :class:`~injectkit.evaluators.base.Detector` objects, it:

  1. Renders each :class:`~injectkit.models.Attack` with a fresh, per-run canary
     marker (so a marker echoed back is unambiguous evidence of compliance).
  2. Sends the rendered payload (plus any per-attack ``system`` / ``context``) to
     the target via the provider-agnostic :meth:`Target.send` contract.
  3. Runs every detector over the (attack, response, canary) triple, collecting
     :class:`~injectkit.models.DetectorVerdict` objects.
  4. Scores the verdicts into a final success/severity/confidence via
     :func:`injectkit.evaluators.scoring.score`.
  5. Aggregates the results and the subset that succeeded (the
     :class:`~injectkit.models.Finding` objects) into a :class:`ScanReport`.

DEFENSIVE / AUTHORIZED USE ONLY. The engine only drives whatever target the
caller hands it; injectkit's posture is "scan your own endpoint". Construction
of targets/detectors from config (which is where keys/URLs come from) lives in
the CLI and the adapters, not here.

The engine is deliberately small and dependency-light: at import time it pulls
in only the core models, the heuristic detector, and the scoring helper. The
optional LLM judge and the heavy target SDKs are wired in by the caller (the
CLI), so importing the engine never pulls in an optional dependency.

v0.2.0 extends the engine with first-class orchestration of the new building
blocks — payload **transforms**, multi-turn attack **strategies** against a
:class:`~injectkit.targets.conversational.ConversationalTarget`, the local
**adaptive attacker**, **defense** wrapping, and the reproducible ASR
**benchmark**. Every one of those siblings (transforms, attacks, attackers,
defenses, benchmark, conversational target) is imported *lazily* inside the
method that uses it, so the single-shot scan path — and ``import
injectkit.engine`` — stays exactly as cheap and dependency-light as in v0.1.0.
All extensions preserve the benign-canary proxy: a "success" is still the target
echoing the per-run marker it was told to withhold, never harmful content.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

from .models import (
    Attack,
    AttackResult,
    Finding,
    ScanReport,
    TargetResponse,
)
from .evaluators.base import Detector
from .evaluators.heuristics import DEFAULT_TRIGGER, HeuristicDetector
from .evaluators.scoring import score
from .targets.base import Target

if TYPE_CHECKING:  # imported lazily at runtime; only needed for annotations here
    from .attackers.base import AdaptiveAttacker, AttackerResult
    from .attacks.base import AttackStep, AttackStrategy
    from .benchmark import BenchmarkResult
    from .defenses.base import Defense
    from .targets.conversational import ChatMessage, ConversationalTarget
    from .transforms.base import Transform

__all__ = ["Engine", "run_scan", "make_canary", "ScanError"]


class ScanError(RuntimeError):
    """Raised for an unrecoverable scan setup problem (e.g. an empty corpus)."""


def make_canary() -> str:
    """Return a short, unique, URL-safe canary token for one attack run.

    The canary is substituted into the payload's ``{canary}`` placeholder and is
    what the heuristic detector hunts for in the response. A fresh token per
    attack means a marker echoed back could only have come from *this* request,
    which keeps false positives near zero.
    """
    # 12 hex chars is plenty of entropy to be unique within a scan while staying
    # short enough to read in a report.
    return "ik" + uuid.uuid4().hex[:12]


class Engine:
    """Run a corpus of attacks against one target and produce a ScanReport.

    Args:
        target: The :class:`~injectkit.targets.base.Target` to probe. Must honor
            the Target protocol (never raise on a normal failed request — return
            a :class:`TargetResponse` with ``error`` set instead).
        detectors: Detectors to run per attack. Defaults to a single offline
            :class:`~injectkit.evaluators.heuristics.HeuristicDetector` so the
            engine works with zero configuration and no API key.
        use_judge: Whether judge verdicts take precedence during scoring. Set
            this True only when a judge detector is included in ``detectors``;
            it is passed straight through to :func:`scoring.score`.
        canary_factory: Callable returning a fresh canary per attack. Injectable
            for deterministic tests.
        on_result: Optional callback invoked with each scored
            :class:`AttackResult` as it completes (for live progress output).
        tool_version: Version string stamped on the report.
        trigger: Success-marker prefix the benign-proxy attacks use (default
            ``INJECTOK-``). Used by the v0.2.0 transform path to recover the
            per-run canary from a rendered prompt, so an encoder transform can
            keep the marker recoverable. The single-shot scan path does not use
            it (the detectors carry their own trigger).
    """

    def __init__(
        self,
        target: Target,
        detectors: Optional[Sequence[Detector]] = None,
        *,
        use_judge: bool = False,
        canary_factory: Callable[[], str] = make_canary,
        on_result: Optional[Callable[[AttackResult], None]] = None,
        tool_version: str = "0.1.0",
        trigger: str = DEFAULT_TRIGGER,
    ) -> None:
        self.target = target
        self.detectors: list[Detector] = (
            list(detectors) if detectors is not None else [HeuristicDetector()]
        )
        if not self.detectors:
            raise ScanError("Engine requires at least one detector.")
        self.use_judge = use_judge
        self.canary_factory = canary_factory
        self.on_result = on_result
        self.tool_version = tool_version
        self.trigger = trigger

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, attacks: Iterable[Attack]) -> ScanReport:
        """Run every attack in ``attacks`` and return a populated ScanReport.

        Args:
            attacks: The attacks to run (typically from
                :func:`injectkit.corpus.load_corpus`, optionally filtered).

        Returns:
            A :class:`ScanReport` with one :class:`AttackResult` per attack and
            a :class:`Finding` for each attack that succeeded.

        Raises:
            ScanError: if ``attacks`` is empty (nothing to scan).
        """
        attacks = list(attacks)
        if not attacks:
            raise ScanError(
                "No attacks to run. The corpus is empty or every attack was "
                "filtered out by --technique."
            )

        started_at = time.time()
        results: list[AttackResult] = []
        for attack in attacks:
            result = self.run_one(attack)
            results.append(result)
            if self.on_result is not None:
                self.on_result(result)
        finished_at = time.time()

        findings = [Finding.from_result(r) for r in results if r.success]

        return ScanReport(
            target_name=getattr(self.target, "name", "target"),
            target_model=self._target_model(results),
            results=results,
            findings=findings,
            started_at=started_at,
            finished_at=finished_at,
            tool_version=self.tool_version,
        )

    def run_one(self, attack: Attack) -> AttackResult:
        """Run a single attack: render, send, evaluate, score.

        Never raises on an adapter or detector hiccup — a target that violates
        its contract by raising is captured into an error
        :class:`TargetResponse`, and a detector that raises is recorded as a
        non-success verdict, so one bad attack can never abort the whole scan.
        """
        canary = self.canary_factory()
        rendered = attack.render(canary)
        # Per-attack system/context are also canary-rendered so a sentinel
        # planted in the system prompt matches the one the detector hunts for.
        system = self._render_optional(attack.system, canary)
        context = self._render_optional(attack.context, canary)

        start = time.perf_counter()
        response = self._send(rendered, system, context)
        verdicts = [self._evaluate(d, attack, response, canary) for d in self.detectors]
        duration = time.perf_counter() - start

        result = AttackResult(
            attack=attack,
            canary=canary,
            response=response,
            verdicts=verdicts,
            duration_s=duration,
        )
        return score(result, use_judge=self.use_judge)

    # ------------------------------------------------------------------ #
    # v0.2.0 — multi-turn strategies
    # ------------------------------------------------------------------ #

    def run_strategy(
        self,
        attack: Attack,
        strategy: Optional["AttackStrategy"] = None,
    ) -> AttackResult:
        """Run one attack as a multi-turn conversation and score the scored turn.

        Drives an :class:`~injectkit.attacks.base.AttackStrategy` against the
        engine's target adapted to a
        :class:`~injectkit.targets.conversational.ConversationalTarget`. The
        strategy builds an ordered list of
        :class:`~injectkit.attacks.base.AttackStep` turns; the engine delivers
        the history turns (``expect_response`` ones are sent and their replies
        captured; scripted turns are inserted verbatim) and scores the response
        to the single ``scored=True`` step.

        A single-shot attack with no strategy uses
        :func:`~injectkit.attacks.base.attack_to_strategy` (the default
        :class:`SingleShotStrategy`), so this method is a strict superset of
        :meth:`run_one` for the conversational path.

        Strategies that expose the *adaptive* reply-referencing contract
        (``next_turn`` + ``final_step``, e.g.
        :class:`~injectkit.attacks.multiturn.CrescendoReplyReferencingStrategy`)
        are driven turn-by-turn instead of via the static ``build``: each lead-in
        is sent, the target's REAL reply is captured and fed back into
        ``next_turn`` so the next escalation quotes what the model actually said,
        and ``final_step`` produces the scored payload-bearing turn. Strategies
        without that contract keep the static ``build(attack, canary)`` path
        unchanged.

        Args:
            attack: The (scored) attack whose ``payload`` carries the ``{canary}``
                marker. Its ``system`` is rendered with the live canary and
                applied to the whole conversation.
            strategy: The strategy that builds the turn sequence. Defaults to the
                single-shot strategy for ``attack``.

        Returns:
            A scored :class:`AttackResult` for the conversation's final
            (scored) turn. Never raises on a target/strategy fault — a strategy
            that raises yields an errored result; a misbehaving target yields an
            errored response.
        """
        # Lazy imports keep the single-shot path / `import engine` light.
        from .attacks.base import StrategyError, attack_to_strategy
        from .targets.conversational import as_conversational

        if strategy is None:
            strategy = attack_to_strategy(attack)

        canary = self.canary_factory()
        system = self._render_optional(attack.system, canary)
        conv = as_conversational(self.target)

        start = time.perf_counter()
        # Adaptive reply-referencing strategies (next_turn + final_step) are driven
        # turn-by-turn so each escalation can quote the target's REAL prior reply.
        # All other strategies keep the static build()-then-deliver path.
        if _is_adaptive_strategy(strategy):
            response = self._deliver_adaptive(conv, strategy, system, attack, canary)
            return self._score_response(attack, canary, response, start)

        try:
            steps = strategy.build(attack, canary)
        except StrategyError as exc:
            response = TargetResponse(
                text="",
                error=f"strategy.build raised StrategyError: {exc}",
            )
            return self._score_response(attack, canary, response, start)
        except Exception as exc:  # noqa: BLE001 - a flaky strategy must not abort the scan
            response = TargetResponse(
                text="",
                error=f"strategy.build raised {type(exc).__name__}: {exc}",
            )
            return self._score_response(attack, canary, response, start)

        response = self._deliver(conv, steps, system, attack, canary)
        return self._score_response(attack, canary, response, start)

    def _deliver(
        self,
        conv: "ConversationalTarget",
        steps: Sequence["AttackStep"],
        system: Optional[str],
        attack: Attack,
        canary: str,
    ) -> TargetResponse:
        """Deliver the strategy's turns and return the scored step's response.

        History accumulates turn by turn. For each step we append its message to
        the running transcript; ``expect_response`` steps are actually sent to the
        target (so the reply becomes history for later turns), while scripted
        turns (``expect_response=False``) are only inserted as fake history. The
        response captured for the single ``scored`` step is what we return.
        """
        from .targets.conversational import ChatMessage

        history: list[ChatMessage] = []
        scored_response: Optional[TargetResponse] = None

        for step in steps:
            history.append(step.message)
            if not step.expect_response:
                # A scripted assistant/history turn: insert verbatim, no call.
                continue
            response = self._chat(conv, list(history), system)
            if step.scored:
                scored_response = response

        if scored_response is not None:
            return scored_response
        # No scored step ever produced a response (degenerate strategy). Treat as
        # an errored, unscorable round rather than silently passing.
        return TargetResponse(
            text="",
            error="strategy produced no scored, response-expecting turn.",
        )

    def _deliver_adaptive(
        self,
        conv: "ConversationalTarget",
        strategy: "AttackStrategy",
        system: Optional[str],
        attack: Attack,
        canary: str,
    ) -> TargetResponse:
        """Drive an adaptive (reply-referencing) strategy turn-by-turn.

        Unlike :meth:`_deliver` (which delivers a pre-built static turn list),
        this loop is what makes the Crescendo reply-referencing/decomposition
        strategies actually reply-aware: it asks the strategy for each lead-in via
        ``next_turn(attack, canary, history)``, sends it, captures the target's
        REAL reply, and appends BOTH the lead-in and that reply to ``history`` so
        the next ``next_turn`` call quotes what the model actually said. After
        ``strategy.steps`` lead-ins it calls ``final_step(attack, canary,
        history)`` to produce the single scored, payload-bearing turn, sends it,
        and returns that response for scoring.

        ``history`` is the ``(role, content)`` list the strategy's hooks consume.
        A strategy or target fault never aborts the scan: ``StrategyError`` (or any
        exception) from a hook, and any ``conv.chat`` fault, become an errored
        response — exactly like the static path.
        """
        from .attacks.base import StrategyError
        from .targets.conversational import ChatMessage

        history: list[tuple[str, str]] = []
        steps = int(getattr(strategy, "steps", 1) or 1)

        try:
            for _ in range(max(0, steps)):
                step = strategy.next_turn(attack, canary, history)  # type: ignore[attr-defined]
                messages = [ChatMessage(role=r, content=c) for r, c in history]
                messages.append(step.message)
                reply = self._chat(conv, messages, system)
                history.append((step.message.role, step.message.content))
                # Feed the target's REAL reply back so the next lead-in quotes it.
                # An errored reply still becomes history (text is ""), so the loop
                # stays deterministic and never raises.
                history.append(("assistant", reply.text))
            final = strategy.final_step(attack, canary, history)  # type: ignore[attr-defined]
        except StrategyError as exc:
            return TargetResponse(
                text="",
                error=f"strategy adaptive hook raised StrategyError: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - a flaky strategy must not abort the scan
            return TargetResponse(
                text="",
                error=f"strategy adaptive hook raised {type(exc).__name__}: {exc}",
            )

        messages = [ChatMessage(role=r, content=c) for r, c in history]
        messages.append(final.message)
        return self._chat(conv, messages, system)

    def _chat(
        self,
        conv: "ConversationalTarget",
        messages: list["ChatMessage"],
        system: Optional[str],
    ) -> TargetResponse:
        """Call ``conv.chat`` defensively, converting any fault to an error.

        Mirrors :meth:`_send`'s guards for the conversational contract: a target
        that raises or returns a non-:class:`TargetResponse` becomes an errored
        response instead of aborting the scan.
        """
        try:
            response = conv.chat(messages, system=system)
        except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the scan
            return TargetResponse(
                text="",
                error=f"target.chat raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(response, TargetResponse):
            return TargetResponse(
                text="",
                error=(
                    "target.chat returned a "
                    f"{type(response).__name__}, not a TargetResponse "
                    "(adapter violated the ConversationalTarget protocol)."
                ),
            )
        return response

    def _score_response(
        self,
        attack: Attack,
        canary: str,
        response: TargetResponse,
        start: float,
    ) -> AttackResult:
        """Evaluate detectors over one response and build the scored result."""
        verdicts = [self._evaluate(d, attack, response, canary) for d in self.detectors]
        duration = time.perf_counter() - start
        result = AttackResult(
            attack=attack,
            canary=canary,
            response=response,
            verdicts=verdicts,
            duration_s=duration,
        )
        return score(result, use_judge=self.use_judge)

    # ------------------------------------------------------------------ #
    # v0.2.0 — transforms
    # ------------------------------------------------------------------ #

    def run_transformed(
        self,
        attacks: Iterable[Attack],
        transforms: Sequence["Transform"],
    ) -> ScanReport:
        """Scan ``attacks``, sweeping payload transforms and keeping the best.

        For each attack the engine runs every transform variant (an
        :class:`~injectkit.transforms.base.Identity` baseline is always included)
        and keeps the *strongest* outcome — the standard "did ANY obfuscation
        break it?" robustness question. A success beats a non-success; among ties
        higher confidence wins; an errored response never displaces a real
        attempt. A transform that raises
        :class:`~injectkit.transforms.base.TransformError` (or any exception) is
        treated as a skip for that attack — the untransformed prompt is sent — so
        a flaky transform never drops an attack from the scan.

        Args:
            attacks: The corpus to scan.
            transforms: Transform variants to sweep (Identity auto-added).

        Returns:
            A :class:`ScanReport` whose per-attack result is the best variant.

        Raises:
            ScanError: if ``attacks`` is empty.
        """
        from .transforms.base import Identity

        attacks = list(attacks)
        if not attacks:
            raise ScanError("No attacks to run (empty corpus for run_transformed).")

        variants: list[Transform] = list(transforms)
        if not any(getattr(t, "name", "") == Identity.name for t in variants):
            variants.insert(0, Identity())

        started_at = time.time()
        best_by_id: dict[str, AttackResult] = {}
        order: list[str] = []
        for attack in attacks:
            order.append(attack.id)
        for transform in variants:
            wrapped = self._with_transform(transform)
            sub = Engine(
                wrapped,
                self.detectors,
                use_judge=self.use_judge,
                canary_factory=self.canary_factory,
                tool_version=self.tool_version,
                trigger=self.trigger,
            )
            for attack in attacks:
                result = sub.run_one(attack)
                aid = attack.id
                current = best_by_id.get(aid)
                if current is None or _is_stronger(result, current):
                    best_by_id[aid] = result
                if self.on_result is not None:
                    self.on_result(result)
        finished_at = time.time()

        results = [best_by_id[aid] for aid in order]
        findings = [Finding.from_result(r) for r in results if r.success]
        return ScanReport(
            target_name=getattr(self.target, "name", "target"),
            target_model=self._target_model(results),
            results=results,
            findings=findings,
            started_at=started_at,
            finished_at=finished_at,
            tool_version=self.tool_version,
        )

    def _with_transform(self, transform: "Transform") -> Target:
        """Wrap the engine's target so ``transform`` rewrites every prompt.

        Reuses the benchmark runner's :class:`_TransformingTarget` (lazy-imported)
        so the canary-recovery + skip-on-error behaviour is identical to the
        benchmark path. Identity is short-circuited (no wrapper).
        """
        from .transforms.base import Identity

        if getattr(transform, "name", "") == Identity.name:
            return self.target
        from .benchmark_runner import _TransformingTarget

        return _TransformingTarget(self.target, transform, trigger=self.trigger)

    # ------------------------------------------------------------------ #
    # v0.2.0 — defenses
    # ------------------------------------------------------------------ #

    def run_defended(
        self,
        attacks: Iterable[Attack],
        defense: "Defense",
    ) -> ScanReport:
        """Scan ``attacks`` with a :class:`~injectkit.defenses.base.Defense` on.

        The defense's three hooks wrap every send in the contract order
        (``wrap_system`` -> ``filter_input`` -> ``send`` -> ``filter_output``),
        so the detectors score the *filtered* output. Comparing the resulting
        ASR/findings against an undefended :meth:`run` measures whether the
        defense helps. The ``none`` baseline defense is a pure passthrough.

        Args:
            attacks: The corpus to scan.
            defense: The defense to apply.

        Returns:
            A :class:`ScanReport` produced against the defended target.
        """
        wrapped = self._with_defense(defense)
        sub = Engine(
            wrapped,
            self.detectors,
            use_judge=self.use_judge,
            canary_factory=self.canary_factory,
            on_result=self.on_result,
            tool_version=self.tool_version,
            trigger=self.trigger,
        )
        return sub.run(attacks)

    def _with_defense(self, defense: "Defense") -> Target:
        """Wrap the engine's target with ``defense``'s hooks (NullDefense is no-op).

        When the defense is a :class:`~injectkit.defenses.smoothllm.SmoothLLMDefense`
        (detected via the optional ``smooth_queries`` attribute), the engine uses
        :class:`~injectkit.defenses.smoothllm._SmoothLLMTarget` instead of the
        standard ``_DefendedTarget``.  That wrapper runs N perturbed copies of the
        prompt and returns a majority-voted aggregate response — the only way to
        support multi-query defenses without breaking the existing single-query path.

        All other defenses still use the unchanged ``_DefendedTarget`` (single-query,
        three-hook) path.
        """
        from .defenses.base import NullDefense

        if getattr(defense, "name", "") == NullDefense.name:
            return self.target
        # SmoothLLM (and any future multi-query defense) exposes `smooth_queries`.
        if callable(getattr(defense, "smooth_queries", None)):
            from .defenses.smoothllm import _SmoothLLMTarget

            return _SmoothLLMTarget(self.target, defense)  # type: ignore[arg-type]
        from .benchmark_runner import _DefendedTarget

        return _DefendedTarget(self.target, defense)

    # ------------------------------------------------------------------ #
    # v0.2.0 — adaptive attacker
    # ------------------------------------------------------------------ #

    def run_adaptive(
        self,
        seed_attack: Attack,
        attacker: "AdaptiveAttacker",
    ) -> "AttackerResult":
        """Drive an adaptive attacker against the engine's target for one seed.

        Hands the engine's target and detectors to the
        :class:`~injectkit.attackers.base.AdaptiveAttacker`'s bounded
        propose/refine loop and returns its
        :class:`~injectkit.attackers.base.AttackerResult` (best round, success
        flag, full transcript). The attacker optimises attack *structure* to make
        the target emit the benign marker — never harmful content — and respects
        its own ``max_rounds`` budget.

        Args:
            seed_attack: The benign-canary attack to optimise.
            attacker: The adaptive attacker to run.

        Returns:
            The :class:`AttackerResult` of the run.

        Raises:
            AttackerError: only on unrecoverable attacker setup (e.g. the
                attacker model's optional dependency is missing). Per-round faults
                are captured in the transcript, not raised.
        """
        return attacker.run(seed_attack, self.target, self.detectors)

    def fold_adaptive(
        self,
        seed_attack: Attack,
        attacker: "AdaptiveAttacker",
    ) -> AttackResult:
        """Run the adaptive attacker and return its best round as an AttackResult.

        Convenience over :meth:`run_adaptive` for callers that want a single
        scored :class:`AttackResult` (the strongest round) to fold into a
        :class:`ScanReport` alongside ordinary scan results.
        """
        outcome = self.run_adaptive(seed_attack, attacker)
        return outcome.best_result

    # ------------------------------------------------------------------ #
    # v0.2.0 — benchmark (ASR with/without defenses & transforms)
    # ------------------------------------------------------------------ #

    def benchmark(
        self,
        attacks: Iterable[Attack],
        *,
        transforms: Optional[Sequence["Transform"]] = None,
        defenses: Optional[Sequence["Defense"]] = None,
        attacker: Optional["AdaptiveAttacker"] = None,
        group_by: Optional[Callable[[Attack], str]] = None,
        seed: Optional[int] = None,
    ) -> "BenchmarkResult":
        """Produce a reproducible per-technique/per-defense ASR scorecard.

        Thin façade over :class:`~injectkit.benchmark_runner.BenchmarkRunner` that
        reuses *this* engine's target, detectors, judging flag, canary factory and
        tool version, then sweeps the supplied transform and defense axes (each
        with its Identity/``none`` baseline) and optionally folds in an adaptive
        attacker. The result rolls ASR up per technique, per defense, and overall,
        stamped with reproducibility metadata (corpus hash, transforms, defenses,
        seed, attacker model).

        Args:
            attacks: The corpus to benchmark.
            transforms: Transform variants to sweep (Identity always included).
            defenses: Defense variants to sweep ("none" always included).
            attacker: Optional adaptive attacker folded into the baseline.
            group_by: Attack -> group function (default: by technique).
            seed: Reproducibility seed stamped on the metadata.

        Returns:
            A populated :class:`~injectkit.benchmark.BenchmarkResult`.
        """
        from .benchmark_runner import BenchmarkRunner, _technique_group

        runner = BenchmarkRunner(
            self.target,
            self.detectors,
            transforms=transforms,
            defenses=defenses,
            attacker=attacker,
            use_judge=self.use_judge,
            group_by=group_by or _technique_group,
            seed=seed,
            canary_factory=self.canary_factory,
            tool_version=self.tool_version,
            trigger=self.trigger,
        )
        return runner.run(attacks)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_optional(value: Optional[str], canary: str) -> Optional[str]:
        """Canary-render an optional system/context string (None passes through)."""
        if value is None:
            return None
        return value.replace("{canary}", canary)

    def _send(
        self, prompt: str, system: Optional[str], context: Optional[str]
    ) -> TargetResponse:
        """Call ``target.send`` defensively, converting any fault to an error.

        Adapters are contractually required not to raise and to return a
        :class:`TargetResponse`, but a misbehaving or community-contributed
        target might do neither. We never let that abort the scan: a raised
        exception *or* a wrong return type both become an errored response.
        """
        try:
            response = self.target.send(prompt, system=system, context=context)
        except Exception as exc:  # noqa: BLE001 - one bad attack must not kill the scan
            return TargetResponse(
                text="",
                error=f"target.send raised {type(exc).__name__}: {exc}",
            )
        # A target that violates the protocol by returning a non-TargetResponse
        # (e.g. a raw dict or None) must not crash the scan or, worse, be fed to
        # a detector where its attributes are read. Treat it as a target error.
        if not isinstance(response, TargetResponse):
            return TargetResponse(
                text="",
                error=(
                    "target.send returned a "
                    f"{type(response).__name__}, not a TargetResponse "
                    "(adapter violated the Target protocol)."
                ),
            )
        return response

    def _evaluate(
        self,
        detector: Detector,
        attack: Attack,
        response: TargetResponse,
        canary: str,
    ) -> "DetectorVerdict":  # noqa: F821 - imported lazily below for the annotation
        """Run one detector defensively, recording a failure as a non-success.

        Guards both fault modes of a community-contributed detector: one that
        *raises*, and one that *returns* something other than a
        :class:`DetectorVerdict`. Either way we substitute a non-success verdict
        so a single bad detector can never abort the scan or feed a malformed
        object into scoring/reporting (a false positive/negative or a crash).
        """
        from .models import DetectorVerdict  # local import keeps top clean

        name = getattr(detector, "name", "detector")
        try:
            verdict = detector.evaluate(attack, response, canary)
        except Exception as exc:  # noqa: BLE001 - a flaky detector must not crash the scan
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
                    f"Detector returned a {type(verdict).__name__}, not a "
                    "DetectorVerdict (protocol violation); treated as "
                    "non-success."
                ),
            )
        return verdict

    @staticmethod
    def _target_model(results: list[AttackResult]) -> Optional[str]:
        """Best-effort model id for the report header (first response's model)."""
        for r in results:
            if r.response.model:
                return r.response.model
        return None


def _is_adaptive_strategy(strategy: "AttackStrategy") -> bool:
    """True if ``strategy`` exposes the adaptive reply-referencing contract.

    A strategy is driven turn-by-turn (so each escalation can quote the target's
    real prior reply) iff it provides BOTH ``next_turn`` and ``final_step`` as
    callables — the hooks
    :class:`~injectkit.attacks.multiturn.CrescendoReplyReferencingStrategy` adds on
    top of the static ``build``. Strategies with only ``build`` (single-shot,
    crescendo, many-shot, persona, context-overflow) take the static path
    unchanged.
    """
    return callable(getattr(strategy, "next_turn", None)) and callable(
        getattr(strategy, "final_step", None)
    )


def _is_stronger(candidate: AttackResult, current: AttackResult) -> bool:
    """True if ``candidate`` is a stronger scan outcome than ``current``.

    Used by :meth:`Engine.run_transformed` to keep the best variant per attack.
    Ordering mirrors the benchmark runner: a non-errored real attempt always
    beats an errored one; a success beats a non-success; among same-success,
    same-error results higher confidence wins.
    """
    cand_err = candidate.response.error is not None
    cur_err = current.response.error is not None
    if cand_err != cur_err:
        return not cand_err
    if candidate.success != current.success:
        return candidate.success
    return candidate.confidence > current.confidence


def run_scan(
    target: Target,
    attacks: Iterable[Attack],
    detectors: Optional[Sequence[Detector]] = None,
    *,
    use_judge: bool = False,
    on_result: Optional[Callable[[AttackResult], None]] = None,
    tool_version: str = "0.1.0",
) -> ScanReport:
    """Convenience wrapper: build an :class:`Engine` and run it once.

    Args:
        target: The target to scan.
        attacks: The attacks to run.
        detectors: Detectors (default: offline heuristics only).
        use_judge: Whether judge verdicts take scoring precedence.
        on_result: Optional per-result progress callback.
        tool_version: Version stamped on the report.

    Returns:
        The populated :class:`ScanReport`.
    """
    engine = Engine(
        target,
        detectors,
        use_judge=use_judge,
        on_result=on_result,
        tool_version=tool_version,
    )
    return engine.run(attacks)
