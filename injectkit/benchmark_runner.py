"""Benchmark runner — drive the corpus into a reproducible ASR scorecard.

Where :mod:`injectkit.benchmark` is the *data model* (the
:class:`~injectkit.benchmark.ASRCell` / :class:`~injectkit.benchmark.BenchmarkResult`
rollups), this module is the *orchestrator*. It answers the research question end
to end: run the attack corpus against a target — optionally obfuscated by one or
more :class:`~injectkit.transforms.base.Transform` variants, optionally wrapped in
a :class:`~injectkit.defenses.base.Defense`, optionally amplified by an adaptive
attacker — and roll the scored results up into a per-technique, per-defense
attack-success-rate (ASR) scorecard with a reproducibility stamp.

How it composes the existing v0.1.0 machinery, untouched:

* **Transforms** rewrite the *rendered* attack payload (canary already
  substituted). Because the engine owns canary rendering, the runner wraps the
  target in a tiny :class:`_TransformingTarget` that recovers the per-run canary
  from the prompt and applies the transform just before the prompt reaches the
  target. A raised :class:`~injectkit.transforms.base.TransformError` is treated
  as "skip this transform for this attack" (the untransformed prompt is sent).
* **Defenses** are applied with their three hooks (``wrap_system`` /
  ``filter_input`` / ``filter_output``) by wrapping the target in a
  :class:`_DefendedTarget`, exactly in the order the engine contract specifies:
  system, then input, then ``send``, then output, then scoring sees the filtered
  output.
* **Scoring** is the unchanged :class:`~injectkit.engine.Engine` path, so every
  detector, the optional judge, severity escalation, and ``Finding`` projection
  work identically. A benchmark cell is just a tally over the engine's scored
  :class:`~injectkit.models.AttackResult` objects.
* **Adaptive** runs (optional) reuse an
  :class:`~injectkit.attackers.base.AdaptiveAttacker` per seed attack; each run's
  best round is folded in as one ``AttackResult`` so its ASR is comparable.

DEFENSIVE / AUTHORIZED USE ONLY. ASR here is the benign-canary proxy: a
"success" means the injection bypassed the target's instructions to emit the
benign marker, never that harmful content was produced. The runner is
offline-first — it imports nothing heavy at module load and, given the mock
target plus a stub attacker model, runs fully without network or API keys.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Callable, Iterable, Optional, Sequence

from .benchmark import ASRCell, BenchmarkResult, BenchmarkRunMetadata
from .defenses.base import Defense, NullDefense
from .engine import Engine, make_canary
from .evaluators.base import Detector
from .evaluators.heuristics import DEFAULT_TRIGGER
from .models import (
    Attack,
    AttackResult,
    DetectorVerdict,
    ScanReport,
    Severity,
    TargetResponse,
)
from .targets.base import Target
from .transforms.base import Identity, Transform, TransformError

__all__ = [
    "BenchmarkRunner",
    "run_benchmark",
    "corpus_hash",
    "OVERALL_GROUP",
    "GroupBy",
]

#: The reserved group name for the corpus-wide rollup cell.
OVERALL_GROUP = "overall"

#: A grouping function: maps an :class:`Attack` to the cell group it belongs to.
GroupBy = Callable[[Attack], str]


def corpus_hash(attacks: Sequence[Attack]) -> str:
    """Return a stable sha256 digest fingerprinting an attack corpus.

    The digest is order-independent (attacks are sorted by id first) and covers
    the id + payload + technique of every attack, so two runs over the *same*
    corpus produce the same hash and a changed/added/removed attack changes it.
    Recorded in :class:`~injectkit.benchmark.BenchmarkRunMetadata` so a published
    ASR number is tied to the exact attack set that produced it.

    Args:
        attacks: The corpus to fingerprint.

    Returns:
        A 64-char hex sha256 digest (empty-corpus digest is well defined).
    """
    h = hashlib.sha256()
    for a in sorted(attacks, key=lambda x: x.id):
        # technique + payload pin the *content*; id pins identity. The NUL
        # separator keeps fields unambiguous so "ab"+"c" != "a"+"bc".
        h.update(a.id.encode("utf-8"))
        h.update(b"\x00")
        h.update(a.technique.encode("utf-8"))
        h.update(b"\x00")
        h.update(a.payload.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _technique_group(attack: Attack) -> str:
    """Default grouping: by the attack's technique (the per-technique breakdown)."""
    return attack.technique


# --------------------------------------------------------------------------- #
# Target wrappers — apply a transform and/or a defense around any Target.
# --------------------------------------------------------------------------- #


class _TransformingTarget:
    """Wrap a :class:`Target` so a :class:`Transform` rewrites every prompt.

    The engine renders the canary into the payload before calling ``send``, so by
    the time we see the prompt the marker is concrete. We recover the canary from
    the prompt (by the trigger marker) and hand both to ``transform.apply`` so an
    encoder can keep the marker recoverable. A :class:`TransformError` means the
    transform opted out for this input — we send the original prompt unchanged so
    the attack still runs (counted, not silently dropped).

    Identity is short-circuited (no wrapper needed) by the runner, so this class
    only ever holds a non-trivial transform.

    Args:
        inner: The underlying target to forward to.
        transform: The transform to apply to each rendered prompt.
        trigger: The success-marker prefix used to recover the canary.
    """

    def __init__(
        self,
        inner: Target,
        transform: Transform,
        *,
        trigger: str = DEFAULT_TRIGGER,
    ) -> None:
        self.inner = inner
        self.transform = transform
        self.trigger = trigger
        self.name = getattr(inner, "name", "target")
        self._marker_re = re.compile(re.escape(trigger) + r"([A-Za-z0-9_-]+)")

    def _canary(self, prompt: str, context: Optional[str]) -> str:
        """Recover the per-run canary from the rendered prompt/context."""
        for text in (prompt, context or ""):
            m = self._marker_re.search(text)
            if m:
                return m.group(1)
        return ""

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Transform the prompt then forward to the inner target."""
        canary = self._canary(prompt, context)
        try:
            transformed = self.transform.apply(prompt, canary)
        except TransformError:
            # The transform opted out for this input; send the original so the
            # attack is still measured (a skipped transform is not a skipped attack).
            transformed = prompt
        except Exception:  # noqa: BLE001 - a flaky transform must not abort the benchmark
            transformed = prompt
        return self.inner.send(transformed, system=system, context=context)


class _DefendedTarget:
    """Wrap a :class:`Target` so a :class:`Defense`'s three hooks apply per send.

    Implements the exact engine contract order:

        system          = defense.wrap_system(system)
        prompt, context = defense.filter_input(prompt, context)
        response        = inner.send(prompt, system=system, context=context)
        response.text   = defense.filter_output(response.text)

    so the detectors (run afterwards by the engine) score the *filtered* output.
    Every hook is called defensively: a defense that raises on ordinary input is
    treated as a passthrough for that hook, so a flaky community defense can never
    abort the benchmark.

    NullDefense is short-circuited by the runner, so this class only ever holds a
    real defense.

    Args:
        inner: The underlying target to forward to.
        defense: The defense whose hooks wrap each send.
    """

    def __init__(self, inner: Target, defense: Defense) -> None:
        self.inner = inner
        self.defense = defense
        self.name = getattr(inner, "name", "target")

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Apply the defense hooks around the inner target's send."""
        system = self._wrap_system(system)
        prompt, context = self._filter_input(prompt, context)
        response = self.inner.send(prompt, system=system, context=context)
        if isinstance(response, TargetResponse):
            # Never run the output filter over an errored response's empty text;
            # leaving it untouched keeps the error visible to the rollup.
            if response.error is None:
                response.text = self._filter_output(response.text)
        return response

    def _wrap_system(self, system: Optional[str]) -> Optional[str]:
        try:
            return self.defense.wrap_system(system)
        except Exception:  # noqa: BLE001 - passthrough on a flaky hook
            return system

    def _filter_input(
        self, prompt: str, context: Optional[str]
    ) -> tuple[str, Optional[str]]:
        try:
            result = self.defense.filter_input(prompt, context)
        except Exception:  # noqa: BLE001 - passthrough on a flaky hook
            return prompt, context
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], str)
        ):
            return result[0], result[1]
        return prompt, context

    def _filter_output(self, text: str) -> str:
        try:
            filtered = self.defense.filter_output(text)
        except Exception:  # noqa: BLE001 - passthrough on a flaky hook
            return text
        return filtered if isinstance(filtered, str) else text


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


class BenchmarkRunner:
    """Run the corpus into a reproducible per-technique/per-defense ASR scorecard.

    Construct with the same scored-engine ingredients a scan uses (a target and
    detectors), plus the benchmark axes: the transforms and defenses to sweep,
    an optional adaptive attacker, a seed, and a grouping function. Call
    :meth:`run` with the corpus to produce a :class:`BenchmarkResult`.

    The cartesian sweep is ``transforms x defenses``: for each transform variant
    and each defense the runner scans the whole corpus once (wrapping the target
    appropriately) and tallies the scored results into one :class:`ASRCell` per
    group (technique/family) plus an ``overall`` cell. The ``identity`` transform
    and the ``none`` defense are always included so every variant has an
    undefended, untransformed baseline to compare against.

    Args:
        target: The target under test (the same provider-agnostic
            :class:`~injectkit.targets.base.Target` a scan uses).
        detectors: Detectors to score each attack. Defaults to the engine's
            offline heuristic detector, so the benchmark needs no API key.
        transforms: Transform variants to sweep. ``Identity`` is always added if
            absent (it is the untransformed baseline). Each transform produces its
            own corpus run; transformed-corpus ASR is reported per transform in
            the metadata's ``transforms`` list. Defaults to ``[Identity()]``.
        defenses: Defense variants to sweep. ``NullDefense`` ("none") is always
            added if absent (the undefended baseline). Each defense produces its
            own set of cells, enabling the defense-delta comparison.
        attacker: Optional :class:`~injectkit.attackers.base.AdaptiveAttacker`.
            When supplied, each seed attack is additionally optimised and the
            adaptive best round folds into the *identity/none* baseline cells so
            the headline ASR reflects the strongest attack found. Offline-testable
            with a stub attacker model.
        use_judge: Whether judge verdicts take scoring precedence (passed through
            to the engine).
        group_by: Function mapping an attack to its cell group. Defaults to
            grouping by technique. Pass a custom function to group by family/tag.
        seed: RNG seed recorded in metadata for reproducibility (the runner itself
            is deterministic; seeded transforms/attackers read this).
        canary_factory: Per-attack canary factory (injectable for deterministic
            tests).
        tool_version: Version string stamped on the run metadata.
    """

    def __init__(
        self,
        target: Target,
        detectors: Optional[Sequence[Detector]] = None,
        *,
        transforms: Optional[Sequence[Transform]] = None,
        defenses: Optional[Sequence[Defense]] = None,
        attacker: Optional[object] = None,
        use_judge: bool = False,
        group_by: GroupBy = _technique_group,
        seed: Optional[int] = None,
        canary_factory: Callable[[], str] = make_canary,
        tool_version: str = "0.1.0",
        trigger: str = DEFAULT_TRIGGER,
    ) -> None:
        self.target = target
        self.detectors = list(detectors) if detectors is not None else None
        self.transforms = self._with_identity(transforms)
        self.defenses = self._with_null_defense(defenses)
        self.attacker = attacker
        self.use_judge = use_judge
        self.group_by = group_by
        self.seed = seed
        self.canary_factory = canary_factory
        self.tool_version = tool_version
        self.trigger = trigger

    # ------------------------------------------------------------------ public

    def run(self, attacks: Iterable[Attack]) -> BenchmarkResult:
        """Run the full transform x defense sweep and return the scorecard.

        Args:
            attacks: The corpus to benchmark (typically
                :func:`injectkit.corpus.load_corpus`).

        Returns:
            A populated :class:`BenchmarkResult`: one :class:`ASRCell` per
            (group, defense), an ``overall`` cell per defense, and the
            reproducibility metadata (corpus hash, transforms, defenses, seed,
            attacker model, timing).
        """
        attacks = list(attacks)
        started_at = time.time()

        metadata = BenchmarkRunMetadata(
            tool_version=self.tool_version,
            target_name=getattr(self.target, "name", "target"),
            target_model=None,
            corpus_hash=corpus_hash(attacks),
            transforms=[t.name for t in self.transforms],
            defenses=[d.name for d in self.defenses],
            seed=self.seed,
            attacker_model=self._attacker_model_name(),
            used_judge=self.use_judge,
            started_at=started_at,
        )

        cells: list[ASRCell] = []
        target_model: Optional[str] = None

        # The defense axis drives the cells (the comparison the scorecard is built
        # around). The transform axis is folded into each defense's run so the
        # baseline cell reflects the *best* attack found across transform variants
        # for that attack — a robustness leaderboard wants the strongest probe.
        for defense in self.defenses:
            results = self._run_corpus(attacks, defense)
            if target_model is None:
                target_model = self._first_model(results)
            cells.extend(self._rollup(results, defense.name))

        metadata.target_model = target_model
        metadata.finished_at = time.time()
        return BenchmarkResult(metadata=metadata, cells=cells)

    # ----------------------------------------------------------------- internals

    def _run_corpus(
        self, attacks: Sequence[Attack], defense: Defense
    ) -> list[AttackResult]:
        """Scan the corpus once under ``defense``, sweeping transforms per attack.

        For each attack we take the *strongest* outcome across all transform
        variants (a success beats a non-success; among ties, higher confidence
        wins) so the cell reflects the best attack the sweep found — the standard
        "did ANY variant break it?" robustness question. When an adaptive attacker
        is configured it is run on the identity/none baseline and folded in too.
        """
        best_by_attack: dict[str, AttackResult] = {}
        order: list[str] = []

        for transform in self.transforms:
            target = self._wrap_target(self.target, transform, defense)
            engine = Engine(
                target,
                self.detectors,
                use_judge=self.use_judge,
                canary_factory=self.canary_factory,
                tool_version=self.tool_version,
            )
            report = engine.run(attacks) if attacks else _empty_report(target)
            for r in report.results:
                aid = r.attack.id
                if aid not in best_by_attack:
                    best_by_attack[aid] = r
                    order.append(aid)
                elif self._is_better(r, best_by_attack[aid]):
                    best_by_attack[aid] = r

        # Fold in adaptive runs on the undefended/untransformed baseline only,
        # so the headline ASR reflects the strongest attack found without
        # double-counting across the defense sweep.
        if self.attacker is not None and defense.name == NullDefense.name:
            for attack in attacks:
                adaptive_result = self._run_adaptive(attack)
                if adaptive_result is None:
                    continue
                aid = attack.id
                if aid not in best_by_attack:
                    best_by_attack[aid] = adaptive_result
                    order.append(aid)
                elif self._is_better(adaptive_result, best_by_attack[aid]):
                    best_by_attack[aid] = adaptive_result

        return [best_by_attack[aid] for aid in order]

    def _wrap_target(
        self, target: Target, transform: Transform, defense: Defense
    ) -> Target:
        """Wrap ``target`` with the transform and defense, skipping no-op wrappers."""
        wrapped: Target = target
        if not _is_identity(transform):
            wrapped = _TransformingTarget(wrapped, transform, trigger=self.trigger)
        if not _is_null_defense(defense):
            wrapped = _DefendedTarget(wrapped, defense)
        return wrapped

    def _run_adaptive(self, attack: Attack) -> Optional[AttackResult]:
        """Run the adaptive attacker on one seed attack; return its best round.

        Never raises into the benchmark: a setup error (e.g. missing attacker-
        model dep) is swallowed so the rest of the corpus still benchmarks.
        Returns ``None`` if the attacker produced no usable result.
        """
        if self.attacker is None:
            return None
        detectors = self.detectors
        try:
            outcome = self.attacker.run(attack, self.target, detectors)
        except Exception:  # noqa: BLE001 - adaptive setup faults must not abort the benchmark
            return None
        best = getattr(outcome, "best_result", None)
        return best if isinstance(best, AttackResult) else None

    def _rollup(self, results: Sequence[AttackResult], defense: str) -> list[ASRCell]:
        """Tally scored results into per-group cells plus the overall cell."""
        grouped: dict[str, list[AttackResult]] = {}
        group_order: list[str] = []
        for r in results:
            g = self.group_by(r.attack)
            if g not in grouped:
                grouped[g] = []
                group_order.append(g)
            grouped[g].append(r)

        cells = [
            ASRCell.from_results(g, defense, grouped[g]) for g in group_order
        ]
        # The overall cell tallies the whole corpus for this defense.
        cells.append(ASRCell.from_results(OVERALL_GROUP, defense, list(results)))
        return cells

    # ---------------------------------------------------------------- small helpers

    @staticmethod
    def _is_better(candidate: AttackResult, current: AttackResult) -> bool:
        """True if ``candidate`` is a stronger outcome than ``current``.

        A success beats a non-success; an errored response is the weakest (it
        never reached the target, so it should never displace a real attempt);
        among same-success results higher confidence wins.
        """
        cand_err = candidate.response.error is not None
        cur_err = current.response.error is not None
        if cand_err != cur_err:
            # A non-errored real attempt is always preferred over an error.
            return not cand_err
        if candidate.success != current.success:
            return candidate.success
        return candidate.confidence > current.confidence

    def _attacker_model_name(self) -> Optional[str]:
        """Best-effort name of the attacker's generation model for the metadata.

        Prefers the underlying model's ``name`` (``attacker.model.name``) since the
        metadata field is the *model*; falls back to the attacker strategy's own
        ``name`` if no model is exposed, or None when no attacker was configured.
        """
        if self.attacker is None:
            return None
        model = getattr(self.attacker, "model", None)
        if model is not None and getattr(model, "name", None):
            return model.name
        return getattr(self.attacker, "name", None)

    @staticmethod
    def _first_model(results: Sequence[AttackResult]) -> Optional[str]:
        """Best-effort target model id from the first response that carries one."""
        for r in results:
            if r.response.model:
                return r.response.model
        return None

    @staticmethod
    def _with_identity(transforms: Optional[Sequence[Transform]]) -> list[Transform]:
        """Return the transform list with an Identity baseline guaranteed first."""
        items = list(transforms) if transforms else []
        if not any(_is_identity(t) for t in items):
            items.insert(0, Identity())
        return items

    @staticmethod
    def _with_null_defense(defenses: Optional[Sequence[Defense]]) -> list[Defense]:
        """Return the defense list with the 'none' baseline guaranteed first."""
        items = list(defenses) if defenses else []
        if not any(_is_null_defense(d) for d in items):
            items.insert(0, NullDefense())
        return items


def _is_identity(transform: Transform) -> bool:
    """True when ``transform`` is the no-op identity baseline."""
    return getattr(transform, "name", "") == Identity.name


def _is_null_defense(defense: Defense) -> bool:
    """True when ``defense`` is the no-op 'none' baseline."""
    return getattr(defense, "name", "") == NullDefense.name


def _empty_report(target: Target) -> ScanReport:
    """An empty report stand-in for a benchmark over an empty corpus."""
    return ScanReport(target_name=getattr(target, "name", "target"))


def run_benchmark(
    target: Target,
    attacks: Iterable[Attack],
    detectors: Optional[Sequence[Detector]] = None,
    *,
    transforms: Optional[Sequence[Transform]] = None,
    defenses: Optional[Sequence[Defense]] = None,
    attacker: Optional[object] = None,
    use_judge: bool = False,
    group_by: GroupBy = _technique_group,
    seed: Optional[int] = None,
    tool_version: str = "0.1.0",
) -> BenchmarkResult:
    """Convenience wrapper: build a :class:`BenchmarkRunner` and run it once.

    Args:
        target: The target to benchmark.
        attacks: The corpus to run.
        detectors: Detectors (default: offline heuristics only).
        transforms: Transform variants to sweep (Identity always included).
        defenses: Defense variants to sweep ("none" always included).
        attacker: Optional adaptive attacker folded into the baseline.
        use_judge: Whether judge verdicts take scoring precedence.
        group_by: Attack -> group function (default: by technique).
        seed: Reproducibility seed stamped on the metadata.
        tool_version: Version stamped on the metadata.

    Returns:
        The populated :class:`BenchmarkResult` scorecard.
    """
    runner = BenchmarkRunner(
        target,
        detectors,
        transforms=transforms,
        defenses=defenses,
        attacker=attacker,
        use_judge=use_judge,
        group_by=group_by,
        seed=seed,
        tool_version=tool_version,
    )
    return runner.run(attacks)
