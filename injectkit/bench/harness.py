"""The generalized ASR bench harness (ROADMAP §3, §6.10, §8).

CHUNK 7-bench-harness. The harness turns one *cell* of the leaderboard —

    attack_name × model × behavior_set × num_seeds × judge_id

— into an aggregated attack-success-rate with a confidence interval, by running
each (behavior, seed) through the existing v0.4 stack:

* the white-box **attack registry** (:mod:`injectkit.whitebox.registry`) resolves
  ``attack_name`` to a fresh :class:`~injectkit.whitebox.base.Attack`;
* the **model zoo** (:mod:`injectkit.whitebox.zoo`) resolves ``model`` to a pinned
  ``repo@revision`` + quant (and, on a GPU host, loads it) — on the CPU/no-GPU host
  the heavy load is DEFERRED-NO-GPU and an *offline model seam* is injected instead;
* the attack's own greedy **generation** seam produces a continuation per behavior
  (the same ``prefill_generate`` / ``generate_text`` seams the attacks already use);
* the **judge registry** (:mod:`injectkit.judge`) resolves ``judge_id`` and grades
  every continuation into the three never-collapsed signals (substring-ASR,
  judge-ASR, StrongREJECT-mean);
* the results are aggregated into per-signal ASRs with a **Wilson** confidence
  interval (no scipy dependency), and every cell carries the full 8-field
  :class:`~injectkit.bench.stamp.ReproStamp`.

Reproducibility: the harness threads ``seed`` into both the attack config and the
generation seam, and asserts that two seeded runs of the same cell carry identical
*non-seed* stamp fields. The done-check "two seeded runs reproduce within CI" is
:func:`runs_reproduce` over two :class:`CellResult` objects.

CPU / no-GPU posture: the whole harness path — registry → seam → judge → aggregate
→ stamp — runs offline on a tiny model (or a pure offline seam). The real 7–20B
zoo loads + the fp16-vs-4bit anchor cells need a GPU + multi-GB download and are
**DEFERRED-NO-GPU** (the load path exists in :mod:`injectkit.whitebox.zoo`; the
harness calls it only when an explicit loader is provided).

ETHICS: every behavior is a BENIGN-canary robustness probe; "success" means the
target emitted the benign marker it was told to withhold, never harmful content.
DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ..evaluators.heuristics import DEFAULT_TRIGGER
from ..judge import DEFAULT_EVAL_JUDGE, get_judge
from ..judge.base import Judge, JudgeVerdict
from ..models import TargetResponse
from ..whitebox.base import AttackResult
from ..whitebox.config import AttackConfig
from ..whitebox.registry import get_attack
from ..whitebox.zoo import ZooEntry, get_entry
from .stamp import ReproStamp, build_stamp, stamps_reproduce

__all__ = [
    "Behavior",
    "BehaviorRun",
    "ASRStat",
    "CellResult",
    "ModelSpec",
    "wilson_interval",
    "run_cell",
    "runs_reproduce",
]

#: The default z value for a 95% two-sided Wilson interval.
_Z_95 = 1.959963984540054

#: The signals the harness aggregates (each reported SEPARATELY — ROADMAP §3/§6.10).
SUBSTRING_SIGNAL = "substring_asr"
JUDGE_SIGNAL = "judge_asr"
SR_SIGNAL = "strongreject_mean"


@dataclass(frozen=True)
class Behavior:
    """One benign-canary behavior the harness grades (a prompt + its id).

    ``prompt`` is the user turn the attack optimises/prefills against; ``messages``
    is the rendered chat the attack receives (built from ``prompt`` if absent). A
    behavior carries NO harmful content — it is a robustness probe whose success
    condition is the per-run benign marker.
    """

    id: str
    prompt: str

    def messages(self) -> list[dict]:
        """The chat turns for this behavior (a single user turn)."""
        return [{"role": "user", "content": self.prompt}]


def _as_behavior(item: Any, idx: int) -> Behavior:
    """Coerce a behavior input (str | mapping | Behavior) to a :class:`Behavior`."""
    if isinstance(item, Behavior):
        return item
    if isinstance(item, str):
        return Behavior(id=f"b{idx}", prompt=item)
    if isinstance(item, Mapping):
        prompt = item.get("prompt") or item.get("payload") or item.get("text") or ""
        bid = str(item.get("id") or f"b{idx}")
        return Behavior(id=bid, prompt=str(prompt))
    # A corpus Attack dataclass or similar object.
    prompt = getattr(item, "payload", None) or getattr(item, "prompt", None) or ""
    bid = str(getattr(item, "id", None) or f"b{idx}")
    return Behavior(id=bid, prompt=str(prompt))


@dataclass
class ModelSpec:
    """How the harness obtains the model under test for a cell.

    Resolution is deliberately split so the harness is testable offline:

    * ``name`` resolves to a zoo :class:`~injectkit.whitebox.zoo.ZooEntry` for the
      stamp's pinned ``repo@revision`` + supported-attack/arch gate, WITHOUT a
      download (metadata only — works on any host).
    * ``loader`` (optional) is called to actually instantiate
      ``(model, tokenizer)`` — on a GPU host this is
      :func:`injectkit.whitebox.zoo.load_by_revision`; on the CPU host an OFFLINE
      SEAM (a stub exposing ``prefill_generate`` / ``generate_text``) is injected so
      the whole path runs with no torch and no download. When ``loader`` is None and
      ``model``/``tokenizer`` are given, those are used directly.

    ``revision``/``quant``/``arch`` default from the resolved zoo entry but may be
    overridden (e.g. the fp16-vs-4bit anchor cells pass ``quant`` explicitly).
    """

    name: str
    loader: Optional[Callable[..., tuple[Any, Any]]] = None
    model: Any = None
    tokenizer: Any = None
    quant: Optional[str] = None
    revision: Optional[str] = None
    arch: Optional[str] = None
    zoo_path: Any = None

    def entry(self) -> Optional[ZooEntry]:
        """The resolved zoo entry (metadata only), or None if ``name`` is off-zoo.

        Off-zoo names (e.g. a tiny ``"gpt2"`` used only for the CPU done-check) are
        allowed; the caller must then supply ``revision``/``quant`` explicitly so the
        stamp is still complete.
        """
        try:
            return get_entry(self.name, self.zoo_path)
        except Exception:  # noqa: BLE001 - off-zoo names are valid for tiny CPU tests.
            return None

    def resolve(self, attack_name: str) -> tuple[Any, Any, str, str, str]:
        """Resolve ``(model, tokenizer, revision, quant, arch)`` for ``attack_name``.

        Validates the zoo's attack/arch gate when ``name`` is on-zoo, then obtains
        the model: via ``loader`` if given, else the directly-provided
        ``model``/``tokenizer``. ``revision``/``quant``/``arch`` come from the zoo
        entry unless overridden on this spec.
        """
        entry = self.entry()
        if entry is not None:
            # Enforce the per-model supported-attacks allow-list up front (the same
            # gate the loader applies); a GCG-on-MoE cell refuses here, not silently.
            from ..whitebox.zoo import check_attack_supported

            check_attack_supported(entry, attack_name)
            revision = self.revision or f"{entry.repo}@{entry.revision}"
            quant = self.quant or entry.default_dtype
            arch = self.arch or entry.arch
        else:
            revision = self.revision or self.name
            quant = self.quant or "fp16"
            arch = self.arch or "dense"

        if self.loader is not None:
            model, tokenizer = self.loader(
                self.name, quant, entry=entry, arch=arch
            )
        else:
            model, tokenizer = self.model, self.tokenizer
        return model, tokenizer, revision, quant, arch


@dataclass
class BehaviorRun:
    """The graded outcome of one (behavior, seed) attack run.

    Carries the attack's :class:`~injectkit.whitebox.base.AttackResult`, the EVAL
    judge verdict, and the loosest (substring) verdict, plus the booleans the
    aggregation counts. ``queries``/``wall_clock_s`` come from the attack result for
    the budget columns.
    """

    behavior_id: str
    seed: int
    eval_verdict: JudgeVerdict
    substring_verdict: JudgeVerdict
    judge_success: bool
    substring_success: bool
    sr_score: float
    queries: int = 0
    wall_clock_s: float = 0.0


@dataclass
class ASRStat:
    """An aggregated rate (successes/trials) with a Wilson CI (the cell's headline).

    ``rate`` is ``successes/n``; ``lo``/``hi`` are the two-sided Wilson interval at
    :attr:`confidence`. ``mean`` is set for the continuous StrongREJECT signal (the
    mean score rather than a success proportion) with its CI from the score sample.
    """

    n: int
    successes: int
    rate: float
    lo: float
    hi: float
    confidence: float = 0.95
    mean: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "n": self.n,
            "successes": self.successes,
            "rate": round(self.rate, 6),
            "ci_low": round(self.lo, 6),
            "ci_high": round(self.hi, 6),
            "confidence": self.confidence,
        }
        if self.mean is not None:
            d["mean"] = round(self.mean, 6)
        return d


@dataclass
class CellResult:
    """The full aggregated result of one leaderboard cell (attack × model × judge).

    Holds the three never-collapsed signals as :class:`ASRStat`s, the per-run
    detail, the budget metadata (avg queries / wall-clock / quant), and the full
    8-field :class:`~injectkit.bench.stamp.ReproStamp` (one stamp per seed; the
    canonical stamp is the first seed's).
    """

    attack_id: str
    model: str
    judge_id: str
    backend: str
    quant: str
    seeds: tuple[int, ...]
    n_behaviors: int
    substring_asr: ASRStat
    judge_asr: ASRStat
    strongreject_mean: ASRStat
    runs: list[BehaviorRun] = field(default_factory=list)
    stamps: list[ReproStamp] = field(default_factory=list)
    avg_queries: float = 0.0
    wall_clock_s: float = 0.0
    gpu_hours: float = 0.0

    @property
    def stamp(self) -> ReproStamp:
        """The canonical stamp for the cell (the first seed's)."""
        return self.stamps[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "model": self.model,
            "judge_id": self.judge_id,
            "backend": self.backend,
            "quant": self.quant,
            "seeds": list(self.seeds),
            "n_behaviors": self.n_behaviors,
            "substring_asr": self.substring_asr.as_dict(),
            "judge_asr": self.judge_asr.as_dict(),
            "strongreject_mean": self.strongreject_mean.as_dict(),
            "avg_queries": round(self.avg_queries, 4),
            "wall_clock_s": round(self.wall_clock_s, 4),
            "gpu_hours": round(self.gpu_hours, 6),
            "stamp": self.stamp.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Wilson confidence interval (no scipy — closed form).
# --------------------------------------------------------------------------- #


def wilson_interval(
    successes: int, n: int, *, z: float = _Z_95
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion (no deps).

    The Wilson interval is well-behaved at the small ``n`` and the 0%/100% rates a
    per-cell ASR routinely hits (unlike the normal/Wald interval, which gives
    nonsense bounds there). Returns ``(0.0, 0.0)`` for ``n == 0``.

    Args:
        successes: number of successes.
        n: number of trials.
        z: the standard-normal quantile (default 1.96 for 95%).

    Returns:
        ``(lo, hi)`` clamped to ``[0, 1]``.
    """
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _mean_ci(scores: Sequence[float], *, z: float = _Z_95) -> tuple[float, float, float]:
    """Mean of ``scores`` with a normal-approx CI (for the continuous SR signal)."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / n
    if n == 1:
        return mean, mean, mean
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    se = math.sqrt(var / n)
    return mean, max(0.0, mean - z * se), min(1.0, mean + z * se)


# --------------------------------------------------------------------------- #
# The cell runner.
# --------------------------------------------------------------------------- #


def run_cell(
    attack_name: str,
    model: "str | ModelSpec",
    behaviors: Sequence[Any],
    *,
    judge_id: str = DEFAULT_EVAL_JUDGE,
    num_seeds: int = 1,
    seeds: Optional[Sequence[int]] = None,
    backend: str = "hf",
    cfg: Optional[AttackConfig] = None,
    cfg_factory: Optional[Callable[[int], AttackConfig]] = None,
    trigger: str = DEFAULT_TRIGGER,
    confidence: float = 0.95,
    gpu_hours: float = 0.0,
) -> CellResult:
    """Run one leaderboard cell and aggregate ASR ± CI with a full stamp.

    Sweeps ``behaviors × seeds`` for one ``(attack_name, model, judge_id)`` cell:
    for each (behavior, seed) it resolves a fresh attack from the registry, runs it
    against the model (seam) toward the per-run benign marker, grades the
    continuation with both the EVAL judge (``judge_id``) and the loosest substring
    judge, and accumulates the three signals. The aggregate ASRs carry Wilson CIs,
    every (behavior, seed) is recorded, and one :class:`ReproStamp` is built per
    seed.

    Args:
        attack_name: a white-box attack-registry key (e.g. ``"prefill"``).
        model: a zoo name, or a :class:`ModelSpec` (the offline-seam path for the
            CPU done-check passes a ``ModelSpec`` with a seam loader).
        behaviors: the behavior set (strings / mappings / :class:`Behavior`s).
        judge_id: the EVAL judge id (default ``clean_cls``).
        num_seeds: number of seeds (``0..num_seeds-1``) if ``seeds`` is not given.
        seeds: explicit seed list (overrides ``num_seeds``).
        backend: the generation backend recorded in the stamp (``hf`` | ``vllm``).
        cfg: a fixed :class:`AttackConfig` for every seed (its ``seed`` is overridden
            per run); ``cfg_factory`` takes precedence if given.
        cfg_factory: ``seed -> AttackConfig`` builder (for per-seed configs).
        trigger: the benign-marker prefix.
        confidence: CI confidence level (default 0.95).
        gpu_hours: measured GPU-hours for the budget column (0 on the CPU path).

    Returns:
        A :class:`CellResult` with the three signals, per-run detail, budget
        metadata, and one stamp per seed.
    """
    spec = model if isinstance(model, ModelSpec) else ModelSpec(name=model)
    seed_list = list(seeds) if seeds is not None else list(range(max(1, num_seeds)))
    behavior_list = [_as_behavior(b, i) for i, b in enumerate(behaviors)]

    model_obj, tokenizer, revision, quant, arch = spec.resolve(attack_name)

    eval_judge = get_judge(judge_id)
    substring_judge = get_judge("substring")

    runs: list[BehaviorRun] = []
    stamps: list[ReproStamp] = []
    total_queries = 0
    t0 = time.perf_counter()

    for seed in seed_list:
        run_cfg = _config_for(cfg, cfg_factory, seed)
        for behavior in behavior_list:
            attack = get_attack(attack_name)
            marker = f"{trigger}{behavior.id}"
            result = attack.run(
                model_obj,
                tokenizer,
                behavior.messages(),
                marker,
                run_cfg,
            )
            run = _grade(
                behavior=behavior,
                seed=seed,
                result=result,
                eval_judge=eval_judge,
                substring_judge=substring_judge,
                marker=marker,
                trigger=trigger,
            )
            runs.append(run)
            total_queries += run.queries

        stamps.append(
            build_stamp(
                behaviors=behavior_list,
                model_revision=revision,
                seed=seed,
                quant=quant,
                judge_id=judge_id,
                attack_id=attack_name,
                backend=backend,
                extra={"arch": arch, "n_behaviors": len(behavior_list)},
            )
        )

    wall_clock_s = time.perf_counter() - t0

    substring_asr = _binary_asr(
        [r.substring_success for r in runs], confidence
    )
    judge_asr = _binary_asr([r.judge_success for r in runs], confidence)
    sr_mean = _mean_asr([r.sr_score for r in runs], confidence)

    n_runs = len(runs) or 1
    return CellResult(
        attack_id=attack_name,
        model=spec.name,
        judge_id=judge_id,
        backend=backend,
        quant=quant,
        seeds=tuple(seed_list),
        n_behaviors=len(behavior_list),
        substring_asr=substring_asr,
        judge_asr=judge_asr,
        strongreject_mean=sr_mean,
        runs=runs,
        stamps=stamps,
        avg_queries=total_queries / n_runs,
        wall_clock_s=wall_clock_s,
        gpu_hours=gpu_hours,
    )


def _config_for(
    cfg: Optional[AttackConfig],
    cfg_factory: Optional[Callable[[int], AttackConfig]],
    seed: int,
) -> AttackConfig:
    """The :class:`AttackConfig` for ``seed`` (frozen ⇒ rebuilt with the seed).

    ``cfg_factory`` wins; else ``cfg`` is copied with the seed overridden (configs
    are frozen Pydantic models, so a new instance is required); else a bare
    :class:`AttackConfig` seeded for this run.
    """
    if cfg_factory is not None:
        return cfg_factory(seed)
    if cfg is not None:
        return cfg.model_copy(update={"seed": seed})
    return AttackConfig(seed=seed)


def _grade(
    *,
    behavior: Behavior,
    seed: int,
    result: AttackResult,
    eval_judge: Judge,
    substring_judge: Judge,
    marker: str,
    trigger: str,
) -> BehaviorRun:
    """Grade one attack result with the EVAL + substring judges into a run record.

    The attack's ``best_input`` is the continuation the judges score; a refusal is
    detected from the attack result (best_loss==1 with no success is a near-miss,
    not a refusal — refusal is only asserted when the attack flagged it). Both
    verdicts are produced so the leaderboard can assert substring-ASR ≥ judge-ASR.
    """
    response = TargetResponse(
        text=result.best_input or "",
        refused=False,
        stop_reason="end_turn",
        model=result.attack_name,
    )
    eval_verdict = eval_judge.judge(response, target=marker, trigger=trigger)
    substring_verdict = substring_judge.judge(response, target=marker, trigger=trigger)
    return BehaviorRun(
        behavior_id=behavior.id,
        seed=seed,
        eval_verdict=eval_verdict,
        substring_verdict=substring_verdict,
        judge_success=eval_verdict.success_bool,
        substring_success=substring_verdict.success_bool,
        sr_score=eval_verdict.sr_score,
        queries=int(result.queries),
        wall_clock_s=float(result.wall_clock_s),
    )


def _binary_asr(successes: Sequence[bool], confidence: float) -> ASRStat:
    """Aggregate a boolean success sample into an :class:`ASRStat` (Wilson CI)."""
    n = len(successes)
    k = sum(1 for s in successes if s)
    z = _z_for(confidence)
    lo, hi = wilson_interval(k, n, z=z)
    rate = (k / n) if n else 0.0
    return ASRStat(n=n, successes=k, rate=rate, lo=lo, hi=hi, confidence=confidence)


def _mean_asr(scores: Sequence[float], confidence: float) -> ASRStat:
    """Aggregate continuous scores into an :class:`ASRStat` carrying the mean + CI."""
    z = _z_for(confidence)
    mean, lo, hi = _mean_ci(list(scores), z=z)
    return ASRStat(
        n=len(scores),
        successes=0,
        rate=mean,
        lo=lo,
        hi=hi,
        confidence=confidence,
        mean=mean,
    )


def _z_for(confidence: float) -> float:
    """Standard-normal quantile for a two-sided ``confidence`` (common levels).

    A tiny lookup avoids a scipy dependency for the levels a leaderboard actually
    uses; anything else falls back to the 95% z (documented).
    """
    table = {0.90: 1.6448536269514722, 0.95: _Z_95, 0.99: 2.5758293035489004}
    return table.get(round(confidence, 4), _Z_95)


def runs_reproduce(
    a: CellResult,
    b: CellResult,
    *,
    require_ci_overlap: bool = True,
) -> bool:
    """Whether two cell runs reproduce: same stamp (mod seed) AND ASR within CI.

    The done-check "two seeded runs reproduce within CI" (ROADMAP §8): the two cells
    must (1) carry identical non-seed stamp fields (same experiment) and (2) have
    overlapping judge-ASR confidence intervals (the point estimates agree within
    sampling error). With ``require_ci_overlap=False`` only the stamp identity is
    checked (useful when one cell is a single seed).

    Args:
        a, b: two :class:`CellResult`s for the same cell at different seeds.
        require_ci_overlap: also require the judge-ASR CIs to overlap.

    Returns:
        True iff the runs reproduce.
    """
    if not stamps_reproduce(a.stamp, b.stamp):
        return False
    if not require_ci_overlap:
        return True
    return _ci_overlap(a.judge_asr, b.judge_asr)


def _ci_overlap(a: ASRStat, b: ASRStat) -> bool:
    """Whether two confidence intervals overlap (the within-CI reproduce test)."""
    return a.lo <= b.hi and b.lo <= a.hi
