"""The capability-paradox bench harness — ASR vs model-capability curve.

NICHE-STRATEGY.md sequencing #3 ("injectkit frontier-model run — the MCPTox
capability-paradox curve"). This module generalises the single-cell
:func:`injectkit.bench.harness.run_cell` over a *set* of target models ordered
along a configurable **capability axis**, producing the curve the MCPTox finding
predicts: that **more-capable models can be MORE susceptible** to tool poisoning
(MCPTox, arXiv:2508.14925). The arms race cannot be won by better models — which
is the portfolio's central validation of deterministic enforcement.

What it does
------------
One call, :func:`run_capability_sweep`, takes ``attack × {set of models} × seeds``
and, for each model:

* resolves a per-model capability score (from an explicit value, else the zoo
  entry's ``params_b`` — the metadata is read without any download);
* runs the existing :func:`run_cell` to get the three never-collapsed signals
  (substring-ASR, judge-ASR, StrongREJECT-mean), each with a **Wilson** CI and the
  full 8-field reproducibility stamp;
* records a :class:`CapabilityPoint` (capability score + the cell).

The points are **sorted along the capability axis** into a
:class:`CapabilityCurve`, which (a) exposes the ordered ``(capability, judge-ASR ±
CI)`` series the plot/table consumes, (b) builds a :class:`~injectkit.bench.leaderboard.Leaderboard`
(model × attack matrix, one column = one capability rung), and (c) reports a
**monotonicity verdict**: whether ASR rises with capability (the paradox), falls
(the conventional "bigger is safer" assumption), or is flat/mixed — with the
honest caveat that a curve over a handful of seeded points is *indicative*, not a
significance test.

Offline / testability (the critical constraint)
-----------------------------------------------
The whole sweep runs offline against the deterministic seam/mock-model path the
rest of the bench harness already uses: each model is a
:class:`~injectkit.bench.harness.ModelSpec` whose ``loader`` returns an in-process
seam (no torch, no download, no API key). The full test suite drives this path.

The **frontier-model run** — sweeping the real zoo entries (Llama-3.1-8B,
Qwen2.5-7B, Gemma-2-9B, Mistral-7B, Phi-4, GPT-OSS-20B) or the live
anthropic/ollama/openai targets — needs a GPU + multi-GB downloads or API keys and
is **DEFERRED-NO-GPU**. Its code path is real (the loader seam is the same one the
zoo + the live targets plug into) and is exercised here against tiny/offline
models; it is documented as a one-command step in ``docs/BENCHMARK.md`` and is NOT
faked.

ETHICS: every behavior is a BENIGN-canary robustness probe; "success" means the
target emitted the benign marker it was told to withhold, never harmful content.
DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from ..evaluators.heuristics import DEFAULT_TRIGGER
from ..judge import DEFAULT_EVAL_JUDGE
from .harness import ASRStat, CellResult, ModelSpec, run_cell
from .leaderboard import Leaderboard

__all__ = [
    "PARADOX",
    "INVERSE",
    "FLAT",
    "ModelUnderTest",
    "CapabilityPoint",
    "CapabilityCurve",
    "run_capability_sweep",
]

#: Monotonicity verdicts for the ASR-vs-capability curve. ``PARADOX`` is the
#: MCPTox finding (ASR rises with capability); ``INVERSE`` is the conventional
#: "more-capable is safer" assumption; ``FLAT`` is no clear trend (incl. mixed).
PARADOX = "capability_paradox"
INVERSE = "inverse"
FLAT = "flat"

#: The signal whose curve the verdict is computed on (the calibrated headline).
_VERDICT_SIGNAL = "judge_asr"


@dataclass
class ModelUnderTest:
    """One model on the capability axis: how to obtain it + where it sits.

    ``spec`` is the :class:`~injectkit.bench.harness.ModelSpec` the harness runs
    (an offline seam for the CPU done-check, or a real zoo/live loader on a GPU
    host). ``capability`` is the model's position on the axis; if ``None`` it is
    resolved from the spec's zoo entry ``params_b`` (metadata only — no download),
    and a model with neither an explicit score nor a zoo entry is rejected up front
    so the curve's x-axis is never silently undefined.

    ``label`` defaults to the spec name and is what the leaderboard/plot shows.
    """

    spec: ModelSpec
    capability: Optional[float] = None
    label: Optional[str] = None

    def resolved_label(self) -> str:
        """The display label (explicit ``label`` else the spec name)."""
        return self.label or self.spec.name

    def resolved_capability(self) -> float:
        """The capability-axis score: explicit value, else the zoo ``params_b``.

        Raises:
            ValueError: if neither an explicit ``capability`` nor a zoo entry with
                ``params_b`` is available — the axis must be defined for every point
                or the curve is meaningless.
        """
        if self.capability is not None:
            return float(self.capability)
        entry = self.spec.entry()
        params_b = getattr(entry, "params_b", None) if entry is not None else None
        if params_b is None:
            raise ValueError(
                f"model {self.resolved_label()!r} has no capability score: pass an "
                "explicit `capability=` or use a zoo model whose entry carries "
                "`params_b` (the capability axis must be defined for every point)."
            )
        return float(params_b)


def _as_mut(item: "ModelUnderTest | ModelSpec | str") -> ModelUnderTest:
    """Coerce a model input to a :class:`ModelUnderTest`.

    A bare :class:`ModelSpec`/name is allowed when it resolves to a zoo entry with
    ``params_b`` (so the axis is defined); otherwise the caller must pass a
    :class:`ModelUnderTest` with an explicit ``capability``.
    """
    if isinstance(item, ModelUnderTest):
        return item
    if isinstance(item, ModelSpec):
        return ModelUnderTest(spec=item)
    if isinstance(item, str):
        return ModelUnderTest(spec=ModelSpec(name=item))
    raise TypeError(
        f"cannot interpret {item!r} as a model under test; pass a ModelUnderTest, "
        "a ModelSpec, or a zoo model name."
    )


@dataclass
class CapabilityPoint:
    """One point on the curve: a model's capability score + its run cell.

    ``capability`` is the x-axis value; ``cell`` is the full
    :class:`~injectkit.bench.harness.CellResult` (the three signals ± CI + the
    8-field stamp). ``label`` is the display name.
    """

    label: str
    capability: float
    cell: CellResult

    def signal(self, name: str = _VERDICT_SIGNAL) -> ASRStat:
        """The named signal's :class:`ASRStat` (default judge-ASR)."""
        return getattr(self.cell, name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "capability": self.capability,
            "cell": self.cell.as_dict(),
        }


@dataclass
class CapabilityCurve:
    """The ASR-vs-capability curve: points sorted along the capability axis.

    Built by :func:`run_capability_sweep`. The points are kept sorted by ascending
    capability so :meth:`series` is plot-ready and the monotonicity
    :meth:`verdict` reads left-to-right. ``attack_id`` and the axis label are
    carried for the artifact header.
    """

    attack_id: str
    capability_axis: str = "params_b"
    points: list[CapabilityPoint] = field(default_factory=list)

    def series(self, signal: str = _VERDICT_SIGNAL) -> list[tuple[float, ASRStat]]:
        """The ordered ``[(capability, ASRStat), ...]`` series for ``signal``.

        Sorted by ascending capability — exactly the order a curve/scatter plots in
        and the table prints in.
        """
        return [(p.capability, p.signal(signal)) for p in self.sorted_points()]

    def sorted_points(self) -> list[CapabilityPoint]:
        """Points sorted by ascending capability (stable on ties by label)."""
        return sorted(self.points, key=lambda p: (p.capability, p.label))

    def verdict(self, signal: str = _VERDICT_SIGNAL) -> str:
        """The monotonicity verdict of the ASR-vs-capability curve.

        Compares the named signal's rate at the lowest-capability point against the
        highest: ``PARADOX`` if it rises with capability (the MCPTox finding),
        ``INVERSE`` if it falls ("more-capable is safer"), ``FLAT`` if it does not
        move (within a small epsilon) or there are fewer than two distinct points.

        This is an **indicative** read of a handful of seeded points, NOT a
        significance test — the honest frontier caveat applies (see the module
        docstring and ``docs/BENCHMARK.md``).
        """
        pts = self.sorted_points()
        if len(pts) < 2:
            return FLAT
        lo = pts[0].signal(signal).rate
        hi = pts[-1].signal(signal).rate
        eps = 1e-9
        if hi > lo + eps:
            return PARADOX
        if hi < lo - eps:
            return INVERSE
        return FLAT

    def leaderboard(self, *, title: Optional[str] = None) -> Leaderboard:
        """A :class:`~injectkit.bench.leaderboard.Leaderboard` of the curve.

        One cell per model (the matrix axes are the models seen, in
        capability-ascending order, and the single attack), so the existing
        CSV/JSON/Markdown exporters render the curve as the auditable model × attack
        matrix with every 8-field stamp attached.
        """
        board = Leaderboard(
            title=title
            or f"injectkit capability-paradox curve — {self.attack_id} "
            f"(x-axis: {self.capability_axis})"
        )
        for p in self.sorted_points():
            board.add(p.cell)
        return board

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable curve: ordered points + the verdict + the axis."""
        return {
            "attack_id": self.attack_id,
            "capability_axis": self.capability_axis,
            "verdict": self.verdict(),
            "points": [p.as_dict() for p in self.sorted_points()],
        }


def run_capability_sweep(
    attack_name: str,
    models: Sequence[Union["ModelUnderTest", ModelSpec, str]],
    behaviors: Sequence[Any],
    *,
    judge_id: str = DEFAULT_EVAL_JUDGE,
    num_seeds: int = 1,
    seeds: Optional[Sequence[int]] = None,
    backend: str = "hf",
    trigger: str = DEFAULT_TRIGGER,
    confidence: float = 0.95,
    capability_axis: str = "params_b",
    **cell_kwargs: Any,
) -> CapabilityCurve:
    """Run ``attack × {set of models} × seeds`` and build the capability curve.

    For each model this runs the existing :func:`run_cell` (same attack registry,
    judge layer, generation seam, Wilson CIs, and 8-field stamp) and records a
    :class:`CapabilityPoint` at the model's capability score. The points are sorted
    along the capability axis into a :class:`CapabilityCurve` whose
    :meth:`~CapabilityCurve.verdict` says whether ASR rises with capability — the
    MCPTox capability-paradox curve (arXiv:2508.14925).

    Args:
        attack_name: a white-box attack-registry key (e.g. ``"prefill"``).
        models: the model set — :class:`ModelUnderTest`s (with an explicit
            capability), bare :class:`ModelSpec`s, or zoo names (capability resolved
            from the zoo ``params_b``). Mixed inputs are allowed.
        behaviors: the shared benign-canary behavior set every model is graded on
            (the same corpus across the axis, so the corpus hash is identical and
            the only thing that varies is the model).
        judge_id: the EVAL judge id (default ``clean_cls``).
        num_seeds / seeds: seed control (forwarded to :func:`run_cell`).
        backend: generation backend recorded in the stamp.
        trigger: the benign-marker prefix.
        confidence: CI confidence level (default 0.95).
        capability_axis: the axis label recorded on the curve (default
            ``"params_b"`` — model parameter count in billions).
        **cell_kwargs: forwarded to :func:`run_cell` (e.g. ``cfg`` / ``gpu_hours``).

    Returns:
        A :class:`CapabilityCurve` of one :class:`CapabilityPoint` per model.

    Raises:
        ValueError: if ``models`` is empty, or any model has no capability score.
    """
    muts = [_as_mut(m) for m in models]
    if not muts:
        raise ValueError(
            "run_capability_sweep needs at least one model on the capability axis."
        )

    points: list[CapabilityPoint] = []
    for mut in muts:
        capability = mut.resolved_capability()  # raises early if undefined.
        cell = run_cell(
            attack_name,
            mut.spec,
            behaviors,
            judge_id=judge_id,
            num_seeds=num_seeds,
            seeds=seeds,
            backend=backend,
            trigger=trigger,
            confidence=confidence,
            **cell_kwargs,
        )
        points.append(
            CapabilityPoint(
                label=mut.resolved_label(),
                capability=capability,
                cell=cell,
            )
        )

    return CapabilityCurve(
        attack_id=attack_name,
        capability_axis=capability_axis,
        points=points,
    )
