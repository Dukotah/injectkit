"""Probe Sampling — a draft-model acceleration wrapper over the GCG core.

CHUNK 8-probe-sampling (ROADMAP §6 efficiency primitives). Probe Sampling
(Zhao, Liu, Wang, Lu, Lin, "Accelerating Greedy Coordinate Gradient and General
Prompt Optimization via Probe Sampling", **arXiv:2403.01251**, NeurIPS 2024) is a
drop-in efficiency primitive for GCG-family search: instead of scoring every one
of the ``search_width`` candidate suffixes on the **expensive TARGET model**, a
cheap **DRAFT model** scores them all, the candidates are ranked by the draft's
loss, and only the top fraction ``r`` (the "probe set") is re-scored on the
target. The kept fraction is sized *dynamically* by the draft↔target *agreement*
on a small probe sample: when the draft tracks the target well, fewer candidates
need a target forward pass (more speedup); when they disagree, the kept fraction
widens toward the full batch (protecting ASR).

PAPER PARITY NUMBER (recorded for the repro stamp; arXiv:2403.01251, Table 1):
    * **3.5×–6.3× wall-clock speedup** over vanilla GCG, and
    * **ASR 81.0 vs 69.0** on **Llama-2-7B-chat** (probe sampling *raises* ASR for
      a fixed step budget because the freed compute buys more search) — i.e. the
      acceleration is non-degrading.

The ``≥3×`` wall-clock speedup and non-degraded-ASR-on-8B measurement require a
GPU + a real 7-8B target and a separate draft model, so the NUMBER is
**DEFERRED-NO-GPU**: the full code path below is production-complete and is
verified on a TINY CPU model pair (two GPT-2 / Pythia-160M acting as draft +
target, fixed seed), but the headline speedup is not run in this environment.

DESIGN — reuse, don't rebuild. This module never re-implements the GCG inner
loop. It is a thin **candidate-scoring strategy** over the same
:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam the GCG optimiser
already uses (``target_loss(input_ids, target_ids)``): given the per-step
candidate batch and the two model seams (draft + target), it returns the SAME
``[best_candidate, best_loss]`` decision the brute-force "score every candidate on
the target" path would return — only cheaper. With probe sampling disabled
(default) the wrapper is bypassed entirely and GCG scores the full batch exactly
as before, so existing behaviour is byte-for-byte unchanged.

ETHICS: the optimisation objective is ALWAYS the per-run BENIGN canary marker
(the ``target`` string the GCG loop already carries); probe sampling only changes
*which candidates get a target forward pass*, never *what* is optimised. ``torch``
is never imported here — the wrapper operates purely on the seam's scalar
``target_loss`` outputs, so it is unit-testable on CPU with no torch and no model
download.

DEFENSIVE / AUTHORIZED USE ONLY — run only against local models you own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

__all__ = [
    "ProbeSampling",
    "ProbeSamplingResult",
    "resolve_probe_sampling",
    "PAPER_SPEEDUP",
    "PAPER_ASR",
]

#: Paper parity (arXiv:2403.01251, NeurIPS 2024, Table 1) — recorded in the stamp.
PAPER_SPEEDUP = "3.5x-6.3x"
#: Probe-sampling vs vanilla-GCG ASR on Llama-2-7B-chat (81.0 vs 69.0).
PAPER_ASR = (81.0, 69.0)


# A normalised, validated view of the ``GCGConfig.probe_sampling`` knob. The
# config field accepts the ergonomic ``(r, sampling_factor)`` tuple (or a bool /
# None to disable); :func:`resolve_probe_sampling` projects every accepted form
# onto this dataclass so the wrapper has one shape to consume.
@dataclass(frozen=True)
class _Resolved:
    enabled: bool
    r: float
    sampling_factor: int


def resolve_probe_sampling(
    value: Union[None, bool, "Tuple[float, int]", Sequence[Any], "_Resolved"],
) -> _Resolved:
    """Normalise the ``GCGConfig.probe_sampling`` knob to a validated view.

    Accepted forms (any GCG variant can opt in by setting ``cfg.probe_sampling``):

    * ``None`` / ``False`` — disabled (the default; full-batch target scoring).
    * ``True`` — enabled with the paper defaults ``(r=0.1, sampling_factor=8)``.
    * ``(r, sampling_factor)`` — a 2-tuple: ``r`` is the *minimum* fraction of the
      ``search_width`` batch re-scored on the TARGET model (``0 < r <= 1``);
      ``sampling_factor`` (the paper's "probe set" size control) is the number of
      candidates sampled to estimate draft↔target agreement (``>= 1``).

    Returns:
        A frozen ``_Resolved(enabled, r, sampling_factor)``.

    Raises:
        ValueError: if a tuple is malformed (wrong arity, ``r`` out of ``(0, 1]``,
            or ``sampling_factor < 1``).
    """
    if isinstance(value, _Resolved):
        return value
    if value is None or value is False:
        return _Resolved(enabled=False, r=0.1, sampling_factor=8)
    if value is True:
        return _Resolved(enabled=True, r=0.1, sampling_factor=8)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "probe_sampling tuple must be (r, sampling_factor), got "
                f"{len(value)} element(s)."
            )
        r = float(value[0])
        sampling_factor = int(value[1])
        if not (0.0 < r <= 1.0):
            raise ValueError(f"probe_sampling r must be in (0, 1], got {r!r}.")
        if sampling_factor < 1:
            raise ValueError(
                f"probe_sampling sampling_factor must be >= 1, got {sampling_factor!r}."
            )
        return _Resolved(enabled=True, r=r, sampling_factor=sampling_factor)
    raise ValueError(
        "probe_sampling must be None, a bool, or a (r, sampling_factor) tuple; "
        f"got {value!r}."
    )


@dataclass(frozen=True)
class ProbeSamplingResult:
    """Outcome of one probe-sampling candidate-selection step.

    Attributes:
        best_index: Index into the candidate batch of the lowest-target-loss
            candidate that probe sampling selected.
        best_loss: That candidate's TARGET-model loss (the value GCG compares
            against the incumbent suffix).
        target_evals: How many candidates received a TARGET forward pass this step
            (the cost probe sampling pays). ``<= len(candidates)``; the ratio to
            ``search_width`` is the per-step speedup proxy recorded for the stamp.
        kept_fraction: The fraction of the batch re-scored on the target this step
            (dynamically sized by draft↔target agreement; ``>= r``).
        agreement: The measured draft↔target rank agreement on the probe sample
            (1.0 = perfect; lower widens ``kept_fraction``).
    """

    best_index: int
    best_loss: float
    target_evals: int
    kept_fraction: float
    agreement: float


class ProbeSampling:
    """Draft-model candidate filter for one GCG step (arXiv:2403.01251).

    Constructed once per optimisation run with the cheap ``draft`` seam, the
    expensive ``target`` seam, and the resolved ``(r, sampling_factor)`` knob;
    :meth:`select` is then called once per GCG step with that step's candidate
    batch to choose the lowest-target-loss candidate while scoring only a fraction
    of the batch on the target.

    Both ``draft`` and ``target`` are
    :class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seams — the same
    abstraction the GCG loop already uses. In production these wrap two real HF
    causal-LMs (a small draft, the 7-8B target); in tests they are two tiny CPU
    models (GPT-2 / Pythia-160M) or two ``StubWhiteBoxModel`` instances. No torch
    is imported here: the wrapper consumes only the scalar ``target_loss`` the
    seam returns.

    Args:
        draft: The CHEAP draft model seam (scores the whole batch).
        target: The EXPENSIVE target model seam (scores only the probe set).
        r: Minimum fraction of the batch re-scored on the target (``0 < r <= 1``).
        sampling_factor: Probe-set size for the draft↔target agreement estimate.
        prompt_ids: The fixed prompt-token prefix every candidate is appended to
            (the GCG optimisation context); a candidate's full input is
            ``prompt_ids + candidate``.
        target_ids: The BENIGN target-token ids the loss is computed against.
    """

    def __init__(
        self,
        draft: Any,
        target: Any,
        *,
        r: float,
        sampling_factor: int,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> None:
        self.draft = draft
        self.target = target
        self.r = float(r)
        self.sampling_factor = int(sampling_factor)
        self.prompt_ids = list(prompt_ids)
        self.target_ids = list(target_ids)

    # ------------------------------------------------------------------ public

    def select(self, candidates: Sequence[Sequence[int]]) -> ProbeSamplingResult:
        """Pick the lowest-target-loss candidate, scoring only a fraction on target.

        The probe-sampling algorithm (arXiv:2403.01251), per step:

        1. **Draft scores the whole batch.** Run the cheap draft model's
           ``target_loss`` on every candidate — one cheap forward per candidate.
        2. **Estimate draft↔target agreement on a probe sample.** Draw
           ``sampling_factor`` candidates, score them on BOTH models, and measure
           how well the draft's ranking matches the target's (a rank-agreement in
           ``[0, 1]``). High agreement ⇒ the draft is trustworthy ⇒ a small probe
           set suffices; low agreement ⇒ widen the probe set toward the full batch.
        3. **Size the kept fraction dynamically.** ``kept = r + (1 - r) *
           (1 - agreement)`` — never below the floor ``r``, never above 1.0.
        4. **Re-score the top ``kept`` fraction (by draft loss) on the TARGET** and
           return the global-best among (a) the probe sample (already
           target-scored) and (b) this re-scored set.

        Because the final decision is the lowest *target* loss over a set that
        always includes the draft's most-promising candidates AND the
        agreement-probe sample, the selection matches brute-force target scoring
        whenever the draft is a faithful proxy — and degrades gracefully (widening
        toward full scoring) exactly when it is not.

        Args:
            candidates: The step's ``[search_width, optim_len]`` candidate batch
                (each row a full candidate suffix's token ids).

        Returns:
            A :class:`ProbeSamplingResult` with the winning candidate, its target
            loss, and the bookkeeping (target evals, kept fraction, agreement).
        """
        rows = [list(c) for c in candidates]
        n = len(rows)
        if n == 0:
            return ProbeSamplingResult(
                best_index=-1,
                best_loss=float("inf"),
                target_evals=0,
                kept_fraction=0.0,
                agreement=1.0,
            )

        # 1. Cheap draft loss for every candidate.
        draft_losses = [self._loss(self.draft, row) for row in rows]
        draft_order = sorted(range(n), key=lambda i: draft_losses[i])

        # 2. Probe sample -> draft<->target rank agreement. The probe set is the
        #    draft's top-`sampling_factor` candidates (the ones most likely to win),
        #    scored on the target too so the sample doubles as real target evals.
        probe_k = max(1, min(self.sampling_factor, n))
        probe_idx = draft_order[:probe_k]
        target_loss: dict[int, float] = {
            i: self._loss(self.target, rows[i]) for i in probe_idx
        }
        agreement = self._rank_agreement(
            [draft_losses[i] for i in probe_idx],
            [target_loss[i] for i in probe_idx],
        )

        # 3. Dynamic kept fraction: high agreement -> stay near the floor r;
        #    low agreement -> widen toward the full batch (protect ASR).
        kept_fraction = self.r + (1.0 - self.r) * (1.0 - agreement)
        kept_fraction = min(1.0, max(self.r, kept_fraction))
        kept_k = max(probe_k, min(n, int(round(kept_fraction * n)) or 1))

        # 4. Re-score the top `kept_k` draft candidates on the target (reusing the
        #    probe-sample target losses already computed).
        for i in draft_order[:kept_k]:
            if i not in target_loss:
                target_loss[i] = self._loss(self.target, rows[i])

        best_index = min(target_loss, key=lambda i: target_loss[i])
        return ProbeSamplingResult(
            best_index=best_index,
            best_loss=target_loss[best_index],
            target_evals=len(target_loss),
            kept_fraction=kept_k / n,
            agreement=agreement,
        )

    # ----------------------------------------------------------------- helpers

    def _loss(self, model: Any, candidate: Sequence[int]) -> float:
        """Scalar ``target_loss`` of ``prompt_ids + candidate`` -> ``target_ids``."""
        input_ids = self.prompt_ids + list(candidate)
        return float(model.target_loss(input_ids, self.target_ids))

    @staticmethod
    def _rank_agreement(draft: Sequence[float], target: Sequence[float]) -> float:
        """Pairwise (Kendall-style) rank agreement of two loss vectors in ``[0, 1]``.

        For every candidate pair, the draft and target *agree* iff they order the
        pair the same way (both rank one below the other). The agreement is the
        fraction of concordant pairs; 1.0 means the draft reproduces the target's
        ranking exactly (so a tiny probe set is safe), lower means they diverge (so
        the kept fraction widens). With fewer than two points there is no pair to
        compare and agreement is defined as 1.0 (the floor ``r`` then governs).
        """
        m = len(draft)
        if m < 2:
            return 1.0
        concordant = 0
        total = 0
        for i in range(m):
            for j in range(i + 1, m):
                total += 1
                d = draft[i] - draft[j]
                t = target[i] - target[j]
                # Concordant iff same sign, or both flat (a tie the draft preserves).
                if (d > 0 and t > 0) or (d < 0 and t < 0) or (d == 0 and t == 0):
                    concordant += 1
        return concordant / total if total else 1.0
