"""Optional GCG variant primitives — momentum / MAGIC / SM-GCG (flag-gated tier).

CHUNK 9-igcg-faster-gcg, OPTIONAL completeness tier (ROADMAP §6.1 GCG family).
These are published GCG *refinements* that ship as **flags on the base
:class:`~injectkit.whitebox.config.GCGConfig`, never separate attacks or blockers**.
Each is a small, torch-free, unit-testable primitive that perturbs gradient
aggregation or candidate acceptance on the proven shared greedy-coordinate-gradient
core; with every flag at its default the behaviour is byte-for-byte plain GCG.

* **Momentum** (MAC, "Boosting Jailbreak Attack with Momentum",
  **arXiv:2405.01229**): blend each step's token gradient with an
  exponentially-decayed running average so the coordinate search carries
  inertia and escapes shallow local minima. See :class:`MomentumState`.
* **MAGIC** ("MAGIC: Mask-Guided/Adaptive Gradient-Informed Coordinate update",
  **arXiv:2412.08615**): grow the number of suffix slots updated per step from
  the gradient's own magnitude signal, saving queries vs single-coordinate GCG.
  See :func:`magic_coordinate_count`.
* **SM-GCG** (simulated-annealing / momentum candidate acceptance): accept a
  *non-improving* swap with a temperature-decayed probability to escape plateaus
  (a softened greedy acceptance). See :func:`sm_accept` / :func:`anneal_temperature`.

ETHICS — NON-NEGOTIABLE: none of these change the optimisation objective (always
the benign per-run canary marker); they only change *how the gradient is
aggregated* or *which swaps are accepted*. ``torch`` is never imported here — the
primitives operate on plain nested lists / scalars, so they are unit-testable on
CPU with no torch and no model download.

DEFERRED-NO-GPU: the published ASR/efficiency gains for each variant on a real
7-8B target need a GPU run; only the primitive LOGIC is verified on CPU here.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

__all__ = [
    "MomentumState",
    "magic_coordinate_count",
    "anneal_temperature",
    "sm_accept",
]


class MomentumState:
    """Exponentially-decayed running average of the token gradient (arXiv:2405.01229).

    MAC accelerates GCG by giving the coordinate search *inertia*: the gradient
    used to pick candidates at step ``t`` is ``m_t = beta * m_{t-1} + (1 - beta) *
    g_t`` rather than the raw ``g_t``. With ``beta == 0`` this is exactly plain
    GCG (``m_t == g_t``), so the flag is a no-op at its default.

    The state is a ``[suffix_len, vocab]`` grid of floats; :meth:`blend` accepts a
    fresh gradient grid (nested lists, or anything indexable/iterable row-wise) and
    returns the momentum-blended grid, updating the internal average in place.
    Resilient to a changing suffix length (e.g. multi-coordinate updates): when the
    new grid's shape differs from the stored average, it re-seeds from the new grid.
    """

    def __init__(self, beta: float = 0.0) -> None:
        self.beta = max(0.0, min(1.0, float(beta)))
        self._avg: Optional[list[list[float]]] = None

    def blend(self, grad: Sequence[Sequence[float]]) -> list[list[float]]:
        """Return the momentum-blended gradient grid and update the running average."""
        rows = [[float(x) for x in row] for row in grad]
        if self.beta <= 0.0:
            self._avg = rows
            return rows
        prev = self._avg
        same_shape = (
            prev is not None
            and len(prev) == len(rows)
            and all(len(p) == len(r) for p, r in zip(prev, rows))
        )
        if not same_shape:
            self._avg = rows
            return [list(r) for r in rows]
        blended = [
            [self.beta * pv + (1.0 - self.beta) * gv for pv, gv in zip(prow, grow)]
            for prow, grow in zip(prev, rows)  # type: ignore[arg-type]
        ]
        self._avg = blended
        return [list(r) for r in blended]


def magic_coordinate_count(
    grad: Sequence[Sequence[float]],
    *,
    max_coords: int,
    min_coords: int = 1,
) -> int:
    """Adaptive number of coordinates to update this step from the gradient signal.

    MAGIC (arXiv:2412.08615) grows the multi-coordinate update width from how
    *peaked* the gradient is: when many slots have a strong best-candidate signal
    (large most-negative entry relative to the mean), update more of them at once;
    when the signal is flat, update fewer. The count is the number of slots whose
    best-candidate magnitude exceeds the per-grid mean best-candidate magnitude,
    clamped to ``[min_coords, max_coords]``.

    Returns ``min_coords`` for an empty/degenerate grid (so callers always update
    at least one slot — identical to single-coordinate GCG).
    """
    rows = [[float(x) for x in row] for row in grad]
    if not rows:
        return max(1, int(min_coords))
    # Per-slot "best candidate strength" = magnitude of the most-negative entry.
    strengths = [abs(min(row)) if row else 0.0 for row in rows]
    mean = sum(strengths) / len(strengths) if strengths else 0.0
    count = sum(1 for s in strengths if s > mean)
    lo = max(1, int(min_coords))
    hi = max(lo, int(max_coords))
    return max(lo, min(hi, count or lo))


def anneal_temperature(initial: float, step: int, *, decay: float = 0.95) -> float:
    """Temperature at ``step`` under geometric annealing ``initial * decay**step``.

    SM-GCG cools the acceptance temperature over the run so early steps explore
    (accept some non-improving swaps to escape plateaus) and late steps converge
    to strict greedy. ``initial <= 0`` ⇒ ``0.0`` (strict greedy throughout).
    """
    if initial <= 0.0:
        return 0.0
    decay = max(0.0, min(1.0, float(decay)))
    return float(initial) * (decay ** max(0, int(step)))


def sm_accept(
    delta: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    """Simulated-annealing acceptance of a swap with loss change ``delta``.

    Always accept an improvement (``delta <= 0``). For a non-improving swap
    (``delta > 0``), accept with the Metropolis probability ``exp(-delta / T)`` —
    so at high ``T`` the search escapes plateaus, and as ``T`` cools it becomes
    strictly greedy. ``temperature <= 0`` ⇒ strict greedy (reject every
    non-improving swap), i.e. exactly plain GCG acceptance.

    Args:
        delta: ``new_loss - current_loss`` (negative ⇒ improvement).
        temperature: The current annealing temperature (``<= 0`` ⇒ greedy).
        rng: A seeded :class:`random.Random` for reproducibility.

    Returns:
        Whether to accept the swap.
    """
    if delta <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    prob = math.exp(-float(delta) / float(temperature))
    return rng.random() < prob
