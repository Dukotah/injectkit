"""Faster-GCG — efficient discrete-optimization GCG (Li et al., arXiv:2410.15362).

CHUNK 9-igcg-faster-gcg (ROADMAP §6.1 GCG family). Faster-GCG ("Faster-GCG:
Efficient Discrete Optimization Jailbreak Attacks Against Aligned Large Language
Models", **arXiv:2410.15362**) accelerates GCG's discrete search with three
changes; this module implements all three on top of the proven shared
greedy-coordinate-gradient core (:class:`injectkit.attackers.gcg.GCGSuffixAttacker`,
driven through the :class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam —
never rebuilt):

1. **Distance-regularized gradient estimation** — the greedy one-hot gradient
   over-trusts the linearisation for candidate tokens whose embedding is *far*
   from the current token. Faster-GCG adds a regulariser that penalises a
   candidate by its embedding distance from the current token, so the candidate
   *score* stays accurate near the operating point and the search wastes fewer
   forward passes on mis-ranked far-away tokens. See :func:`distance_regularized_scores`.
2. **Temperature sampling over candidates** — instead of a hard top-``k`` cut,
   candidates are sampled with a softmax over their (regularised) scores at a
   temperature, exploring promising-but-not-top tokens and reducing the redundancy
   of repeatedly re-trying the same hard-top-``k`` set. See :func:`temperature_sample`.
3. **Visited-set deduplication** — a bounded set of already-evaluated suffixes is
   kept so the search never re-scores a candidate it has already seen, breaking
   the cycles that waste GCG queries near a plateau. See :class:`VisitedSet`.

ETHICS — NON-NEGOTIABLE: the optimisation objective is ALWAYS the per-run BENIGN
canary marker ``<trigger><canary>``; Faster-GCG only changes *how candidates are
scored / sampled / de-duplicated*, never *what* is optimised. ``torch`` is never
imported here — the helper logic is pure Python over plain id lists and scalar
scores, so it is unit-testable on CPU with no torch and no model download. The
attack drives the same :class:`WhiteBoxModel` seam the GCG loop already uses.

DEFERRED-NO-GPU: the headline wall-clock speedup over GCG on a real 7-8B target
(arXiv:2410.15362) needs a GPU run to TIME; only the LOGIC/WIRING of the three
mechanisms is verified on the tiny CPU model + stub seam here. The full code path
is production-complete.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional, Sequence

from .base import Attack, AttackResult
from .config import AttackConfig, FasterGCGConfig, GCGConfig
from .registry import register
from .targets import advprefix_target

__all__ = [
    "FasterGCGAttack",
    "run",
    "VisitedSet",
    "distance_regularized_scores",
    "temperature_sample",
    "PAPER_SPEEDUP",
]

#: Paper parity (arXiv:2410.15362) — Faster-GCG reports a large wall-clock speedup
#: over vanilla GCG at matched/better ASR. The NUMBER is DEFERRED-NO-GPU (needs a
#: real 7-8B GPU run to time); recorded here for the repro stamp.
PAPER_SPEEDUP = "wall-clock speedup vs GCG (DEFERRED-NO-GPU)"


# --------------------------------------------------------------------------- #
# (3) Visited-set deduplication
# --------------------------------------------------------------------------- #


class VisitedSet:
    """A bounded LRU set of already-evaluated suffixes (arXiv:2410.15362).

    Faster-GCG avoids the cycles GCG falls into near a plateau by never re-scoring
    a candidate suffix it has already evaluated. This is an order-preserving,
    capacity-bounded set: :meth:`add` records a suffix (evicting the oldest when
    full), :meth:`seen` tests membership. ``capacity == 0`` disables dedup (every
    candidate is treated as unseen — identical to plain GCG).

    Suffixes are keyed by their token-id tuple, so it is torch-free and
    deterministic.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = max(0, int(capacity))
        self._order: list[tuple[int, ...]] = []
        self._set: set[tuple[int, ...]] = set()

    def seen(self, suffix_ids: Sequence[int]) -> bool:
        """True iff ``suffix_ids`` has already been added (and dedup is enabled)."""
        if self.capacity == 0:
            return False
        return tuple(int(x) for x in suffix_ids) in self._set

    def add(self, suffix_ids: Sequence[int]) -> None:
        """Record ``suffix_ids``, evicting the oldest entry when at capacity."""
        if self.capacity == 0:
            return
        key = tuple(int(x) for x in suffix_ids)
        if key in self._set:
            return
        self._order.append(key)
        self._set.add(key)
        while len(self._order) > self.capacity:
            old = self._order.pop(0)
            self._set.discard(old)

    def __len__(self) -> int:
        return len(self._order)


# --------------------------------------------------------------------------- #
# (1) Distance-regularized gradient estimation
# --------------------------------------------------------------------------- #


def distance_regularized_scores(
    grad_row: Sequence[float],
    current_id: int,
    embeddings: Optional[Sequence[Sequence[float]]] = None,
    *,
    distance_reg_lambda: float = 0.1,
) -> list[float]:
    """Add an embedding-distance penalty to a slot's per-vocab gradient scores.

    Faster-GCG's first improvement: the raw one-hot gradient ``grad_row[v]`` (the
    estimated loss change from swapping the slot to token ``v``) is corrected by
    ``+ lambda * dist(emb[v], emb[current])`` so candidates *far* from the current
    token — where the linear gradient estimate is least trustworthy — are penalised
    and the search prefers nearby, reliably-scored swaps. Lower score ⇒ more
    promising (same convention as GCG's most-negative gradient).

    When no ``embeddings`` matrix is supplied (the offline seam path, which has no
    real embedding table) the penalty degrades to a token-id *index* distance
    ``|v - current_id|`` normalised by vocab size — a cheap, deterministic proxy
    that still exercises the regularisation logic on CPU with no torch.

    Args:
        grad_row: The ``[vocab]`` gradient scores for one suffix slot.
        current_id: The slot's current token id (distance is measured from this).
        embeddings: Optional ``[vocab, d]`` embedding matrix (real model path).
        distance_reg_lambda: The regulariser coefficient λ (``>= 0``; ``0`` ⇒ off).

    Returns:
        A ``[vocab]`` list of distance-regularised scores (lower = more promising).
    """
    row = [float(x) for x in grad_row]
    lam = float(distance_reg_lambda)
    if lam <= 0.0 or not row:
        return row
    vocab = len(row)
    if embeddings is not None and len(embeddings) >= vocab and current_id < len(embeddings):
        cur = [float(x) for x in embeddings[current_id]]
        out: list[float] = []
        for v in range(vocab):
            ev = embeddings[v]
            dist = math.sqrt(sum((float(a) - b) ** 2 for a, b in zip(ev, cur)))
            out.append(row[v] + lam * dist)
        return out
    # Offline proxy: normalised token-id index distance.
    return [row[v] + lam * (abs(v - current_id) / max(1, vocab)) for v in range(vocab)]


# --------------------------------------------------------------------------- #
# (2) Temperature sampling over candidates
# --------------------------------------------------------------------------- #


def temperature_sample(
    scores: Sequence[float],
    *,
    k: int,
    temperature: float,
    rng: random.Random,
    candidate_ids: Optional[Sequence[int]] = None,
) -> list[int]:
    """Sample ``k`` candidate token ids by softmax over ``-scores / temperature``.

    Faster-GCG's second improvement replaces the hard top-``k`` cut with a
    *temperature softmax* over the candidate scores: a lower score (more promising)
    gets higher probability, but promising-but-not-top tokens still get sampled —
    exploring more of the landscape and avoiding the redundancy of always trying
    the same top-``k``. Sampling is WITHOUT replacement and deterministic given
    ``rng``.

    Args:
        scores: Per-candidate scores (lower = more promising; the
            distance-regularised gradient row, optionally already top-k-restricted).
        k: How many distinct candidate ids to draw.
        temperature: Softmax temperature (higher ⇒ flatter / more exploratory).
        rng: A seeded :class:`random.Random` for reproducibility.
        candidate_ids: Optional ids the ``scores`` correspond to (defaults to the
            positional indices ``0..len(scores)-1``, i.e. the vocab ids).

    Returns:
        ``k`` distinct token ids sampled by the softmax (fewer if the pool is
        smaller than ``k``).
    """
    ids = list(candidate_ids) if candidate_ids is not None else list(range(len(scores)))
    vals = [float(s) for s in scores]
    if not ids:
        return []
    k = max(1, min(int(k), len(ids)))
    temp = max(1e-6, float(temperature))

    # Softmax over -score/temp (lower score ⇒ higher weight), numerically stable.
    logits = [-(v / temp) for v in vals]
    m = max(logits)
    weights = [math.exp(l - m) for l in logits]

    chosen: list[int] = []
    pool = list(range(len(ids)))
    pool_w = list(weights)
    for _ in range(k):
        total = sum(pool_w)
        if total <= 0.0 or not pool:
            break
        r = rng.random() * total
        acc = 0.0
        pick = 0
        for idx, w in enumerate(pool_w):
            acc += w
            if acc >= r:
                pick = idx
                break
        chosen.append(ids[pool[pick]])
        pool.pop(pick)
        pool_w.pop(pick)
    return chosen


# --------------------------------------------------------------------------- #
# The registered Faster-GCG attack
# --------------------------------------------------------------------------- #


@register("faster_gcg")
class FasterGCGAttack(Attack):
    """Faster-GCG as a v0.4 :class:`~injectkit.whitebox.base.Attack` (arXiv:2410.15362).

    Composes the three Faster-GCG accelerations (distance-regularized scoring,
    temperature candidate sampling, visited-set dedup) over the proven
    :class:`injectkit.attackers.gcg.GCGSuffixAttacker` coordinate-descent loop —
    the inner ``token_gradients``/``target_loss`` machinery is reused verbatim
    through the :class:`WhiteBoxModel` seam, never reimplemented.

    Dense-only like the rest of the gradient family. The objective is ALWAYS the
    benign per-run marker.
    """

    name = "faster_gcg"
    supported_arch = {"dense"}

    def run(
        self,
        model: Any,
        tokenizer: Any,
        messages: list[dict],
        target: str,
        cfg: AttackConfig,
        defense: "Optional[object]" = None,
    ) -> AttackResult:
        """Optimise a benign-marker suffix with the three Faster-GCG accelerations.

        See the module docstring for the algorithm. ``cfg`` may be any
        :class:`AttackConfig`; a non-:class:`FasterGCGConfig` is coerced to
        Faster-GCG defaults plus the shared GCG knobs. Returns a v0.4
        :class:`~injectkit.whitebox.base.AttackResult`.
        """
        fcfg = cfg if isinstance(cfg, FasterGCGConfig) else _as_faster_config(cfg)
        prompt = _last_user_content(messages)
        model_name = getattr(model, "name", "") or ""

        if not target:
            target = advprefix_target(
                model_name, trigger=fcfg.trigger, use_baseline=not fcfg.use_advprefix
            )

        from ..attackers.gcg import GCGSuffixAttacker

        legacy = fcfg.to_legacy()
        attacker = GCGSuffixAttacker(
            model, legacy, init_suffix=fcfg.init_suffix, name=self.name
        )

        prompt_ids = list(model.token_ids(prompt))
        target_ids = list(model.token_ids(target))
        embeddings = _maybe_embeddings(model)

        steps, best_suffix, best_loss, succeeded = self._faster_loop(
            model, attacker, fcfg, prompt_ids, target_ids, embeddings
        )

        best_input = f"{prompt} {best_suffix}".rstrip() if best_suffix else prompt
        defense_id = getattr(defense, "name", "") if defense is not None else ""

        return AttackResult(
            attack_name=self.name,
            best_input=best_input,
            best_loss=best_loss,
            per_step_losses=[loss for _, loss in steps],
            optimized_obj=best_suffix,
            optimized_obj_kind="suffix",
            succeeded=succeeded,
            queries=len(steps),
            defense_id=defense_id,
            stamp={"target": target, "paper_speedup": PAPER_SPEEDUP},
        )

    def _faster_loop(
        self,
        model: Any,
        attacker: Any,
        fcfg: FasterGCGConfig,
        prompt_ids: list[int],
        target_ids: list[int],
        embeddings: Optional[Sequence[Sequence[float]]],
    ) -> tuple[list[tuple[int, float]], str, float, bool]:
        """Run the Faster-GCG coordinate loop, reusing the GCG seam primitives.

        Per step: compute the gradient; for each slot apply the distance
        regulariser (improvement 1) and draw candidates by temperature sampling
        (improvement 2) instead of a hard top-``k``; score each candidate suffix
        on the seam, *skipping* any already in the visited set (improvement 3),
        keep the best, and record the visited suffix. Stop on a benign-marker
        success or at the step budget. The candidate top-k restriction and seam
        scoring reuse the attacker's proven helpers.
        """
        rng = random.Random(fcfg.seed)
        visited = VisitedSet(fcfg.visited_set_size)
        target_text = model.decode(target_ids)

        suffix_ids = list(model.token_ids(attacker.init_suffix)) or [0]
        visited.add(suffix_ids)
        steps: list[tuple[int, float]] = []
        best_suffix = model.decode(suffix_ids)
        best_loss = float("inf")
        succeeded = False

        for step_no in range(1, fcfg.max_steps + 1):
            input_ids = list(prompt_ids) + list(suffix_ids)
            suffix_slice = slice(len(prompt_ids), len(input_ids))
            grads = model.token_gradients(input_ids, target_ids, suffix_slice)

            step_best_ids = list(suffix_ids)
            step_best_loss = float(
                model.target_loss(list(prompt_ids) + step_best_ids, target_ids)
            )
            for slot in range(len(suffix_ids)):
                pool = attacker._top_k_candidates(grads, slot)
                if not pool:
                    continue
                # (1) distance-regularize the (restricted) candidate scores.
                grad_row = _grad_row(grads, slot)
                scores = [
                    distance_regularized_scores(
                        grad_row,
                        suffix_ids[slot],
                        embeddings,
                        distance_reg_lambda=fcfg.distance_reg_lambda,
                    )[tid]
                    for tid in pool
                ]
                # (2) temperature-sample candidates (or hard top-k when off).
                if fcfg.temp_sampling:
                    sampled = temperature_sample(
                        scores,
                        k=min(fcfg.batch_size, len(pool)),
                        temperature=fcfg.temperature,
                        rng=rng,
                        candidate_ids=pool,
                    )
                else:
                    order = sorted(range(len(pool)), key=lambda i: scores[i])
                    sampled = [pool[i] for i in order[: fcfg.batch_size]]

                for token_id in sampled:
                    if token_id == step_best_ids[slot]:
                        continue
                    trial = list(step_best_ids)
                    trial[slot] = token_id
                    # (3) visited-set: skip suffixes already evaluated.
                    if visited.seen(trial):
                        continue
                    visited.add(trial)
                    loss = float(
                        model.target_loss(list(prompt_ids) + trial, target_ids)
                    )
                    if loss < step_best_loss:
                        step_best_loss = loss
                        step_best_ids = trial
            suffix_ids = step_best_ids

            suffix_text = model.decode(suffix_ids)
            steps.append((step_no, step_best_loss))
            if step_best_loss < best_loss:
                best_loss = step_best_loss
                best_suffix = suffix_text
            if bool(target_text) and target_text in suffix_text:
                succeeded = True
                break

        return steps, best_suffix, best_loss, succeeded


def _grad_row(grads: Any, slot: int) -> list[float]:
    """Read slot ``slot`` of a ``[suffix_len, vocab]`` gradient grid as a float list."""
    try:
        row = grads[slot]
    except (IndexError, KeyError, TypeError):
        return []
    tolist = getattr(row, "tolist", None)
    if callable(tolist):
        row = tolist()
    try:
        return [float(x) for x in row]
    except TypeError:
        return []


def _maybe_embeddings(model: Any) -> Optional[Sequence[Sequence[float]]]:
    """Return the model's embedding matrix as nested lists if cheaply available.

    Real HF models expose ``get_input_embeddings().weight``; the offline seam does
    not, so this returns ``None`` and the distance regulariser falls back to its
    deterministic token-id-index proxy. Never imports torch; only reads attributes.
    """
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        return None
    try:
        weight = getter().weight
        tolist = getattr(weight, "tolist", None)
        return tolist() if callable(tolist) else None
    except Exception:  # noqa: BLE001 - any failure ⇒ use the proxy
        return None


def _as_faster_config(cfg: AttackConfig) -> FasterGCGConfig:
    """Coerce any :class:`AttackConfig` to a :class:`FasterGCGConfig` (defaults)."""
    base = cfg.model_dump() if isinstance(cfg, GCGConfig) else {
        "max_steps": cfg.max_steps,
        "target": cfg.target,
        "trigger": cfg.trigger,
        "seed": cfg.seed,
    }
    allowed = set(FasterGCGConfig.model_fields)
    return FasterGCGConfig(**{k: v for k, v in base.items() if k in allowed})


def _last_user_content(messages: list[dict]) -> str:
    """Content of the last ``user`` turn (or the last turn, or "")."""
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(messages[-1].get("content", ""))


def run(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    target: str,
    cfg: Optional[AttackConfig] = None,
    *,
    defense: "Optional[object]" = None,
) -> AttackResult:
    """First-class Faster-GCG entrypoint (arXiv:2410.15362) — faster GCG in <10 lines.

        from injectkit.whitebox import faster_gcg
        result = faster_gcg.run(model, tok, messages, target, FasterGCGConfig(max_steps=1))

    A thin functional wrapper over :meth:`FasterGCGAttack.run` with a defaulted config.
    """
    return FasterGCGAttack().run(
        model, tokenizer, messages, target, cfg or FasterGCGConfig(), defense
    )
