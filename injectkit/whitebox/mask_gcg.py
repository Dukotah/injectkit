"""Mask-GCG — token-position pruning for efficient GCG (arXiv:2509.06350).

CHUNK 9-igcg-faster-gcg, OPTIONAL completeness tier (ROADMAP §6.1 GCG family).
Mask-GCG ("Mask-GCG: Pruning Redundant Token Positions for Efficient Adversarial
Suffix Optimization", **arXiv:2509.06350**) observes that only a subset of a GCG
suffix's *positions* actually carry the attack: the rest are redundant. It learns
a per-position importance mask and **prunes** (freezes) the low-importance
positions, so each step only optimises the slots that matter — fewer candidate
evaluations for the same ASR.

This module implements the pruning mask on top of the proven shared
greedy-coordinate-gradient core (:class:`injectkit.attackers.gcg.GCGSuffixAttacker`,
driven through the :class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam —
never rebuilt): after a short warmup that optimises all positions, the per-position
importance is scored and the lowest-importance positions are frozen; subsequent
steps mutate only the kept (active) positions.

ETHICS — NON-NEGOTIABLE: the optimisation objective is ALWAYS the per-run BENIGN
canary marker; pruning only narrows *which slots* are mutated, never *what* is
optimised. ``torch`` is never imported here — importance scoring uses the seam's
scalar ``target_loss``, so it is unit-testable on CPU with no torch.

DEFERRED-NO-GPU: the full-scale efficiency number on a real 7-8B target needs a
GPU run; only the masking LOGIC/WIRING is verified on the tiny CPU model + stub
seam here. Ships as a FLAG-gated variant, not a blocker.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import Attack, AttackResult
from .config import AttackConfig, GCGConfig, MaskGCGConfig
from .registry import register
from .targets import advprefix_target

__all__ = [
    "MaskGCGAttack",
    "run",
    "position_importance",
    "prune_mask",
]


def position_importance(
    model: Any,
    prompt_ids: Sequence[int],
    suffix_ids: Sequence[int],
    target_ids: Sequence[int],
) -> list[float]:
    """Score each suffix POSITION's importance to the (benign) target loss.

    Mask-GCG keeps the positions that *matter*. A position's importance is
    estimated, seam-only and torch-free, by how much perturbing it changes the
    loss: replace that slot with a neutral filler id and measure ``|Δ loss|`` — a
    large change means the position carries signal (keep it); a near-zero change
    means it is redundant (prunable).

    Args:
        model: The :class:`WhiteBoxModel` seam.
        prompt_ids: The fixed prompt prefix.
        suffix_ids: The current suffix ids.
        target_ids: The benign target ids.

    Returns:
        One importance score per suffix position (higher = more important).
    """
    n = len(list(suffix_ids))
    if n == 0:
        return []
    base = float(model.target_loss(list(prompt_ids) + list(suffix_ids), list(target_ids)))
    filler = list(suffix_ids)[0]
    importances: list[float] = []
    for slot in range(n):
        probe = filler if suffix_ids[slot] != filler else (filler + 1)
        trial = list(suffix_ids)
        trial[slot] = probe
        loss = float(model.target_loss(list(prompt_ids) + trial, list(target_ids)))
        importances.append(abs(loss - base))
    return importances


def prune_mask(
    importances: Sequence[float],
    *,
    keep_fraction: float,
    min_active: int = 1,
) -> list[bool]:
    """Build a keep-mask from per-position importances (Mask-GCG pruning).

    Keeps the top ``keep_fraction`` most-important positions (at least
    ``min_active``), freezes the rest. Ties are broken toward earlier positions
    for determinism.

    Args:
        importances: Per-position importance scores (higher = keep).
        keep_fraction: Fraction of positions to keep active (``0 < x <= 1``).
        min_active: Minimum positions always kept (so a short suffix is not frozen).

    Returns:
        A boolean keep-mask, one entry per position (``True`` ⇒ optimise this slot).
    """
    n = len(importances)
    if n == 0:
        return []
    keep_k = max(int(min_active), int(round(max(0.0, min(1.0, keep_fraction)) * n)) or 1)
    keep_k = min(n, keep_k)
    # Order positions by descending importance, stable on index for ties.
    order = sorted(range(n), key=lambda i: (-float(importances[i]), i))
    keep = set(order[:keep_k])
    return [i in keep for i in range(n)]


@register("mask_gcg")
class MaskGCGAttack(Attack):
    """Mask-GCG as a v0.4 :class:`~injectkit.whitebox.base.Attack` (arXiv:2509.06350).

    Token-position pruning over the proven
    :class:`injectkit.attackers.gcg.GCGSuffixAttacker` coordinate-descent loop —
    the inner ``token_gradients``/``target_loss`` machinery is reused verbatim
    through the :class:`WhiteBoxModel` seam, never reimplemented. After
    ``warmup_steps`` optimising all positions, the importance mask is computed and
    subsequent steps mutate only the kept positions.

    Dense-only like the rest of the gradient family. The objective is ALWAYS the
    benign per-run marker.
    """

    name = "mask_gcg"
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
        """Optimise a benign-marker suffix with token-position pruning.

        See the module docstring for the algorithm. ``cfg`` may be any
        :class:`AttackConfig`; a non-:class:`MaskGCGConfig` is coerced to Mask-GCG
        defaults plus the shared GCG knobs.
        """
        mcfg = cfg if isinstance(cfg, MaskGCGConfig) else _as_mask_config(cfg)
        prompt = _last_user_content(messages)
        model_name = getattr(model, "name", "") or ""

        if not target:
            target = advprefix_target(
                model_name, trigger=mcfg.trigger, use_baseline=not mcfg.use_advprefix
            )

        from ..attackers.gcg import GCGSuffixAttacker

        legacy = mcfg.to_legacy()
        attacker = GCGSuffixAttacker(
            model, legacy, init_suffix=mcfg.init_suffix, name=self.name
        )

        prompt_ids = list(model.token_ids(prompt))
        target_ids = list(model.token_ids(target))
        target_text = model.decode(target_ids)

        suffix_ids = list(model.token_ids(attacker.init_suffix)) or [0]
        steps: list[float] = []
        best_suffix = model.decode(suffix_ids)
        best_loss = float("inf")
        succeeded = False
        mask: Optional[list[bool]] = None  # None ⇒ all positions active (warmup)

        for step_no in range(1, mcfg.max_steps + 1):
            # After warmup, compute the prune mask once from position importance.
            if mask is None and step_no > mcfg.warmup_steps:
                imp = position_importance(model, prompt_ids, suffix_ids, target_ids)
                mask = prune_mask(
                    imp,
                    keep_fraction=mcfg.keep_fraction,
                    min_active=mcfg.min_active,
                )

            input_ids = list(prompt_ids) + list(suffix_ids)
            suffix_slice = slice(len(prompt_ids), len(input_ids))
            grads = model.token_gradients(input_ids, target_ids, suffix_slice)

            step_best_ids = list(suffix_ids)
            step_best_loss = float(
                model.target_loss(list(prompt_ids) + step_best_ids, target_ids)
            )
            for slot in range(len(suffix_ids)):
                if mask is not None and not mask[slot]:
                    continue  # frozen (pruned) position
                pool = attacker._top_k_candidates(grads, slot)
                if not pool:
                    continue
                for token_id in attacker._sample_candidates(pool):
                    if token_id == step_best_ids[slot]:
                        continue
                    trial = list(step_best_ids)
                    trial[slot] = token_id
                    loss = float(
                        model.target_loss(list(prompt_ids) + trial, target_ids)
                    )
                    if loss < step_best_loss:
                        step_best_loss = loss
                        step_best_ids = trial
            suffix_ids = step_best_ids

            suffix_text = model.decode(suffix_ids)
            steps.append(step_best_loss)
            if step_best_loss < best_loss:
                best_loss = step_best_loss
                best_suffix = suffix_text
            if bool(target_text) and target_text in suffix_text:
                succeeded = True
                break

        active = sum(mask) if mask is not None else len(suffix_ids)
        best_input = f"{prompt} {best_suffix}".rstrip() if best_suffix else prompt
        defense_id = getattr(defense, "name", "") if defense is not None else ""

        return AttackResult(
            attack_name=self.name,
            best_input=best_input,
            best_loss=best_loss,
            per_step_losses=list(steps),
            optimized_obj=best_suffix,
            optimized_obj_kind="suffix",
            succeeded=succeeded,
            queries=len(steps),
            defense_id=defense_id,
            stamp={"target": target, "active_positions": active},
        )


def _as_mask_config(cfg: AttackConfig) -> MaskGCGConfig:
    """Coerce any :class:`AttackConfig` to a :class:`MaskGCGConfig` (defaults)."""
    base = cfg.model_dump() if isinstance(cfg, GCGConfig) else {
        "max_steps": cfg.max_steps,
        "target": cfg.target,
        "trigger": cfg.trigger,
        "seed": cfg.seed,
    }
    allowed = set(MaskGCGConfig.model_fields)
    return MaskGCGConfig(**{k: v for k, v in base.items() if k in allowed})


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
    """First-class Mask-GCG entrypoint (arXiv:2509.06350) — pruned GCG in <10 lines.

        from injectkit.whitebox import mask_gcg
        result = mask_gcg.run(model, tok, messages, target, MaskGCGConfig(max_steps=2))

    A thin functional wrapper over :meth:`MaskGCGAttack.run` with a defaulted config.
    """
    return MaskGCGAttack().run(
        model, tokenizer, messages, target, cfg or MaskGCGConfig(), defense
    )
