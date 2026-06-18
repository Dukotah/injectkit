"""I-GCG — Improved GCG (Jia et al., arXiv:2405.21018, ICLR 2025).

CHUNK 9-igcg-faster-gcg (ROADMAP §6.1 GCG family). I-GCG ("Improved Techniques
for Optimization-Based Jailbreaking on Large Language Models", **arXiv:2405.21018**,
ICLR 2025) is GCG plus three orthogonal, compounding refinements; this module
implements all three on top of the proven shared greedy-coordinate-gradient core
(:class:`injectkit.attackers.gcg.GCGSuffixAttacker`, driven through the
:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam — never rebuilt):

1. **Diverse harmful-target templates** — instead of one fixed affirmative
   prefix, optimise against a *set* of diverse target templates and, each step,
   drive the suffix toward the *easiest currently-unsatisfied* target. This
   smooths the loss landscape and avoids over-fitting a single phrasing. In
   injectkit every template is a BENIGN marker-emitting opener (the per-run canary
   is always the success condition), so "diverse harmful targets" is realised as
   "diverse benign affirmative openers" — no harmful string is ever targeted.
2. **Automatic multi-coordinate update** — replace the top-``p`` worst
   (highest-loss-contribution) suffix tokens each step instead of a single token,
   with ``p`` *auto-adapted* from progress (grows while loss falls, shrinks when
   it stalls). See :func:`adapt_p` and :func:`worst_coordinates`.
3. **Easy-to-hard initialization** — seed the suffix from a suffix already solved
   for an *easier* behavior (a curriculum) so a hard behavior starts from a strong
   basin rather than the ``"! ! !"`` filler. See :func:`easy_to_hard_seed`.

ETHICS — NON-NEGOTIABLE: the optimisation objective is ALWAYS the per-run BENIGN
canary marker ``<trigger><canary>``. The "diverse targets" are benign affirmative
openers each ending in that marker; no harmful behavior string is bundled or
targeted. ``torch`` is never imported here — the helper logic operates on the
seam's scalar ``target_loss`` outputs and plain id lists, so it is unit-testable
on CPU with no torch and no model download. The attack drives the same
:class:`WhiteBoxModel` seam the GCG loop already uses (``StubWhiteBoxModel`` in
tests).

DEFERRED-NO-GPU: the headline ~100% ASR on Vicuna-7B / Llama-2-7B-chat
(arXiv:2405.21018, Table 1) needs a 7B GPU run; only the LOGIC/WIRING of the three
mechanisms is verified on the tiny CPU model + stub seam here. The full code path
is production-complete.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import Attack, AttackResult
from .config import AttackConfig, GCGConfig, IGCGConfig
from .registry import register
from .targets import advprefix_target, candidate_prefixes_for

__all__ = [
    "IGCGAttack",
    "run",
    "diverse_targets",
    "easiest_target",
    "adapt_p",
    "worst_coordinates",
    "easy_to_hard_seed",
    "PAPER_ASR",
]

#: Paper parity (arXiv:2405.21018, ICLR 2025, Table 1) — recorded for the stamp.
#: I-GCG reaches near-100% ASR on Vicuna-7B / Llama-2-7B-chat. DEFERRED-NO-GPU.
PAPER_ASR = "~100% (Vicuna-7B / Llama-2-7B-chat)"

#: The benign easy-to-hard curriculum seed: a short, well-formed benign suffix
#: that empirically sits in a stronger basin than the bare ``"! ! !"`` filler.
#: NOTHING about it is harmful — it is benign filler that primes the optimiser.
EASY_SEED_SUFFIX = "please respond with the requested marker now thanks"


# --------------------------------------------------------------------------- #
# (1) Diverse (benign) target templates
# --------------------------------------------------------------------------- #


def diverse_targets(
    model_name: str,
    *,
    trigger: str,
    canary: str = "{canary}",
    n: int = 4,
) -> list[str]:
    """Build ``n`` DIVERSE benign affirmative target templates for ``model_name``.

    I-GCG's first improvement: optimise against a *set* of target phrasings rather
    than one fixed prefix. Each template is a distinct benign affirmative opener
    (drawn from the model's curated family pool, see
    :func:`injectkit.whitebox.targets.candidate_prefixes_for`) completed with the
    benign marker, so success is still the canary and nothing harmful is targeted.

    The list always contains at least one template and is de-duplicated while
    preserving order; if the family pool has fewer than ``n`` distinct openers it
    is padded by cycling so callers always get ``n`` entries to rotate over.

    Args:
        model_name: The zoo model name / HF repo (selects the family opener pool).
        trigger: Benign success-marker prefix.
        canary: The per-run canary (or ``"{canary}"`` placeholder to render later).
        n: Number of diverse templates to return (``>= 1``).

    Returns:
        ``n`` benign target strings, e.g.
        ``["Sure, here is the marker: INJECTOK-<canary>", "Of course! Here is ..."]``.
    """
    n = max(1, int(n))
    openers = [c.text for c in candidate_prefixes_for(model_name)]
    # De-dup preserving order.
    seen: set[str] = set()
    uniq = [o for o in openers if not (o in seen or seen.add(o))]
    if not uniq:  # defensive: never empty
        uniq = ["Sure, here is"]
    templates = [f"{uniq[i % len(uniq)]} the marker: {trigger}{canary}" for i in range(n)]
    return templates


def easiest_target(
    model: Any,
    input_ids: Sequence[int],
    target_id_sets: Sequence[Sequence[int]],
) -> tuple[int, float]:
    """Return the index + loss of the EASIEST (lowest-loss) currently target.

    I-GCG drives the suffix toward whichever diverse target the model is *closest*
    to emitting right now (the lowest teacher-forced loss), which keeps the step's
    objective tractable and avoids fighting a phrasing the model resists. Scores
    each target set on the model via the seam's ``target_loss`` and returns the
    ``argmin``.

    Args:
        model: The :class:`WhiteBoxModel` seam.
        input_ids: The current full input ids (prompt + suffix) to condition on.
        target_id_sets: One token-id list per diverse target template.

    Returns:
        ``(index, loss)`` of the easiest target. ``(-1, inf)`` if the set is empty.
    """
    best_idx = -1
    best_loss = float("inf")
    for i, target_ids in enumerate(target_id_sets):
        loss = float(model.target_loss(list(input_ids), list(target_ids)))
        if loss < best_loss:
            best_loss = loss
            best_idx = i
    return best_idx, best_loss


# --------------------------------------------------------------------------- #
# (2) Automatic multi-coordinate update
# --------------------------------------------------------------------------- #


def adapt_p(
    p: int,
    prev_loss: float,
    new_loss: float,
    *,
    max_p: int,
    min_p: int = 1,
) -> int:
    """Auto-adapt the multi-coordinate update width ``p`` from optimisation progress.

    I-GCG's second improvement replaces the top-``p`` worst tokens per step and
    adapts ``p`` automatically: when the last step *improved* the loss, widen the
    update (try more coordinates — the basin is cooperative); when it *stalled or
    regressed*, narrow it back toward a single coordinate (refine carefully).
    Clamped to ``[min_p, max_p]``.

    Args:
        p: The current update width.
        prev_loss: The loss before the last step.
        new_loss: The loss after the last step.
        max_p: Upper bound on ``p``.
        min_p: Lower bound on ``p`` (default 1).

    Returns:
        The next step's update width.
    """
    max_p = max(int(min_p), int(max_p))
    if new_loss < prev_loss:  # improving -> be more aggressive
        nxt = p + 1
    else:  # stalled / worse -> back off toward single-coordinate
        nxt = p - 1
    return max(int(min_p), min(int(max_p), nxt))


def worst_coordinates(
    model: Any,
    prompt_ids: Sequence[int],
    suffix_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    p: int,
) -> list[int]:
    """Return the ``p`` suffix slot indices that contribute MOST to the loss.

    I-GCG replaces the top-``p`` *worst* tokens each step. "Worst" = the slots
    whose removal/perturbation most reduces the loss; here it is estimated cheaply
    and seam-only by masking each slot in turn (replacing it with a neutral filler
    id) and measuring the loss *drop* — the slots with the largest drop are the
    ones currently hurting the objective most and are the best update targets.

    This is a pure-Python, torch-free estimate over the same ``target_loss`` seam
    the GCG loop already uses, so it is unit-testable on CPU.

    Args:
        model: The :class:`WhiteBoxModel` seam.
        prompt_ids: The fixed prompt prefix.
        suffix_ids: The current suffix ids (the optimisable slots).
        target_ids: The benign target ids.
        p: How many worst slots to return (clamped to the suffix length).

    Returns:
        Up to ``p`` slot indices, most-harmful first.
    """
    n = len(list(suffix_ids))
    if n == 0:
        return []
    p = max(1, min(int(p), n))
    base = float(model.target_loss(list(prompt_ids) + list(suffix_ids), list(target_ids)))
    # Neutral filler id: a slot's own value swapped to a stable placeholder so the
    # measured delta isolates that position's current contribution.
    filler = list(suffix_ids)[0]
    drops: list[tuple[float, int]] = []
    for slot in range(n):
        if suffix_ids[slot] == filler:
            # Masking with the same id gives no signal; probe with a different id.
            probe = (filler + 1) if filler + 1 != suffix_ids[slot] else (filler + 2)
        else:
            probe = filler
        trial = list(suffix_ids)
        trial[slot] = probe
        loss = float(model.target_loss(list(prompt_ids) + trial, list(target_ids)))
        drops.append((base - loss, slot))  # positive drop ⇒ slot hurts the loss
    drops.sort(key=lambda d: d[0], reverse=True)
    return [slot for _, slot in drops[:p]]


# --------------------------------------------------------------------------- #
# (3) Easy-to-hard initialization
# --------------------------------------------------------------------------- #


def easy_to_hard_seed(cfg: IGCGConfig) -> Optional[str]:
    """Resolve the easy-to-hard curriculum seed suffix, or ``None``.

    I-GCG's third improvement seeds a hard behavior's suffix from a *solved* easier
    one. Precedence: an explicit ``cfg.init_suffix`` (a real solved seed handed in
    by a curriculum driver) always wins; otherwise, when ``easy_to_hard_init`` is
    on, the bundled benign :data:`EASY_SEED_SUFFIX` primes the optimiser; when the
    flag is off, ``None`` (the attacker falls back to its plain filler).

    Returns:
        The seed suffix string, or ``None`` to use the attacker's default filler.
    """
    if cfg.init_suffix is not None:
        return cfg.init_suffix
    if cfg.easy_to_hard_init:
        return EASY_SEED_SUFFIX
    return None


# --------------------------------------------------------------------------- #
# The registered I-GCG attack
# --------------------------------------------------------------------------- #


@register("igcg")
class IGCGAttack(Attack):
    """I-GCG as a v0.4 :class:`~injectkit.whitebox.base.Attack` (arXiv:2405.21018).

    Composes the three I-GCG improvements (diverse benign targets, automatic
    multi-coordinate update, easy-to-hard init) over the proven
    :class:`injectkit.attackers.gcg.GCGSuffixAttacker` coordinate-descent loop —
    the inner ``token_gradients``/``target_loss`` machinery is reused verbatim
    through the :class:`WhiteBoxModel` seam, never reimplemented.

    Dense-only like the rest of the gradient family (``supported_arch = {"dense"}``).
    The objective is ALWAYS the benign per-run marker.
    """

    name = "igcg"
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
        """Optimise a benign-marker suffix with the three I-GCG improvements.

        See the module docstring for the algorithm. ``cfg`` may be any
        :class:`AttackConfig`; a non-:class:`IGCGConfig` is coerced to I-GCG
        defaults plus the shared GCG knobs. Returns a v0.4
        :class:`~injectkit.whitebox.base.AttackResult`.
        """
        icfg = cfg if isinstance(cfg, IGCGConfig) else _as_igcg_config(cfg)
        prompt = _last_user_content(messages)
        model_name = getattr(model, "name", "") or ""

        # (1) Diverse BENIGN target templates. When the caller pins an explicit
        # target it leads the set (an explicit baseline override); otherwise the
        # whole set is the model-specific benign affirmative openers.
        targets = diverse_targets(
            model_name,
            trigger=icfg.trigger,
            n=icfg.num_diverse_targets,
        )
        if target:
            targets = [target, *[t for t in targets if t != target]]
        elif not icfg.use_advprefix:
            # Baseline mode: fixed "Sure, here is" prefix as the single target.
            targets = [advprefix_target(model_name, trigger=icfg.trigger, use_baseline=True)]
        primary_target = targets[0]

        # (3) Easy-to-hard initialization: seed the suffix from a solved-easy seed.
        seed_suffix = easy_to_hard_seed(icfg)

        # Build the proven legacy attacker; reuse its inner loop verbatim.
        from ..attackers.gcg import GCGSuffixAttacker

        legacy = icfg.to_legacy()
        attacker = GCGSuffixAttacker(
            model, legacy, init_suffix=seed_suffix, name=self.name
        )

        prompt_ids = list(model.token_ids(prompt))
        target_id_sets = [list(model.token_ids(t)) for t in targets]

        steps, best_suffix, best_loss, succeeded = self._igcg_loop(
            model, attacker, icfg, prompt_ids, target_id_sets
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
            stamp={"primary_target": primary_target, "paper_asr": PAPER_ASR},
        )

    def _igcg_loop(
        self,
        model: Any,
        attacker: Any,
        icfg: IGCGConfig,
        prompt_ids: list[int],
        target_id_sets: list[list[int]],
    ) -> tuple[list[tuple[int, float]], str, float, bool]:
        """Run the I-GCG coordinate loop, reusing the GCG seam primitives.

        Per step: (a) pick the easiest currently-unsatisfied benign target
        (improvement 1); (b) compute the gradient and, instead of one slot, update
        the top-``p`` worst slots (improvement 2) where ``p`` is auto-adapted from
        progress; (c) record the loss; stop on a benign-marker success or at the
        step budget. Each slot update reuses the attacker's proven per-slot greedy
        candidate scoring (``_top_k_candidates`` + ``_sample_candidates`` + the
        seam ``target_loss``), so no inner machinery is re-implemented.
        """
        suffix_ids = list(model.token_ids(attacker.init_suffix)) or [0]
        steps: list[tuple[int, float]] = []
        best_suffix = model.decode(suffix_ids)
        best_loss = float("inf")
        succeeded = False
        p = max(1, icfg.init_p)
        prev_loss = float("inf")

        for step_no in range(1, icfg.max_steps + 1):
            input_ids = list(prompt_ids) + list(suffix_ids)
            # (1) Easiest currently-closest benign target.
            tgt_idx, _ = easiest_target(model, input_ids, target_id_sets)
            target_ids = target_id_sets[tgt_idx if tgt_idx >= 0 else 0]
            target_text = model.decode(target_ids)

            suffix_slice = slice(len(prompt_ids), len(input_ids))
            grads = model.token_gradients(input_ids, target_ids, suffix_slice)

            # (2) Choose the top-`p` worst slots and greedily update each, reusing
            # the attacker's proven per-slot candidate scoring.
            slots = worst_coordinates(
                model, prompt_ids, suffix_ids, target_ids, p=p
            )
            step_best_ids = list(suffix_ids)
            step_best_loss = float(
                model.target_loss(list(prompt_ids) + step_best_ids, target_ids)
            )
            for slot in slots:
                candidates = attacker._top_k_candidates(grads, slot)
                if not candidates:
                    continue
                for token_id in attacker._sample_candidates(candidates):
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
            steps.append((step_no, step_best_loss))
            if step_best_loss < best_loss:
                best_loss = step_best_loss
                best_suffix = suffix_text
            if bool(target_text) and target_text in suffix_text:
                succeeded = True
                break

            # Auto-adapt p for the next step from this step's progress.
            if icfg.auto_p_adaptation:
                p = adapt_p(p, prev_loss, step_best_loss, max_p=icfg.max_p)
            prev_loss = step_best_loss

        return steps, best_suffix, best_loss, succeeded


def _as_igcg_config(cfg: AttackConfig) -> IGCGConfig:
    """Coerce any :class:`AttackConfig` to an :class:`IGCGConfig` (I-GCG defaults).

    Carries over the shared/GCG knobs that exist on ``cfg`` and fills the I-GCG
    fields with their defaults, so a caller may hand a plain ``GCGConfig`` (or
    base config) and still get I-GCG.
    """
    base = cfg.model_dump() if isinstance(cfg, GCGConfig) else {
        "max_steps": cfg.max_steps,
        "target": cfg.target,
        "trigger": cfg.trigger,
        "seed": cfg.seed,
    }
    # Drop fields IGCGConfig does not redefine-incompatibly; pydantic ignores
    # unknown via construction from a dict of known keys only.
    allowed = set(IGCGConfig.model_fields)
    return IGCGConfig(**{k: v for k, v in base.items() if k in allowed})


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
    """First-class I-GCG entrypoint (arXiv:2405.21018) — improved GCG in <10 lines.

        from injectkit.whitebox import igcg
        result = igcg.run(model, tok, messages, target, IGCGConfig(max_steps=1))

    A thin functional wrapper over :meth:`IGCGAttack.run` with a defaulted config.
    """
    return IGCGAttack().run(
        model, tokenizer, messages, target, cfg or IGCGConfig(), defense
    )
