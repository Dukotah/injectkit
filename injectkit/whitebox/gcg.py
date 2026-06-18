"""GCG re-wrapped behind the v0.4 :class:`~injectkit.whitebox.base.Attack` contract.

ROADMAP §6.1 / chunk 1: the existing, proven greedy-coordinate-gradient optimiser
(:class:`injectkit.attackers.gcg.GCGSuffixAttacker`) is re-exposed as a registered
v0.4 :class:`~injectkit.whitebox.base.Attack` so it resolves through the new
registry and runs through ``Attack.run(model, tokenizer, messages, target, cfg,
defense)``. This is a *re-wrap, not a rebuild*: the inner loop
(``_optimize_suffix``) is reused verbatim through the
:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` seam, so the v0.3
``StubWhiteBoxModel`` drives it offline and the v0.3 GCG tests are untouched.

Public entrypoint (ROADMAP §6.1, "gradient attack in <10 lines"):

    from injectkit.whitebox import gcg
    result = gcg.run(model, tokenizer, messages, target, GCGConfig(max_steps=1))

ETHICS — NON-NEGOTIABLE: the optimisation objective is the per-run BENIGN canary
marker ``<trigger><canary>`` (the ``target`` passed in), never harmful content.
No harmful suffix is bundled or targeted. White-box ⇒ HF-only and compute-heavy;
real runs want a GPU. Tests use the offline stub seam and need neither torch nor
a model download.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

from typing import Any, Optional

from ..attackers.gcg import GCGSuffixAttacker
from .base import Attack, AttackResult
from .config import AttackConfig, GCGConfig
from .gcg_hard import (
    AttackBuffer,
    PromptSlices,
    ProbeSamplingConfig,
    filter_ids,
    locate_optim_slice,
    round_trips,
    sample_candidates,
    token_gradients_onehot,
)
from .probe_sampling import (
    PAPER_ASR,
    PAPER_SPEEDUP,
    ProbeSampling,
    ProbeSamplingResult,
    resolve_probe_sampling,
)
from .registry import register
from .targets import FIXED_BASELINE_PREFIX, advprefix_target

__all__ = [
    "GCGAttack",
    "run",
    # nanoGCG-parity hardening primitives (chunk 3-gcg-advprefix).
    "PromptSlices",
    "locate_optim_slice",
    "filter_ids",
    "round_trips",
    "token_gradients_onehot",
    "sample_candidates",
    "AttackBuffer",
    "ProbeSamplingConfig",
    # AdvPrefix target source.
    "advprefix_target",
    "FIXED_BASELINE_PREFIX",
    # Probe Sampling efficiency primitive (chunk 8; arXiv:2403.01251).
    "ProbeSampling",
    "ProbeSamplingResult",
    "resolve_probe_sampling",
    "PAPER_SPEEDUP",
    "PAPER_ASR",
]


@register("gcg")
class GCGAttack(Attack):
    """White-box GCG as a v0.4 :class:`~injectkit.whitebox.base.Attack` (dense-only).

    Wraps :class:`injectkit.attackers.gcg.GCGSuffixAttacker`'s greedy-coordinate-
    gradient loop. :meth:`run` builds the legacy attacker, drives its
    ``_optimize_suffix`` over the white-box ``model`` seam toward the benign
    ``target`` string, and projects the per-step trajectory onto a v0.4
    :class:`~injectkit.whitebox.base.AttackResult` (best suffix, loss curve,
    success flag).

    ``supported_arch`` is the dense default — GCG is a gradient family, scoped to
    dense transformers for v0.4 (ROADMAP §6.14).
    """

    name = "gcg"
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
        """Optimise a benign-marker suffix via GCG and return the v0.4 result.

        Args:
            model: A white-box model seam
                (:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel`) — an
                in-process HF causal-LM in production, ``StubWhiteBoxModel`` in
                tests. Required: GCG is gradient-driven.
            tokenizer: Unused by the v0.3 seam (it tokenises internally via
                ``model.token_ids``); accepted for interface parity and forwarded
                only if a future seam needs it.
            messages: Chat turns; the optimiser appends the suffix to the last
                user turn's content (the rendered prompt). Earlier turns form the
                static prefix context.
            target: The BENIGN string to emit (the per-run marker). Never harmful.
            cfg: A :class:`~injectkit.whitebox.config.GCGConfig` (or any
                :class:`~injectkit.whitebox.config.AttackConfig`; non-GCG configs
                are coerced to ``GCGConfig`` defaults plus the shared knobs).
            defense: Optional defense for adaptive mode (recorded on the result;
                in-loop adaptive coupling is a later-chunk deliverable, §6.13).

        Returns:
            An :class:`~injectkit.whitebox.base.AttackResult` with the best suffix
            as ``optimized_obj`` (kind ``"suffix"``), the best input, the per-step
            loss curve, and the success flag.

        Raises:
            ArchitectureError: if a caller-supplied ``arch`` is unsupported (the
                bench layer calls :meth:`check_arch` before this).
        """
        gcfg = cfg if isinstance(cfg, GCGConfig) else _as_gcg_config(cfg)
        prompt = _last_user_content(messages)

        # AdvPrefix (arXiv:2412.10321) is the DEFAULT target source for GCG: when
        # the caller does not pin an explicit target, derive a model-specific
        # affirmative prefix (Pareto of prefill-success x low-NLL) for this model,
        # else the documented fixed "Sure, here is" baseline. The marker stays the
        # success condition, so the objective is benign.
        if not target:
            target = advprefix_target(
                getattr(model, "name", "") or "",
                trigger=gcfg.trigger,
                use_baseline=not gcfg.use_advprefix,
            )

        attacker = GCGSuffixAttacker(
            model,
            gcfg.to_legacy(),
            init_suffix=gcfg.init_suffix,
            name=self.name,
        )

        # Probe Sampling (arXiv:2403.01251) opt-in: when cfg.probe_sampling is set,
        # attach a cheap DRAFT model so each step draft-filters its candidate batch
        # before the expensive target scoring. The draft is taken from the model
        # seam's optional ``draft_model`` attribute (a small zoo model in
        # production); if absent it degrades to the target model itself (still
        # exercises the re-scoring logic). The real >=3x speedup is DEFERRED-NO-GPU.
        ps = resolve_probe_sampling(gcfg.probe_sampling)
        if ps.enabled:
            draft = getattr(model, "draft_model", None) or model
            attacker.attach_probe_sampling(
                draft, r=ps.r, sampling_factor=ps.sampling_factor
            )

        # Drive the proven inner loop directly through the white-box seam so we
        # capture the optimisation trajectory (the loss curve + best suffix). This
        # reuses GCGSuffixAttacker._optimize_suffix verbatim — no re-implementation.
        prompt_ids = model.token_ids(prompt)
        target_ids = model.token_ids(target)
        steps = attacker._optimize_suffix(prompt_ids, target_ids)

        best = attacker._best_step(steps)
        best_suffix = best.suffix if best is not None else attacker.init_suffix
        best_loss = best.loss if best is not None else float("inf")
        succeeded = bool(best is not None and best.succeeded)
        best_input = f"{prompt} {best_suffix}".rstrip() if best_suffix else prompt

        defense_id = getattr(defense, "name", "") if defense is not None else ""

        return AttackResult(
            attack_name=self.name,
            best_input=best_input,
            best_loss=best_loss,
            per_step_losses=[s.loss for s in steps],
            optimized_obj=best_suffix,
            optimized_obj_kind="suffix",
            succeeded=succeeded,
            queries=len(steps),
            defense_id=defense_id,
        )


def _as_gcg_config(cfg: AttackConfig) -> GCGConfig:
    """Coerce a base :class:`AttackConfig` to a :class:`GCGConfig`.

    A caller may hand the generic base config; GCG needs its extra knobs, so this
    carries over the shared fields and fills GCG defaults for the rest.
    """
    return GCGConfig(
        max_steps=cfg.max_steps,
        target=cfg.target,
        trigger=cfg.trigger,
        seed=cfg.seed,
    )


def _last_user_content(messages: list[dict]) -> str:
    """Return the content of the last ``user`` turn (or the last turn, or "").

    The suffix is optimised against the final user prompt; a missing/empty message
    list yields an empty prompt rather than raising (the optimiser still runs).
    """
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
    """First-class GCG entrypoint (ROADMAP §6.1) — gradient attack in <10 lines.

        from injectkit.whitebox import gcg
        result = gcg.run(model, tok, messages, target, GCGConfig(max_steps=1))

    A thin functional wrapper over :meth:`GCGAttack.run` with a defaulted config.
    """
    return GCGAttack().run(
        model, tokenizer, messages, target, cfg or GCGConfig(), defense
    )
