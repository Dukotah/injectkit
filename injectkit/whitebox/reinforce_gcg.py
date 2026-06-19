"""REINFORCE-GCG — policy-gradient GCG with a judge in the loop (arXiv:2502.17254).

CHUNK 11-reinforce-uja (ROADMAP §6.1 objective-frontier). REINFORCE-GCG ("Improved
Generation of Adversarial Examples Against Safety-aligned LLMs via an Adapted
REINFORCE Objective", **arXiv:2502.17254**, ICML 2025) replaces GCG's *single
fixed-target NLL* objective with a **policy-gradient (REINFORCE) objective over a
distribution of completions**:

* GCG minimises ``NLL(affirmative_prefix | prompt+suffix)`` — the likelihood of one
  hand-written target string. That over-fits a phrasing and ignores what the model
  actually generates.
* REINFORCE-GCG instead, for each candidate suffix, **samples several completions**
  from the target, **grades each with a small IN-LOOP judge** (the v0.4 judge
  layer; see :func:`injectkit.whitebox.objective_judge.judge_scores`), and applies a
  REINFORCE gradient that pushes the suffix toward completions the judge scores as
  unsafe (reward ``r = sr_score``). The expected reward of the *actual generated
  distribution* is optimised, so the attack transfers and generalises far better.

The policy-gradient surrogate, realised over the proven GCG coordinate-descent seam
(``token_gradients`` / ``target_loss`` on the :class:`WhiteBoxModel`, never
rebuilt): per step we still use the gradient toward the benign target to *propose*
candidate replacement tokens (REINFORCE needs a search basis), but we *score* each
candidate by the REINFORCE objective — the (baseline-subtracted) judge reward of
its sampled completions, weighted by the target NLL so a candidate that makes
high-reward completions *more likely* wins. See :func:`reinforce_loss`.

CIRCULARITY FIREWALL (ROADMAP §6.10.1 — Decision of Record, test-enforced): the
in-loop OPT judge (:attr:`REINFORCEGCGConfig.judge_id`) MUST differ from the
leaderboard EVAL judge (:attr:`REINFORCEGCGConfig.eval_judge_id`). Optimising
against the evaluator overfits it and invalidates the reported ASR. The config
asserts this at construction and :func:`injectkit.whitebox.objective_judge.resolve_opt_judge`
re-asserts it before loading the in-loop judge.

ETHICS — NON-NEGOTIABLE: completions are graded for the BENIGN per-run marker
``<trigger><canary>`` (the robustness proxy). The "unsafety reward" is the judge's
score for emitting that benign marker, never harmful content. ``torch`` is never
imported here — the policy-gradient logic operates on the seam's scalar
``target_loss`` outputs and the judge's scalar ``sr_score``, so it is unit-testable
on CPU with no torch and no model download.

DEFERRED-NO-GPU: the headline 24GB-VRAM fit and the >2x speedup / 85-86% ASR on
Llama-3-8B (arXiv:2502.17254) need a 7-8B GPU run — only the LOGIC/WIRING of the
REINFORCE objective + judge-in-the-loop is verified on the tiny CPU stub here. The
full code path is production-complete.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from .base import Attack, AttackResult
from .config import AttackConfig, GCGConfig, REINFORCEGCGConfig
from .objective_judge import judge_scores, resolve_opt_judge, sample_completions
from .registry import register
from .targets import advprefix_target

__all__ = [
    "REINFORCEGCGAttack",
    "run",
    "reinforce_loss",
    "PAPER_ASR",
    "PAPER_VRAM",
]

#: Paper parity (arXiv:2502.17254, ICML 2025) — recorded for the stamp.
#: REINFORCE-GCG reaches ~85-86% ASR on Llama-3-8B with a >2x speedup, fitting a
#: single 24GB GPU. DEFERRED-NO-GPU (needs a 7-8B GPU run to reproduce).
PAPER_ASR = "~85-86% (Llama-3-8B; arXiv:2502.17254)"
#: The paper's single-GPU memory envelope, recorded for the stamp. DEFERRED-NO-GPU.
PAPER_VRAM = "fits one 24GB GPU"


def reinforce_loss(
    rewards: Sequence[float],
    nlls: Sequence[float],
    *,
    baseline: Optional[float] = None,
) -> float:
    """The REINFORCE surrogate loss for one candidate suffix (lower ⇒ better).

    Faithful to the adapted-REINFORCE objective of arXiv:2502.17254: with per-
    completion rewards ``r_i`` (the in-loop judge's unsafety ``sr_score``) and
    per-completion negative-log-likelihoods ``nll_i = -log p(completion_i)`` under
    the current suffix, the policy-gradient objective maximises the
    advantage-weighted log-likelihood ``Σ (r_i - b) · log p_i = -Σ (r_i - b)·nll_i``
    where ``b`` is a variance-reducing baseline (the mean reward). We return its
    **negation** so the greedy GCG search, which *minimises* its scalar score, moves
    the suffix toward completions that are both high-reward AND made more likely:

        loss = (1/N) · Σ (r_i - b) · nll_i

    A candidate that lowers the NLL of an above-baseline-reward completion (makes a
    high-reward generation more probable) drives this down; one that makes a
    below-baseline completion more likely drives it up. With no reward signal at all
    (all ``r_i`` equal ⇒ advantages 0) it degenerates to ``0`` — the proposal
    gradient then breaks ties — which keeps the seam well-defined on the toy stub.

    Args:
        rewards: Per-completion judge unsafety scores in ``[0, 1]``.
        nlls: Per-completion target NLLs (the seam ``target_loss`` of each
            completion given prompt+suffix). Same length as ``rewards``.
        baseline: The REINFORCE baseline ``b`` subtracted from each reward;
            defaults to the mean reward (the standard low-variance choice).

    Returns:
        The scalar REINFORCE surrogate loss (lower is better). ``0.0`` for an
        empty completion set.
    """
    n = min(len(rewards), len(nlls))
    if n == 0:
        return 0.0
    r = [float(x) for x in list(rewards)[:n]]
    ll = [float(x) for x in list(nlls)[:n]]
    b = (sum(r) / n) if baseline is None else float(baseline)
    return sum((r[i] - b) * ll[i] for i in range(n)) / n


@register("reinforce_gcg")
class REINFORCEGCGAttack(Attack):
    """REINFORCE-GCG as a v0.4 :class:`~injectkit.whitebox.base.Attack` (arXiv:2502.17254).

    Composes the adapted-REINFORCE objective (sample completions → grade with the
    IN-LOOP OPT judge → policy-gradient surrogate) over the proven GCG coordinate-
    descent seam (``token_gradients`` / ``target_loss`` on the
    :class:`WhiteBoxModel`, reused verbatim — the inner search machinery is never
    re-implemented).

    Dense-only like the rest of the gradient family (``supported_arch = {"dense"}``).
    The objective is ALWAYS the benign per-run marker, and the IN-LOOP judge
    (:attr:`REINFORCEGCGConfig.judge_id`) is asserted distinct from the EVAL judge
    (:attr:`REINFORCEGCGConfig.eval_judge_id`) — the §6.10.1 circularity firewall.
    Accepts ``cfg.defense`` for adaptive runs (ROADMAP §6.13).
    """

    name = "reinforce_gcg"
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
        """Optimise a benign-marker suffix with the REINFORCE judge-in-the-loop objective.

        See the module docstring for the algorithm. ``cfg`` may be any
        :class:`AttackConfig`; a non-:class:`REINFORCEGCGConfig` is coerced to
        REINFORCE-GCG defaults. Returns a v0.4
        :class:`~injectkit.whitebox.base.AttackResult` whose ``best_loss`` curve is
        the REINFORCE surrogate (not the raw NLL).
        """
        rcfg = cfg if isinstance(cfg, REINFORCEGCGConfig) else _as_reinforce_config(cfg)
        prompt = _last_user_content(messages)
        model_name = getattr(model, "name", "") or ""

        # Resolve the IN-LOOP OPT judge AFTER asserting it differs from the eval
        # judge (§6.10.1 circularity firewall — raises if they collide).
        judge = resolve_opt_judge(rcfg.judge_id, rcfg.eval_judge_id)

        # The benign target the proposal gradient points at (success is still the
        # marker). An explicit `target` override leads; else the model's AdvPrefix.
        primary_target = target or advprefix_target(
            model_name, trigger=rcfg.trigger, use_baseline=not rcfg.use_advprefix
        )
        canary = _canary_from_target(primary_target, rcfg.trigger)

        from ..attackers.gcg import GCGSuffixAttacker

        legacy = rcfg.to_legacy()
        attacker = GCGSuffixAttacker(
            model, legacy, init_suffix=rcfg.init_suffix, name=self.name
        )

        prompt_ids = list(model.token_ids(prompt))
        target_ids = list(model.token_ids(primary_target))
        rng = random.Random(rcfg.seed)

        steps, best_suffix, best_loss, succeeded = self._reinforce_loop(
            model, attacker, rcfg, judge, prompt_ids, target_ids,
            canary=canary, primary_target=primary_target, trigger=rcfg.trigger, rng=rng,
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
            stamp={
                "primary_target": primary_target,
                "opt_judge_id": rcfg.judge_id,
                "eval_judge_id": rcfg.eval_judge_id,
                "num_samples": rcfg.num_samples,
                "objective": "reinforce",
                "paper_asr": PAPER_ASR,
                "paper_vram": PAPER_VRAM,
            },
        )

    def _reinforce_loop(
        self,
        model: Any,
        attacker: Any,
        rcfg: REINFORCEGCGConfig,
        judge: Any,
        prompt_ids: list[int],
        target_ids: list[int],
        *,
        canary: str,
        primary_target: str,
        trigger: str,
        rng: random.Random,
    ) -> tuple[list[tuple[int, float]], str, float, bool]:
        """Run the REINFORCE coordinate loop, reusing the GCG seam primitives.

        Per step: (a) compute the proposal gradient toward the benign target and
        take the attacker's proven per-slot top-k candidates; (b) for the current
        suffix and each candidate swap, sample ``num_samples`` completions, grade
        them with the IN-LOOP judge, and score the candidate by the REINFORCE
        surrogate (:func:`reinforce_loss`); (c) greedily accept the lowest-surrogate
        candidate; (d) record the loss; stop on a benign-marker success or the step
        budget. The completion sampling + judging is the only new machinery — the
        gradient/candidate plumbing is the verbatim GCG seam.
        """
        suffix_ids = list(model.token_ids(attacker.init_suffix)) if attacker.init_suffix else []
        if not suffix_ids:
            suffix_ids = [0] * max(1, rcfg.suffix_len)
        steps: list[tuple[int, float]] = []
        best_suffix = model.decode(suffix_ids)
        best_loss = float("inf")
        succeeded = False

        for step_no in range(1, rcfg.max_steps + 1):
            input_ids = list(prompt_ids) + list(suffix_ids)
            suffix_slice = slice(len(prompt_ids), len(input_ids))
            grads = model.token_gradients(input_ids, target_ids, suffix_slice)

            # Evaluate the REINFORCE surrogate of the current suffix as the bar.
            step_best_ids = list(suffix_ids)
            step_best_loss = self._candidate_reinforce_loss(
                model, prompt_ids, step_best_ids, judge,
                num_samples=rcfg.num_samples, sample_temp=rcfg.sample_temp,
                canary=canary, primary_target=primary_target, trigger=trigger, rng=rng,
            )

            for slot in range(len(suffix_ids)):
                candidates = attacker._top_k_candidates(grads, slot)
                if not candidates:
                    continue
                for token_id in attacker._sample_candidates(candidates):
                    if token_id == step_best_ids[slot]:
                        continue
                    trial = list(step_best_ids)
                    trial[slot] = token_id
                    loss = self._candidate_reinforce_loss(
                        model, prompt_ids, trial, judge,
                        num_samples=rcfg.num_samples, sample_temp=rcfg.sample_temp,
                        canary=canary, primary_target=primary_target, trigger=trigger,
                        rng=rng,
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

            # Success: the model's own sampled completion carries the benign marker.
            full_prompt = f"{model.decode(prompt_ids)} {suffix_text}".rstrip()
            completions = sample_completions(
                model, full_prompt, num_samples=rcfg.num_samples,
                sample_temp=rcfg.sample_temp, rng=rng,
            )
            marker = primary_target or (f"{trigger}{canary}" if canary else "")
            if marker and any(marker in c for c in completions):
                succeeded = True
                break

        return steps, best_suffix, best_loss, succeeded

    def _candidate_reinforce_loss(
        self,
        model: Any,
        prompt_ids: Sequence[int],
        suffix_ids: Sequence[int],
        judge: Any,
        *,
        num_samples: int,
        sample_temp: float,
        canary: str,
        primary_target: str,
        trigger: str,
        rng: random.Random,
    ) -> float:
        """REINFORCE surrogate loss of one candidate suffix (sample → judge → grad).

        Samples ``num_samples`` completions for ``prompt+suffix``, grades each with
        the IN-LOOP judge (reward = ``sr_score``), measures each completion's target
        NLL via the seam ``target_loss``, and returns :func:`reinforce_loss`.
        """
        full_prompt = f"{model.decode(list(prompt_ids))} {model.decode(list(suffix_ids))}".rstrip()
        completions = sample_completions(
            model, full_prompt, num_samples=num_samples, sample_temp=sample_temp, rng=rng,
        )
        rewards = judge_scores(
            completions, judge, canary=canary, target=primary_target, trigger=trigger,
        )
        cond_ids = list(prompt_ids) + list(suffix_ids)
        nlls = [
            float(model.target_loss(cond_ids, list(model.token_ids(c))))
            for c in completions
        ]
        return reinforce_loss(rewards, nlls)


def _as_reinforce_config(cfg: AttackConfig) -> REINFORCEGCGConfig:
    """Coerce any :class:`AttackConfig` to a :class:`REINFORCEGCGConfig` (defaults).

    Carries over the shared/GCG knobs present on ``cfg`` and fills the REINFORCE
    fields with their defaults, so a caller may hand a plain ``GCGConfig`` (or base
    config) and still get REINFORCE-GCG with a valid (distinct) judge pair.
    """
    base = cfg.model_dump() if isinstance(cfg, GCGConfig) else {
        "max_steps": cfg.max_steps,
        "target": cfg.target,
        "trigger": cfg.trigger,
        "seed": cfg.seed,
    }
    allowed = set(REINFORCEGCGConfig.model_fields)
    return REINFORCEGCGConfig(**{k: v for k, v in base.items() if k in allowed})


def _canary_from_target(target: str, trigger: str) -> str:
    """Best-effort extract the benign canary tail from a ``<trigger><canary>`` target."""
    if trigger and trigger in target:
        return target.split(trigger, 1)[1].split()[0] if target.split(trigger, 1)[1] else ""
    return ""


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
    """First-class REINFORCE-GCG entrypoint (arXiv:2502.17254) — policy-gradient GCG.

        from injectkit.whitebox import reinforce_gcg
        result = reinforce_gcg.run(model, tok, messages, target, REINFORCEGCGConfig(max_steps=1))

    A thin functional wrapper over :meth:`REINFORCEGCGAttack.run` with a defaulted
    config (whose default ``judge_id``/``eval_judge_id`` already satisfy the §6.10.1
    firewall).
    """
    return REINFORCEGCGAttack().run(
        model, tokenizer, messages, target, cfg or REINFORCEGCGConfig(), defense
    )
