"""UJA — Untargeted Jailbreak Attack (arXiv:2510.02999).

CHUNK 11-reinforce-uja (ROADMAP §6.1 objective-frontier). UJA ("Untargeted
Jailbreak Attack", **arXiv:2510.02999**) removes the fixed affirmative target
**entirely** and optimises the model's *own response* directly:

* GCG / REINFORCE-GCG still carry a benign affirmative target (``"Sure, here is …"``)
  that the proposal gradient points at. UJA drops it — there is **no target NLL
  term at all**.
* The objective is simply ``loss = -judge_score(response)``: per candidate suffix,
  sample completions from the target, grade them with a small IN-LOOP judge (the
  v0.4 judge layer; see
  :func:`injectkit.whitebox.objective_judge.judge_scores`), and *maximise the
  judge's unsafety score directly* (minimise its negation). Whatever completion the
  judge scores as most unsafe is the optimisation signal — no phrasing is forced.

The untargeted objective over the proven GCG coordinate-descent seam
(``token_gradients`` / ``target_loss`` on the :class:`WhiteBoxModel`, reused
verbatim): GCG's discrete search still needs a *proposal* gradient to enumerate
candidate replacement tokens, so a benign marker template is used **only to
propose** candidates (never scored against) — the candidate that maximises the
judge score of its sampled completions is accepted. This keeps the search tractable
while the *objective* remains purely the judge's score, faithful to the paper.

CIRCULARITY FIREWALL (ROADMAP §6.10.1 — Decision of Record, test-enforced): the
in-loop OPT judge (:attr:`UJAConfig.judge_id`) MUST differ from the leaderboard
EVAL judge (:attr:`UJAConfig.eval_judge_id`). Since UJA optimises *directly* against
its in-loop judge, optimising against the evaluator would overfit it completely and
make the reported ASR meaningless — so the separation is doubly important here. The
config asserts it at construction and
:func:`injectkit.whitebox.objective_judge.resolve_opt_judge` re-asserts it before
the in-loop judge is loaded.

ETHICS — NON-NEGOTIABLE: completions are graded for the BENIGN per-run marker
``<trigger><canary>`` (the robustness proxy). "Unsafety score" is the judge's score
for emitting that benign marker, never harmful content. ``torch`` is never imported
here — the objective operates on the seam's scalar outputs and the judge's scalar
``sr_score``, so it is unit-testable on CPU with no torch and no model download.

DEFERRED-NO-GPU: the headline ASR parity (arXiv:2510.02999) needs a 7-8B GPU run;
only the untargeted judge-maximising LOGIC/WIRING is verified on the tiny CPU stub
here. The full code path is production-complete.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from .base import Attack, AttackResult
from .config import AttackConfig, UJAConfig
from .objective_judge import judge_scores, resolve_opt_judge, sample_completions
from .registry import register
from .targets import advprefix_target

__all__ = [
    "UJAAttack",
    "run",
    "uja_loss",
    "PAPER_ASR",
]

#: Paper parity (arXiv:2510.02999) — recorded for the stamp. The untargeted
#: objective reaches state-of-the-art ASR; the exact number is DEFERRED-NO-GPU
#: (needs a 7-8B GPU run to reproduce).
PAPER_ASR = "state-of-the-art untargeted ASR (arXiv:2510.02999)"


def uja_loss(rewards: Sequence[float]) -> float:
    """The UJA objective for one candidate suffix: ``-mean(judge_score)`` (lower⇒better).

    UJA maximises the in-loop judge's unsafety score of the model's sampled
    response *directly* (arXiv:2510.02999): there is no target NLL term. We average
    the per-completion judge scores and negate, so the greedy GCG search — which
    *minimises* its scalar score — moves the suffix toward suffixes whose sampled
    completions the judge rates most unsafe.

    Args:
        rewards: Per-completion judge unsafety scores in ``[0, 1]``.

    Returns:
        ``-mean(rewards)`` (in ``[-1, 0]``); ``0.0`` for an empty completion set.
    """
    r = [float(x) for x in rewards]
    if not r:
        return 0.0
    return -(sum(r) / len(r))


@register("uja")
class UJAAttack(Attack):
    """UJA as a v0.4 :class:`~injectkit.whitebox.base.Attack` (arXiv:2510.02999).

    Composes the *untargeted* objective (sample completions → grade with the
    IN-LOOP OPT judge → maximise the score directly, no target NLL) over the proven
    GCG coordinate-descent seam (``token_gradients`` / ``target_loss`` on the
    :class:`WhiteBoxModel`, reused verbatim — a benign marker is used ONLY to
    propose candidate tokens, never scored against).

    Dense-only like the rest of the gradient family (``supported_arch = {"dense"}``).
    The success proxy is ALWAYS the benign per-run marker, and the IN-LOOP judge
    (:attr:`UJAConfig.judge_id`) is asserted distinct from the EVAL judge
    (:attr:`UJAConfig.eval_judge_id`) — the §6.10.1 circularity firewall, which is
    especially load-bearing for UJA since it optimises *directly* against that
    judge. Accepts ``cfg.defense`` for adaptive runs (ROADMAP §6.13).
    """

    name = "uja"
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
        """Optimise a benign-marker suffix with the UJA untargeted judge objective.

        See the module docstring for the algorithm. ``cfg`` may be any
        :class:`AttackConfig`; a non-:class:`UJAConfig` is coerced to UJA defaults.
        Returns a v0.4 :class:`~injectkit.whitebox.base.AttackResult` whose loss
        curve is the negated mean judge score (``-judge_score``), not an NLL.
        """
        ucfg = cfg if isinstance(cfg, UJAConfig) else _as_uja_config(cfg)
        prompt = _last_user_content(messages)
        model_name = getattr(model, "name", "") or ""

        # Resolve the IN-LOOP OPT judge AFTER asserting it differs from the eval
        # judge (§6.10.1 — raises if they collide).
        judge = resolve_opt_judge(ucfg.judge_id, ucfg.eval_judge_id)

        # PROPOSAL-ONLY benign marker template: used solely to enumerate candidate
        # tokens via the gradient. It is NEVER scored against — UJA's objective is
        # purely the judge score. An explicit `target` override leads if given.
        proposal_target = target or advprefix_target(
            model_name, trigger=ucfg.trigger, use_baseline=True
        )
        canary = _canary_from_target(proposal_target, ucfg.trigger)

        attacker = _build_proposal_attacker(model, ucfg, name=self.name)

        prompt_ids = list(model.token_ids(prompt))
        proposal_ids = list(model.token_ids(proposal_target))
        rng = random.Random(ucfg.seed)

        steps, best_suffix, best_loss, succeeded = self._uja_loop(
            model, attacker, ucfg, judge, prompt_ids, proposal_ids,
            canary=canary, proposal_target=proposal_target, trigger=ucfg.trigger, rng=rng,
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
                "opt_judge_id": ucfg.judge_id,
                "eval_judge_id": ucfg.eval_judge_id,
                "num_samples": ucfg.num_samples,
                "objective": "untargeted-judge",
                "paper_asr": PAPER_ASR,
            },
        )

    def _uja_loop(
        self,
        model: Any,
        attacker: Any,
        ucfg: UJAConfig,
        judge: Any,
        prompt_ids: list[int],
        proposal_ids: list[int],
        *,
        canary: str,
        proposal_target: str,
        trigger: str,
        rng: random.Random,
    ) -> tuple[list[tuple[int, float]], str, float, bool]:
        """Run the UJA coordinate loop, reusing the GCG seam primitives.

        Per step: (a) compute the PROPOSAL gradient toward the benign marker and
        take the attacker's proven per-slot top-k candidates (candidates only — the
        marker is never scored against); (b) for the current suffix and each
        candidate swap, sample ``num_samples`` completions, grade them with the
        IN-LOOP judge, and score the candidate by the UJA objective
        (:func:`uja_loss` = ``-mean(judge_score)``); (c) greedily accept the lowest
        (most-unsafe) candidate; (d) record the loss; stop on a benign-marker
        success or the step budget. The completion sampling + judging is the only
        new machinery — the gradient/candidate plumbing is the verbatim GCG seam.
        """
        suffix_ids = list(model.token_ids(attacker.init_suffix)) if attacker.init_suffix else []
        if not suffix_ids:
            suffix_ids = [0] * max(1, ucfg.suffix_len)
        steps: list[tuple[int, float]] = []
        best_suffix = model.decode(suffix_ids)
        best_loss = float("inf")
        succeeded = False

        for step_no in range(1, ucfg.max_steps + 1):
            input_ids = list(prompt_ids) + list(suffix_ids)
            suffix_slice = slice(len(prompt_ids), len(input_ids))
            # Proposal gradient toward the benign marker (candidate enumeration ONLY).
            grads = model.token_gradients(input_ids, proposal_ids, suffix_slice)

            step_best_ids = list(suffix_ids)
            step_best_loss = self._candidate_uja_loss(
                model, prompt_ids, step_best_ids, judge,
                num_samples=ucfg.num_samples, sample_temp=ucfg.sample_temp,
                canary=canary, proposal_target=proposal_target, trigger=trigger, rng=rng,
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
                    loss = self._candidate_uja_loss(
                        model, prompt_ids, trial, judge,
                        num_samples=ucfg.num_samples, sample_temp=ucfg.sample_temp,
                        canary=canary, proposal_target=proposal_target, trigger=trigger,
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
                model, full_prompt, num_samples=ucfg.num_samples,
                sample_temp=ucfg.sample_temp, rng=rng,
            )
            marker = proposal_target or (f"{trigger}{canary}" if canary else "")
            if marker and any(marker in c for c in completions):
                succeeded = True
                break

        return steps, best_suffix, best_loss, succeeded

    def _candidate_uja_loss(
        self,
        model: Any,
        prompt_ids: Sequence[int],
        suffix_ids: Sequence[int],
        judge: Any,
        *,
        num_samples: int,
        sample_temp: float,
        canary: str,
        proposal_target: str,
        trigger: str,
        rng: random.Random,
    ) -> float:
        """UJA objective of one candidate suffix: sample → judge → ``-mean(score)``.

        No target NLL term — the objective is purely the in-loop judge's score of
        the model's own sampled completions (arXiv:2510.02999).
        """
        full_prompt = f"{model.decode(list(prompt_ids))} {model.decode(list(suffix_ids))}".rstrip()
        completions = sample_completions(
            model, full_prompt, num_samples=num_samples, sample_temp=sample_temp, rng=rng,
        )
        rewards = judge_scores(
            completions, judge, canary=canary, target=proposal_target, trigger=trigger,
        )
        return uja_loss(rewards)


def _build_proposal_attacker(model: Any, ucfg: UJAConfig, *, name: str) -> Any:
    """Build the proven GCG attacker UJA reuses for candidate proposal.

    UJA is not a GCGConfig subclass (it has no fixed target), so we project its
    suffix/top-k/seed knobs onto the v0.3 legacy :class:`GCGConfig` dataclass the
    :class:`GCGSuffixAttacker` consumes. The attacker is used ONLY for its proven
    gradient→top-k candidate plumbing; UJA scores candidates by the judge, not by
    this attacker's NLL.
    """
    from ..attackers.gcg import GCGSuffixAttacker
    from ..attackers.whitebox_base import GCGConfig as LegacyGCGConfig

    legacy = LegacyGCGConfig(
        max_steps=ucfg.max_steps,
        suffix_len=ucfg.suffix_len,
        top_k=ucfg.top_k,
        target_string=None,
        init_suffix=ucfg.init_suffix,
        trigger=ucfg.trigger,
        seed=ucfg.seed,
    )
    return GCGSuffixAttacker(model, legacy, init_suffix=ucfg.init_suffix, name=name)


def _as_uja_config(cfg: AttackConfig) -> UJAConfig:
    """Coerce any :class:`AttackConfig` to a :class:`UJAConfig` (UJA defaults).

    Carries over the cross-attack / GCG knobs present on ``cfg`` and fills the UJA
    fields with their defaults (whose ``judge_id``/``eval_judge_id`` already satisfy
    the §6.10.1 firewall), so a caller may hand a plain config and still get UJA.
    """
    base = cfg.model_dump()
    allowed = set(UJAConfig.model_fields)
    return UJAConfig(**{k: v for k, v in base.items() if k in allowed})


def _canary_from_target(target: str, trigger: str) -> str:
    """Best-effort extract the benign canary tail from a ``<trigger><canary>`` target."""
    if trigger and trigger in target:
        tail = target.split(trigger, 1)[1]
        return tail.split()[0] if tail else ""
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
    """First-class UJA entrypoint (arXiv:2510.02999) — untargeted judge-maximising GCG.

        from injectkit.whitebox import uja
        result = uja.run(model, tok, messages, "", UJAConfig(max_steps=1))

    A thin functional wrapper over :meth:`UJAAttack.run` with a defaulted config
    (whose default ``judge_id``/``eval_judge_id`` already satisfy the §6.10.1
    firewall).
    """
    return UJAAttack().run(
        model, tokenizer, messages, target, cfg or UJAConfig(), defense
    )
