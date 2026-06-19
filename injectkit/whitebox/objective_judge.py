"""Judge-in-the-loop objective primitives for REINFORCE-GCG + UJA (CHUNK 11).

ROADMAP §6.1 objective-frontier. The two attacks in this chunk
(:mod:`injectkit.whitebox.reinforce_gcg`, **arXiv:2502.17254**, ICML 2025;
:mod:`injectkit.whitebox.uja`, **arXiv:2510.02999**) replace GCG's *fixed-target
NLL* objective with one that scores the model's own **sampled completions** with a
small IN-LOOP judge. Both need exactly the same two seam operations, factored here
so neither attack re-implements them:

1. :func:`sample_completions` — draw ``num_samples`` completions from the target
   model conditioned on the current ``prompt + suffix`` at temperature
   ``sample_temp``. A real white-box model exposes a ``generate`` / ``sample``
   method (the GPU path); the offline ``StubWhiteBoxModel`` seam used in tests
   does NOT, so this helper falls back to a deterministic, seam-only pseudo-
   completion (``decode(token_ids(...))`` round-trips) so the LOGIC/WIRING is unit-
   testable on CPU with no torch and no model download.
2. :func:`judge_scores` — grade each sampled completion with the **in-loop OPT
   judge** (the v0.4 judge layer, resolved by ``opt_judge_id``) and return its
   continuous unsafety score ``sr_score`` in ``[0, 1]`` per completion.

CIRCULARITY FIREWALL (ROADMAP §6.10.1 — Decision of Record, test-enforced): the
in-loop OPT judge id MUST differ from the leaderboard EVAL judge id. Optimising
*against* the same judge that scores the published number would let the attack
overfit the evaluator and report a meaningless ASR. Both configs in this chunk
assert ``opt_judge_id != eval_judge_id`` at construction; this module never
touches the eval judge — it only ever resolves the OPT judge.

ETHICS — NON-NEGOTIABLE: the completions are graded for the BENIGN per-run marker
``<trigger><canary>`` (the robustness proxy). "Unsafety score" here is the judge's
``sr_score`` for emitting that benign marker, never harmful content. ``torch`` is
never imported at module load; the GPU generation path is reached only when a real
model exposes ``generate``.

DEFERRED-NO-GPU: real completion sampling from a 7-8B target (and the ASR-parity
numbers) need a GPU; only the seam wiring + judge plumbing are exercised on the
tiny CPU stub here. The full code path is production-complete.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from ..models import TargetResponse

__all__ = [
    "sample_completions",
    "judge_scores",
    "resolve_opt_judge",
    "assert_judge_separation",
]


def assert_judge_separation(opt_judge_id: str, eval_judge_id: str) -> None:
    """Enforce the circularity firewall (ROADMAP §6.10.1): opt != eval judge.

    Raises if the in-loop optimisation judge and the leaderboard evaluation judge
    are the same id. Optimising against the evaluator overfits it and makes the
    reported ASR meaningless, so this is a hard, test-enforced Decision of Record
    for every judge-in-the-loop attack (REINFORCE-GCG + UJA).

    Args:
        opt_judge_id: The in-loop judge resolved by the optimiser.
        eval_judge_id: The leaderboard judge that scores the published number.

    Raises:
        ValueError: if the two ids are equal (or either is empty).
    """
    if not opt_judge_id or not eval_judge_id:
        raise ValueError(
            "both opt_judge_id and eval_judge_id must be non-empty judge ids "
            "(circularity firewall, ROADMAP §6.10.1)."
        )
    if opt_judge_id == eval_judge_id:
        raise ValueError(
            f"opt_judge_id ({opt_judge_id!r}) MUST differ from eval_judge_id "
            f"({eval_judge_id!r}): optimising against the evaluation judge overfits "
            "it and invalidates the reported ASR (ROADMAP §6.10.1 circularity "
            "firewall)."
        )


def resolve_opt_judge(opt_judge_id: str, eval_judge_id: str) -> Any:
    """Resolve the IN-LOOP OPT judge, after asserting it differs from the eval judge.

    This is the single seam every judge-in-the-loop attack uses to obtain its
    optimisation judge, so the firewall is checked in exactly one place. It NEVER
    resolves the eval judge — the eval judge is the leaderboard layer's concern.

    Args:
        opt_judge_id: Registry id of the in-loop judge to optimise against.
        eval_judge_id: Registry id of the leaderboard judge (checked, not loaded).

    Returns:
        A fresh :class:`~injectkit.judge.base.Judge` instance for ``opt_judge_id``.
    """
    assert_judge_separation(opt_judge_id, eval_judge_id)
    from ..judge import get_judge

    return get_judge(opt_judge_id)


def sample_completions(
    model: Any,
    prompt: str,
    *,
    num_samples: int,
    sample_temp: float,
    rng: random.Random,
    max_new_tokens: int = 64,
) -> list[str]:
    """Sample ``num_samples`` completions from ``model`` given ``prompt``.

    The judge-in-the-loop objective (REINFORCE-GCG / UJA) scores a *distribution*
    of completions, so each step draws several. Resolution order:

    * If the model exposes a real ``generate(prompt, *, temperature, ...)`` or
      ``sample(...)`` method (a production white-box HF model), it is used — the
      true sampled-completion GPU path. **DEFERRED-NO-GPU**: never reached on the
      offline CPU stub (which has no such method) and never exercised here against
      a 7-8B model.
    * Otherwise the offline fallback produces ``num_samples`` deterministic,
      seam-only pseudo-completions by round-tripping the prompt (optionally with a
      tiny temperature-seeded perturbation index) through ``token_ids`` /
      ``decode``. This has NO semantic meaning — it exists purely so the
      sampling→judging→loss WIRING is unit-testable on CPU with no torch and no
      model download.

    Args:
        model: The white-box model seam.
        prompt: The full conditioning text (rendered ``prompt + suffix``).
        num_samples: How many completions to draw (``>= 1``).
        sample_temp: Sampling temperature (``> 0``); passed to a real ``generate``.
        rng: Seeded RNG so the offline fallback is reproducible.
        max_new_tokens: Generation cap for the real ``generate`` path.

    Returns:
        ``num_samples`` completion strings.
    """
    num_samples = max(1, int(num_samples))
    gen = getattr(model, "generate", None) or getattr(model, "sample", None)
    if callable(gen):
        # Real sampled-completion path (GPU). DEFERRED-NO-GPU: not exercised on the
        # CPU stub. We tolerate either signature shape a real seam might expose.
        out: list[str] = []
        for _ in range(num_samples):
            try:
                text = gen(prompt, temperature=sample_temp, max_new_tokens=max_new_tokens)
            except TypeError:
                text = gen(prompt)
            out.append(str(text))
        return out

    # Offline deterministic fallback — seam-only, no torch, no network.
    completions: list[str] = []
    for i in range(num_samples):
        # A temperature/seed-perturbed re-decode of the prompt ids: deterministic,
        # offline, and varied enough across samples to drive the loss plumbing.
        ids = list(model.token_ids(prompt))
        if ids:
            shift = (rng.randrange(0, 7) + i) if sample_temp > 0 else 0
            ids = [(int(t) + shift) for t in ids]
        completions.append(model.decode(ids))
    return completions


def judge_scores(
    completions: Sequence[str],
    judge: Any,
    *,
    canary: str,
    target: str,
    trigger: str,
) -> list[float]:
    """Grade each completion with the IN-LOOP OPT ``judge`` → unsafety score in [0,1].

    Wraps every completion string in a :class:`~injectkit.models.TargetResponse`
    and calls ``judge.judge(...)``, returning the judge's continuous
    ``sr_score`` per completion (the benign-marker unsafety proxy). This is the
    per-completion reward REINFORCE-GCG weights its policy gradient by, and the
    quantity UJA maximises directly (``loss = -mean(sr_score)``).

    Args:
        completions: The sampled completion strings to grade.
        judge: A resolved in-loop OPT :class:`~injectkit.judge.base.Judge`.
        canary: The per-run benign canary (so marker judges recognise success).
        target: The benign target string the attack optimises toward (if explicit).
        trigger: The benign-marker prefix (e.g. ``INJECTOK-``).

    Returns:
        One ``sr_score`` float in ``[0, 1]`` per completion.
    """
    scores: list[float] = []
    for text in completions:
        response = TargetResponse(text=str(text))
        verdict = judge.judge(
            response, canary=canary, target=target, trigger=trigger
        )
        scores.append(float(verdict.sr_score))
    return scores
