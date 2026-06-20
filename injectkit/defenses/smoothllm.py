"""SmoothLLM defense — randomized smoothing via char-level perturbations.

Reference: Robey et al., "SmoothLLM: Defending Large Language Models Against
Jailbreaking Attacks", arXiv:2310.03684 (2023).

The core idea: given an input prompt, generate N independent perturbed copies
by randomly swapping, inserting, or patching single characters. Send each copy
to the target; observe which ones trigger a "success" signal (the injected
marker is echoed). Aggregate by majority vote: if more than N/2 copies trigger
success, the smoothed response is a success; otherwise it is not. Random
perturbations disrupt adversarially-crafted suffixes (which are brittle to
character-level noise) while leaving natural text functionally intact most of
the time.

Honesty note
------------
This implementation uses char-level perturbations exactly as in the original
paper (random swap, insert, patch at rate q). Unlike neural smoothing, this
does NOT guarantee a certified radius; it is a statistical defense that reduces
empirical ASR on most adversarial suffixes but is not robust against attacks
specifically crafted to survive char-level noise. The benchmark measures honest
empirical ASR, not a certificate.

Engine seam
-----------
The Defense protocol (``wrap_system / filter_input / filter_output``) assumes a
single target query per attack turn. SmoothLLM needs N queries. We extend the
seam MINIMALLY and BACKWARDS-COMPATIBLY:

  - ``SmoothLLMDefense`` exposes an optional ``smooth_queries(prompt, n_copies,
    rng)`` method that returns N perturbed copies of the prompt.
  - The ``_SmoothLLMTarget`` wrapper (used internally by ``run_defended`` when
    the defense is a SmoothLLMDefense) calls ``smooth_queries`` to get N
    prompts, sends each to the inner target, majority-votes the results, and
    returns a synthetic :class:`~injectkit.models.TargetResponse` that reflects
    the majority outcome.
  - All existing single-query defenses are unaffected: they never implement
    ``smooth_queries``, so ``_SmoothLLMTarget`` is never injected for them.

The engine detects a SmoothLLM defense in ``run_defended`` / ``_with_defense``
and wraps appropriately. No other engine paths change.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import random
import string
from typing import Optional, Tuple

from ..evaluators.heuristics import DEFAULT_TRIGGER
from ..models import TargetResponse
from ..targets.base import Target
from .base import Defense, register_defense

__all__ = [
    "SmoothLLMDefense",
    "apply_perturbation",
    "_SmoothLLMTarget",
]

# --------------------------------------------------------------------------- #
# Perturbation operations (char-level, deterministic under a seed).
# --------------------------------------------------------------------------- #

#: Characters available for random insertion and patch.
_PRINTABLE_CHARS = string.ascii_letters + string.digits + string.punctuation + " "


def _perturb_swap(text: str, rng: random.Random) -> str:
    """Swap two randomly chosen adjacent characters (no-op if len < 2)."""
    if len(text) < 2:
        return text
    i = rng.randrange(len(text) - 1)
    chars = list(text)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _perturb_patch(text: str, rng: random.Random) -> str:
    """Replace one randomly chosen character with a random printable char."""
    if not text:
        return text
    i = rng.randrange(len(text))
    chars = list(text)
    chars[i] = rng.choice(_PRINTABLE_CHARS)
    return "".join(chars)


def _perturb_insert(text: str, rng: random.Random) -> str:
    """Insert a random printable char at a random position."""
    i = rng.randrange(len(text) + 1)
    ch = rng.choice(_PRINTABLE_CHARS)
    return text[:i] + ch + text[i:]


_OPS = [_perturb_swap, _perturb_patch, _perturb_insert]


def apply_perturbation(text: str, q: float, rng: random.Random) -> str:
    """Apply random char-level perturbations to ``text`` at rate ``q``.

    Each character position is independently considered for perturbation with
    probability ``q``. One of three operations (swap-adjacent, patch, insert) is
    chosen uniformly at random for each selected position.

    Args:
        text: The string to perturb.
        q: Perturbation rate in ``[0, 1]``. 0 returns ``text`` unchanged; 1
           perturbs (roughly) every character.
        rng: Seeded RNG for determinism.

    Returns:
        The perturbed string (may be slightly longer than ``text`` if inserts
        are chosen).
    """
    if not text or q <= 0.0:
        return text
    result = text
    for _ in range(len(text)):
        if rng.random() < q:
            op = rng.choice(_OPS)
            result = op(result, rng)
    return result


# --------------------------------------------------------------------------- #
# SmoothLLMDefense
# --------------------------------------------------------------------------- #


class SmoothLLMDefense:
    """Defend via randomised smoothing: N perturbed copies + majority vote.

    Implements the SmoothLLM algorithm (Robey et al., arXiv:2310.03684):

      1. Generate ``n_copies`` independent perturbed copies of the input prompt
         by applying random char-level swap / patch / insert operations at rate
         ``q`` (the perturbation rate).
      2. Send each perturbed copy to the target (via ``_SmoothLLMTarget``).
      3. Majority-vote the results: if the success marker is present in more
         than ``n_copies // 2`` responses, the defense considers it a success
         (the injection broke through); otherwise, the aggregate response does
         NOT contain the marker (the defense blocked it).

    Because step 2 requires N target queries per attack turn, this defense uses
    the ``smooth_queries`` extension point. The engine's ``run_defended`` method
    detects ``SmoothLLMDefense`` and uses ``_SmoothLLMTarget`` to orchestrate
    the multi-query loop.

    Args:
        n_copies: Number of perturbed copies to generate per attack (N in the
            paper; 10–20 is typical). Higher N = more robust but N × slower.
        q: Perturbation rate in ``[0, 1]``. The paper uses q=0.05–0.20 for a
           good robustness/utility tradeoff. Higher q = more perturbation =
           stronger smoothing but more benign-text distortion.
        seed: RNG seed for reproducibility. When ``None``, uses a fixed default
           so the defense is deterministic by default. Pass different seeds
           across benchmark runs for statistical confidence.
        trigger: The benign-marker prefix the defense uses to detect success
           in individual copies' responses (must match the engine's trigger).

    Notes:
        - ``wrap_system`` and ``filter_output`` are passthroughs; all defense
          logic lives in the N-query loop orchestrated by ``_SmoothLLMTarget``.
        - ``filter_input`` is also a passthrough; the perturbation is applied
          INSIDE ``_SmoothLLMTarget.send()`` so it runs N times with fresh
          RNG state.
    """

    name = "smoothllm"

    def __init__(
        self,
        n_copies: int = 10,
        q: float = 0.10,
        seed: int = 42,
        trigger: str = DEFAULT_TRIGGER,
    ) -> None:
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}")
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0, 1], got {q}")
        self.n_copies = n_copies
        self.q = q
        self.seed = seed
        self.trigger = trigger

    # -- Defense protocol hooks -------------------------------------------

    def wrap_system(self, system: Optional[str]) -> Optional[str]:
        """Passthrough — smoothing acts on the prompt, not the system prompt."""
        return system

    def filter_input(
        self, prompt: str, context: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """Passthrough — perturbation happens inside ``_SmoothLLMTarget``."""
        return prompt, context

    def filter_output(self, text: str) -> str:
        """Passthrough — the majority-vote result is already the final output."""
        return text

    # -- SmoothLLM extension point ----------------------------------------

    def smooth_queries(
        self, prompt: str, n_copies: int, rng: random.Random
    ) -> list[str]:
        """Return ``n_copies`` independently perturbed copies of ``prompt``.

        Called by ``_SmoothLLMTarget`` to generate the N variants. Each copy
        uses a fresh RNG state derived from the shared ``rng`` so the copies
        are independent but the whole set is deterministic under the seed.

        Args:
            prompt: The (already filter_input'd) prompt to perturb.
            n_copies: How many copies to generate.
            rng: The seeded RNG (advanced in place).

        Returns:
            A list of ``n_copies`` perturbed strings.
        """
        return [apply_perturbation(prompt, self.q, rng) for _ in range(n_copies)]


# --------------------------------------------------------------------------- #
# _SmoothLLMTarget — the multi-query wrapper
# --------------------------------------------------------------------------- #


class _SmoothLLMTarget:
    """Wrap a :class:`Target` so ``SmoothLLMDefense`` orchestrates N queries.

    For each ``send`` call:

      1. Apply ``defense.wrap_system`` once (system hardening, if any).
      2. Apply ``defense.filter_input`` once to get the (possibly pre-filtered)
         prompt and context.
      3. Call ``defense.smooth_queries`` to generate ``n_copies`` perturbed
         prompts.
      4. Send each perturbed prompt (with the same system/context) to the inner
         target and collect responses.
      5. Majority-vote: count how many responses contain the trigger marker.
         - If MAJORITY (> n_copies // 2) contain the marker ⟹ the injection
           succeeded despite smoothing ⟹ return a response WITH the marker
           (honest: the defense did not block it).
         - Otherwise ⟹ the injection failed (smoothing blocked it) ⟹ return
           a neutral non-success response WITHOUT the marker.

    This wrapper is injected by the engine's ``_with_defense`` when the defense
    is a ``SmoothLLMDefense``; ordinary defenses never trigger this path.

    Args:
        inner: The underlying target to query.
        defense: The ``SmoothLLMDefense`` orchestrating the smoothing.
    """

    def __init__(self, inner: Target, defense: SmoothLLMDefense) -> None:
        self.inner = inner
        self.defense = defense
        self.name = getattr(inner, "name", "target")
        # The RNG is seeded from the defense seed; each send advances it in
        # place so consecutive attacks get different perturbations (deterministic
        # under the fixed seed across a full benchmark run).
        self._rng = random.Random(defense.seed)
        self._marker_re_str = defense.trigger

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Run the N-copy smooth-and-vote loop; return the majority-voted result."""
        d = self.defense

        # Hook 1: system hardening (once, shared across all copies).
        try:
            system = d.wrap_system(system)
        except Exception:  # noqa: BLE001
            pass

        # Hook 2: input pre-filter (once; perturbation is layered on top).
        try:
            result = d.filter_input(prompt, context)
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], str)
            ):
                prompt, context = result
        except Exception:  # noqa: BLE001
            pass

        # Hook 3: generate N perturbed copies.
        try:
            perturbed = d.smooth_queries(prompt, d.n_copies, self._rng)
        except Exception:  # noqa: BLE001 - fall back to single unperturbed query
            perturbed = [prompt]

        # Hook 4: send each copy to the inner target.
        responses: list[TargetResponse] = []
        for p in perturbed:
            try:
                resp = self.inner.send(p, system=system, context=context)
            except Exception as exc:  # noqa: BLE001
                resp = TargetResponse(
                    text="",
                    error=f"inner target raised {type(exc).__name__}: {exc}",
                )
            if not isinstance(resp, TargetResponse):
                resp = TargetResponse(text="", error="inner target returned non-TargetResponse")
            responses.append(resp)

        # Hook 5: majority vote over responses.
        return self._majority_vote(responses)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _has_marker(self, text: str) -> bool:
        """True if ``text`` contains the trigger marker."""
        return self.defense.trigger in text

    def _majority_vote(self, responses: list[TargetResponse]) -> TargetResponse:
        """Return the majority-vote aggregate response.

        The vote is over the PRESENCE of the trigger marker in each response's
        text — the minimal signal the engine uses to detect success. This is
        exactly the vote the SmoothLLM paper performs over model "jailbreak"
        outputs.

        If majority succeed ⟹ return the first marker-containing response (the
        injection broke through). If majority fail ⟹ return a clean benign
        response (the defense held). Errored responses count as non-success but
        do not contribute a clean text.
        """
        n = len(responses)
        if n == 0:
            return TargetResponse(text="", error="smoothllm: no copies were sent")

        success_count = sum(
            1 for r in responses
            if r.error is None and self._has_marker(r.text or "")
        )

        if success_count > n // 2:
            # Majority says the injection succeeded — return any marker response.
            for r in responses:
                if r.error is None and self._has_marker(r.text or ""):
                    return r
        # Majority says injection failed (defense held). Return a neutral response.
        # We prefer the first non-errored response for the model/stop_reason metadata.
        for r in responses:
            if r.error is None:
                return TargetResponse(
                    text="Smoothed: injection did not reach majority threshold.",
                    refused=False,
                    stop_reason=r.stop_reason,
                    model=r.model,
                )
        # All copies errored — propagate the first error.
        return responses[0]


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def _register() -> None:
    """Register SmoothLLMDefense on the default registry (idempotent)."""
    from .base import registry as _registry

    if SmoothLLMDefense.name not in _registry.names():
        register_defense(SmoothLLMDefense.name, SmoothLLMDefense)


_register()
