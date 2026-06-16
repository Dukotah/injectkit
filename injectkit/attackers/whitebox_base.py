"""White-box gradient-suffix attacker BASE — GCG/AmpleGCG contract (v0.3.0 scaffold).

This module freezes the contract for the **white-box** attacker family from the
research survey (``docs/RESEARCH.md`` → "Gradient suffixes (GCG family)";
AmpleGCG arXiv:2404.07921, Mask-GCG arXiv:2509.06350). GCG optimises an
adversarial SUFFIX using the target model's GRADIENTS so the model emits a chosen
string. White-box access ⇒ this is **HuggingFace-target ONLY** (we need logits +
embedding gradients), unlike the black-box adaptive/PAIR loop.

ETHICS — NON-NEGOTIABLE (this module + every subclass enforces):

* **Benign target string.** The optimisation objective is to make a **LOCAL**
  white-box model emit the BENIGN canary marker ``<trigger><canary>`` it was told
  to withhold — a robustness test — NEVER harmful output. The default
  :attr:`GCGConfig.target_string` is the benign marker. No harmful AdvBench/
  AmpleGCG suffix artifact is bundled; published harmful-optimised suffixes are
  referenced ONLY via the gated research loader, never here.
* **White-box ⇒ HF-only, lazy/heavy.** ``torch`` + ``transformers`` are
  **lazy-imported** inside the optimisation method with a friendly
  :class:`~injectkit.attackers.base.AttackerError` if missing. Constructing the
  attacker imports nothing heavy. GCG is COMPUTE-HEAVY — real use wants a GPU; the
  docstrings say so.
* **Stub-testable / no real GCG in CI.** The model + gradients are accessed
  through the :class:`WhiteBoxModel` seam so tests inject a ``StubWhiteBoxModel``
  (fake logits/grads) and run AT MOST a trivial 1-step path. Tests MUST NEVER run
  real optimisation or download a model. ``max_steps`` is capped and honoured.

DEFENSIVE / AUTHORIZED USE ONLY. Run only against a local model you own.

This module is the abstract BASE: :meth:`WhiteBoxGCGAttacker.run` and
:meth:`WhiteBoxGCGAttacker._optimize_suffix` raise ``NotImplementedError`` and
are overridden by the concrete
:class:`~injectkit.attackers.gcg.GCGSuffixAttacker`, which implements the
greedy-coordinate-gradient loop through the :class:`WhiteBoxModel` seam (tested
against ``tests/test_whitebox_gcg.py`` with ``StubWhiteBoxModel`` and
``max_steps=1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from ..evaluators.heuristics import DEFAULT_TRIGGER
from ..models import Attack
from .base import AttackerError, AttackerResult

__all__ = [
    "WhiteBoxModel",
    "GCGConfig",
    "GCGStep",
    "WhiteBoxGCGAttacker",
    "import_torch_transformers",
]


def import_torch_transformers() -> "tuple[Any, Any]":
    """Lazy-import ``torch`` + ``transformers`` with a friendly error if missing.

    Shared by every white-box attacker so the heavy deps stay out of import time.
    Mirrors :func:`injectkit.targets.hf._import_hf` but raises the attacker error
    type. Real use is GPU-recommended and compute-heavy.

    Returns:
        A ``(torch, transformers)`` tuple.

    Raises:
        AttackerError: if either optional dependency is not installed.
    """
    try:
        import torch  # noqa: PLC0415 (intentional lazy import)
        import transformers  # noqa: PLC0415 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch/stub
        raise AttackerError(
            "The white-box GCG attacker requires 'transformers' and 'torch' "
            "(white-box access to logits + gradients). Install them with "
            "`pip install 'injectkit[hf]'`. GCG is compute-heavy — a GPU is "
            "strongly recommended for real runs. Tests use a StubWhiteBoxModel "
            "and need neither dependency."
        ) from exc
    return torch, transformers


@runtime_checkable
class WhiteBoxModel(Protocol):
    """The white-box seam: logits + embedding gradients for a LOCAL HF model.

    This is the abstraction GCG needs and the one tests stub. A real
    implementation wraps an in-process ``transformers`` causal-LM (HF-only,
    because we need parameter/embedding gradients); ``StubWhiteBoxModel`` in
    ``tests/conftest.py`` returns deterministic fake logits/grads so the
    optimisation contract can be exercised with NO torch, NO model download, and
    at most one trivial step.

    All tensor-shaped values are typed ``Any`` so the protocol does not import
    ``torch`` at module load.
    """

    #: Stable identifier for metadata (e.g. "hf:meta-llama/Llama-2-7b").
    name: str

    def token_ids(self, text: str) -> Any:
        """Encode ``text`` to a 1-D tensor/sequence of token ids."""
        ...

    def decode(self, ids: Any) -> str:
        """Decode token ids back to text."""
        ...

    def target_loss(self, input_ids: Any, target_ids: Any) -> Any:
        """Return the LM loss of producing ``target_ids`` after ``input_ids``.

        Lower loss ⇒ the model is closer to emitting the (benign) target string.
        GCG minimises this over candidate suffix tokens.
        """
        ...

    def token_gradients(self, input_ids: Any, target_ids: Any, suffix_slice: Any) -> Any:
        """Return the gradient of the target loss w.r.t. the one-hot suffix tokens.

        The core white-box signal: a ``[suffix_len, vocab]`` tensor whose top-k
        most-negative entries are GCG's candidate replacement tokens.
        """
        ...


@dataclass
class GCGConfig:
    """Frozen configuration for a GCG-style suffix optimisation.

    Defaults are SMALL and SAFE for tests; real runs raise ``max_steps`` /
    ``batch_size`` (and want a GPU). The ``target_string`` defaults to the BENIGN
    marker so the optimisation objective is the canary, never harm.
    """

    #: Number of optimisation steps. Capped tiny by default; tests pass 1.
    max_steps: int = 50
    #: Length (in tokens) of the adversarial suffix being optimised.
    suffix_len: int = 20
    #: Per-step number of candidate token swaps evaluated (the GCG batch).
    batch_size: int = 64
    #: Top-k most-promising replacement tokens drawn from the gradient per slot.
    top_k: int = 256
    #: The string the model is optimised to emit. Default: the benign marker,
    #: rendered with the per-run canary by the attacker (NEVER a harmful string).
    target_string: Optional[str] = None
    #: Initial suffix (a benign filler like ``"! ! ! ..."``); the optimiser mutates it.
    init_suffix: Optional[str] = None
    #: Success-marker prefix used to build the benign target string.
    trigger: str = DEFAULT_TRIGGER
    #: RNG seed for reproducible candidate sampling.
    seed: int = 0


@dataclass
class GCGStep:
    """One optimisation step, recorded for the transcript/report.

    Captures the suffix at this step and its target loss, so the full GCG
    trajectory is auditable and reproducible.
    """

    step: int
    #: The adversarial suffix string at this step (benign-marker objective).
    suffix: str
    #: The target loss after this step's swap (lower is closer to emitting marker).
    loss: float
    #: True if the (benign) target string was emitted at this step.
    succeeded: bool = False


class WhiteBoxGCGAttacker:
    """Base white-box GCG attacker — HF-only, lazy-torch, benign target, stubbable.

    Implements the :class:`~injectkit.attackers.base.AdaptiveAttacker` protocol
    (``name``, ``max_rounds``, ``run``) so the engine/CLI drive it like the
    black-box attackers, but it optimises a gradient SUFFIX against a LOCAL
    white-box model rather than rewriting the prompt. Concrete variants
    (plain GCG, AmpleGCG-style, Mask-GCG pruning) subclass and specialise
    :meth:`_optimize_suffix`.

    The objective is strictly the BENIGN canary: a step "succeeds" iff the model
    emits the per-run marker ``<trigger><canary>``. No harmful target string is
    ever set; ``GCGConfig.target_string`` defaults to the marker.

    Args:
        model: The white-box model seam (:class:`WhiteBoxModel`). Production wraps
            an in-process HF model; tests inject ``StubWhiteBoxModel``.
        config: The :class:`GCGConfig` (steps/suffix length/etc.). Defaults are
            tiny and test-safe.
        name: Stable identifier (default ``"gcg"``).

    NOTE: ``max_rounds`` mirrors ``config.max_steps`` so the protocol's budget
    field is meaningful; ``run`` MUST honour it and NEVER exceed it.
    """

    def __init__(
        self,
        model: WhiteBoxModel,
        config: Optional[GCGConfig] = None,
        *,
        name: str = "gcg",
    ) -> None:
        self.model = model
        self.config = config or GCGConfig()
        if self.config.max_steps < 1:
            raise AttackerError("GCGConfig.max_steps must be >= 1.")
        self.name = name
        #: Protocol budget — kept in sync with the GCG step budget.
        self.max_rounds = self.config.max_steps

    def run(
        self,
        seed_attack: Attack,
        target: object,
        detectors: object,
    ) -> AttackerResult:
        """Optimise a benign-marker suffix against the white-box model.

        Satisfies :meth:`injectkit.attackers.base.AdaptiveAttacker.run`. For the
        white-box attacker, ``target`` is expected to be (or wrap) the same local
        HF model exposed via :attr:`model`; ``detectors`` score the final emitted
        text the same way the black-box path does, so the per-step
        :class:`~injectkit.models.AttackResult` projects into the existing
        Finding/report machinery.

        This base method is abstract — the concrete
        :class:`~injectkit.attackers.gcg.GCGSuffixAttacker` overrides it to build
        the benign ``target_string`` (the rendered marker), call
        :meth:`_optimize_suffix`, append the optimised suffix to the seed payload,
        send the result to ``target``, score it, and assemble an
        :class:`AttackerResult` whose transcript carries the per-step
        :class:`GCGStep`s (honouring ``config.max_steps`` exactly and stopping
        early on the first benign-marker success).

        Raises:
            AttackerError: only on unrecoverable setup (missing torch/transformers,
                surfaced via :func:`import_torch_transformers` when the real model
                is first touched). Per-step faults belong in the transcript.
        """
        raise NotImplementedError("WhiteBoxGCGAttacker.run is a v0.3.0 builder TODO")

    def _optimize_suffix(
        self,
        prompt_ids: Any,
        target_ids: Any,
    ) -> list[GCGStep]:
        """Run the GCG coordinate-descent loop, returning the per-step trajectory.

        The white-box core: each step computes :meth:`WhiteBoxModel.token_gradients`,
        samples ``top_k`` candidate replacements per suffix slot, evaluates a
        ``batch_size`` batch via :meth:`WhiteBoxModel.target_loss`, keeps the
        lowest-loss swap, and records a :class:`GCGStep`. Stops at
        ``config.max_steps`` or on a benign-marker success.

        This base method is abstract; the concrete
        :class:`~injectkit.attackers.gcg.GCGSuffixAttacker` implements the loop
        using ONLY the :class:`WhiteBoxModel` seam (so ``StubWhiteBoxModel`` drives
        it — tests run ``max_steps=1`` with fake grads, completing one step with no
        real torch op, and NEVER bundle or load a harmful suffix).
        """
        raise NotImplementedError(
            "WhiteBoxGCGAttacker._optimize_suffix is a v0.3.0 builder TODO"
        )
