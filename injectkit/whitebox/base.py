"""The v0.4 white-box :class:`Attack` ABC — the shared attack interface.

This freezes the single contract every white-box attack family (GCG variants,
embedding, abliteration, fine-tuning, prefill, generative-suffix) implements, as
specified in ROADMAP §6.0:

    class Attack(ABC):
        name: str                 # registry key
        supported_arch: set[str]  # {"dense"} | {"dense","moe"} — checked vs zoo.yaml
        @abstractmethod
        def run(self, model, tokenizer, messages, target, cfg,
                defense=None) -> AttackResult: ...

It is deliberately distinct from two pre-existing v0.3 abstractions and does not
replace either:

* :class:`injectkit.models.Attack` — the *corpus test-case* dataclass.
* :class:`injectkit.attacks.base.AttackStrategy` — the multi-turn turn-sequence
  generator.

The v0.4 ``Attack`` is the **white-box optimiser** seam: given an in-process model
(+ tokenizer) it optimises an adversarial object (suffix / embeddings / prefill
text / weight delta) against a benign target and returns an :class:`AttackResult`.

If ``defense`` is supplied to :meth:`Attack.run`, the attack runs in **adaptive
mode** (ROADMAP §6.13) — the optimiser may use knowledge of the defense in its
loss/loop. A plain (undefended) run passes ``defense=None``.

ETHICS — NON-NEGOTIABLE: the optimisation objective is ALWAYS the per-run BENIGN
canary marker ``<trigger><canary>`` (a robustness probe), never harmful content.
Every concrete attack inherits this constraint.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .config import AttackConfig

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from ..defenses.base import Defense

__all__ = [
    "Attack",
    "AttackResult",
    "ArchitectureError",
]


class ArchitectureError(RuntimeError):
    """Raised when an attack is run against an unsupported model architecture.

    Gradient-family attacks assume a dense-transformer backward pass with stable
    per-token gradients (ROADMAP §6.14). MoE routing is non-differentiable, so an
    attack declaring ``supported_arch = {"dense"}`` refuses a ``"moe"`` target up
    front rather than producing silently-wrong gradients.
    """


@dataclass
class AttackResult:
    """The outcome of one white-box :meth:`Attack.run` (ROADMAP §6.0).

    The structured record every attack family returns, parallel to but distinct
    from the corpus-path :class:`injectkit.models.AttackResult` (which aggregates
    detector verdicts). This one captures the *optimisation* outcome — the best
    adversarial input found, its loss, the per-step trajectory, and the
    reproducibility metadata — so the bench/leaderboard layer (later chunks) can
    record and replay it.

    ``optimized_obj`` is the family-specific artifact: a suffix string (GCG), an
    embedding tensor (soft-prompt), a prefill string (prefill), or a weight delta
    (abliteration/FT). ``optimized_obj_kind`` names which.
    """

    #: Registry name of the attack that produced this result (e.g. "gcg").
    attack_name: str
    #: The best adversarial *input* found (the full text/messages actually sent).
    best_input: str
    #: Lowest optimisation loss reached (lower ⇒ closer to the benign target).
    best_loss: float
    #: Per-step optimisation loss curve (the golden-loss regression tripwire,
    #: ROADMAP §8). Empty for non-iterative attacks (e.g. prefill).
    per_step_losses: list[float] = field(default_factory=list)
    #: The family-specific optimised artifact (suffix | embeddings | prefill text
    #: | delta_weights). Type is family-dependent; ``Any`` here.
    optimized_obj: Any = None
    #: Names which kind of artifact ``optimized_obj`` is, for the artifacts emitter.
    optimized_obj_kind: str = ""
    #: Whether the run reached the benign-marker success condition.
    succeeded: bool = False
    #: Number of model queries / candidate evaluations used (budget accounting).
    queries: int = 0
    #: Wall-clock seconds the run took (0.0 if not measured by the caller).
    wall_clock_s: float = 0.0
    #: Peak VRAM in bytes if measured (0 on CPU / when unmeasured).
    peak_vram: int = 0
    #: Id of the defense the attack ran adaptively against, or "" if undefended.
    defense_id: str = ""
    #: Free-form reproducibility stamp (tool version, model revision, seed, quant,
    #: judge-id, ...). Populated by the bench layer in later chunks.
    stamp: dict[str, Any] = field(default_factory=dict)


class Attack(ABC):
    """Abstract base for every v0.4 white-box attack (ROADMAP §6.0).

    Subclasses set two class attributes and implement :meth:`run`:

    * :attr:`name` — the registry key (``@register`` derives the registry entry
      from it).
    * :attr:`supported_arch` — the set of model architectures the attack supports,
      checked against the zoo's ``arch`` flag (``dense`` | ``moe``) before any run
      via :meth:`check_arch`. Gradient families declare ``{"dense"}``.

    The single method, :meth:`run`, optimises an adversarial object against
    ``model`` (a white-box model seam / in-process HF model) using ``tokenizer``,
    starting from ``messages`` (chat turns) toward the benign ``target`` string,
    parameterised by a typed ``cfg`` :class:`~injectkit.whitebox.config.AttackConfig`,
    and returns an :class:`AttackResult`. When ``defense`` is supplied the attack
    runs in adaptive mode (ROADMAP §6.13).
    """

    #: Registry key for this attack (e.g. "gcg"). Set by every concrete subclass.
    name: str = ""
    #: Architectures this attack supports; checked against the model zoo's arch
    #: flag before a run. Gradient families are dense-only for v0.4–v1.0 (§6.14).
    supported_arch: set[str] = {"dense"}

    def check_arch(self, arch: str) -> None:
        """Raise :class:`ArchitectureError` if ``arch`` is unsupported.

        Called by the run path (and the bench layer) before optimising, so a
        gradient attack refuses a MoE target up front instead of producing
        silently-corrupt gradients (ROADMAP §6.14, §8 "supported_arch vs zoo.yaml").

        Args:
            arch: The target model's architecture flag from the zoo
                (``"dense"`` or ``"moe"``).

        Raises:
            ArchitectureError: if ``arch`` is not in :attr:`supported_arch`.
        """
        if arch not in self.supported_arch:
            raise ArchitectureError(
                f"attack {self.name!r} supports arch {sorted(self.supported_arch)} "
                f"but was asked to run on a {arch!r} model. Gradient-family attacks "
                "are dense-only for v0.4 (MoE routing is non-differentiable; see "
                "ROADMAP §6.14)."
            )

    @abstractmethod
    def run(
        self,
        model: Any,
        tokenizer: Any,
        messages: list[dict],
        target: str,
        cfg: AttackConfig,
        defense: "Optional[Defense]" = None,
    ) -> AttackResult:
        """Optimise an adversarial object against ``model`` and return the result.

        Args:
            model: The white-box model seam — an in-process model exposing the
                logits/gradients the family needs (the only backend with a
                backward pass). In tests this is an offline stub.
            tokenizer: The model's tokenizer. May be ``None`` for model seams that
                tokenise internally (the v0.3 GCG seam does).
            messages: The chat turns to attack, as ``{"role", "content"}`` dicts
                (the rendered prompt the optimiser appends/prefills against).
            target: The string the model is optimised to emit. The caller passes
                the BENIGN per-run marker; a harmful string is never passed.
            cfg: The typed :class:`~injectkit.whitebox.config.AttackConfig`
                subclass for this family (validated knobs: steps, seed, ...).
            defense: Optional :class:`~injectkit.defenses.base.Defense`. When
                supplied the attack runs in ADAPTIVE mode (ROADMAP §6.13),
                optimising with knowledge of the defense in the loss/loop. When
                ``None`` the run is the undefended baseline.

        Returns:
            An :class:`AttackResult` with the best input, loss curve, optimised
            artifact, and reproducibility metadata.
        """
        ...
