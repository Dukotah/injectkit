"""Continuous embedding / soft-prompt attack — the capability-ceiling line.

CHUNK 10-embedding-attack (ROADMAP §6.6). The **continuous embedding attack** of
Schwinn, Geisler, et al. — *"Soft Prompt Threats: Attacking Safety Alignment and
Unlearning in Open-Source LLMs through the Embedding Space"*, **arXiv:2402.09063**
(NeurIPS 2024); reference implementation ``SchwinnL/LLM_Embedding_Attack``.

Where GCG optimises *discrete* suffix tokens (greedy coordinate descent over the
vocab), the embedding attack optimises the **input embeddings directly** — a
*continuous* gradient-descent attack with **no discrete projection**. It prepends
``k`` trainable embedding vectors to the rendered prompt's embeddings and runs
Adam on the affirmative-target NLL loss, so the optimisation variable lives in
``ℝ^{k×d}`` (continuous) rather than ``{0,1}^{L×V}`` (one-hot discrete). Because it
never has to round a continuous step back onto the token simplex, it is **faster
and stronger than discrete GCG** and is reported here as a *capability ceiling*
(the strongest white-box signal — what an attacker with full weight + embedding
access can achieve), not as a transferable suffix. It also doubles as an
unlearning / data-extraction probe (paper §5). **Only weight access enables this**
— a black-box tool structurally cannot reach the embedding layer.

PAPER PARITY NUMBER (recorded for the repro stamp; arXiv:2402.09063): the
embedding attack reaches **higher ASR at lower wall-clock than discrete GCG** on
the open-weight 7-8B targets (Llama-2-7B / Vicuna-7B / Mistral-7B), because the
continuous step removes GCG's discrete-search bottleneck.

The "**embedding-ASR ≥ GCG-ASR at lower wall-clock on an 8B model**" claim needs a
GPU + a real 7-8B target, so that NUMBER is **DEFERRED-NO-GPU**: the full
optimisation code path below is production-complete and the loop is verified to
run + converge on a TINY CPU model (GPT-2 / Pythia-160M, fixed seed) and on the
offline stub seam; the headline ASR/wall-clock comparison is not run in this
environment.

DESIGN — one new seam, reuse everything else. The discrete GCG seam
(:class:`~injectkit.attackers.whitebox_base.WhiteBoxModel`) exposes
``target_loss(input_ids, target_ids)`` and ``token_gradients(...)`` — gradients
w.r.t. *one-hot tokens*. The embedding attack instead needs gradients w.r.t.
*input embeddings*, which is a different forward path (``inputs_embeds=...``). So
this module defines a small additional seam, :class:`EmbeddingModel`
(``embed`` + ``loss_from_embeds`` + ``embedding_dim`` + token decode), satisfied:

* in production by :class:`HFEmbeddingModel`, a thin adapter over an in-process
  ``transformers`` causal-LM (``model.get_input_embeddings()`` for ``embed`` and a
  forward with ``inputs_embeds=`` for the differentiable loss — ``torch`` is
  **lazy-imported**, never at module load), and
* offline/in-tests by any object exposing the same three methods — the test suite
  drives a pure-Python toy embedding model (no torch, no download).

The optimiser (:class:`EmbeddingAttack`) is seam-agnostic: it prepends ``k``
trainable soft-prompt vectors and runs **Adam** to minimise the affirmative-target
loss. With ``torch`` available it uses ``torch.optim.Adam`` on real autograd; on a
pure-Python seam (no torch) it falls back to a **from-scratch Adam** driving the
seam's scalar loss + finite-difference / analytic gradient, so the *exact same
optimisation contract* (k soft vectors, Adam, target-NLL, per-step loss curve,
early stop on convergence) is exercised on CPU with no heavy deps.

NO TOKENIZER ROUND-TRIP. The discrete GCG correctness trap — drop any candidate
whose ``encode(decode(ids)) != ids`` (``filter_ids``) — **does not apply** to a
continuous attack: the optimised object is a real-valued embedding tensor, not a
token sequence, so there is no re-tokenisation step that could silently change the
loss the model sees. (The optional ASETF extension below *does* translate the soft
embeddings back to discrete tokens for transfer — and *that* path re-introduces a
round-trip, documented there.)

ETHICS — NON-NEGOTIABLE: the optimisation objective is ALWAYS the per-run BENIGN
canary marker ``<trigger><canary>`` (the ``target`` the loop carries), never
harmful content. The soft prompt is an internal robustness probe; no harmful
embedding artifact is bundled.

DEFENSIVE / AUTHORIZED USE ONLY — run only against a local model you own.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from pydantic import Field

from .base import Attack, AttackResult
from .config import AttackConfig
from .registry import register
from .targets import advprefix_target

__all__ = [
    "EmbeddingConfig",
    "EmbeddingModel",
    "HFEmbeddingModel",
    "EmbeddingAttack",
    "SoftPrompt",
    "asetf_translate",
    "run",
    "PAPER_CLAIM",
]

#: Paper parity (arXiv:2402.09063, NeurIPS 2024) — recorded in the repro stamp.
#: The continuous embedding attack reaches higher ASR at lower wall-clock than
#: discrete GCG on open-weight 7-8B targets. The NUMBER is DEFERRED-NO-GPU.
PAPER_CLAIM = "embedding-ASR >= GCG-ASR at lower wall-clock (arXiv:2402.09063)"


# --------------------------------------------------------------------------- #
# Typed config (ROADMAP §6.6: EmbeddingConfig(k, lr, num_steps, optim_method))
# --------------------------------------------------------------------------- #


class EmbeddingConfig(AttackConfig):
    """Typed config for the continuous embedding attack (arXiv:2402.09063).

    Adds the soft-prompt knobs on top of the shared :class:`AttackConfig`. The
    cross-attack fields are inherited (``max_steps`` is an upper bound that
    ``num_steps`` defers to; ``target`` / ``trigger`` build the benign objective;
    ``seed`` makes the soft-prompt init reproducible).

    The optimised object is ``k`` continuous embedding vectors prepended to the
    rendered prompt's embeddings; ``num_steps`` Adam updates at learning rate
    ``lr`` minimise the affirmative-target NLL. Defaults are small/safe so the
    offline CPU path runs fast; a real 7-8B run raises ``num_steps`` (and wants a
    GPU).
    """

    #: Number of trainable soft-prompt embedding vectors prepended to the prompt
    #: embeddings (the paper's attack length). ``k=20`` is a typical setting.
    k: int = Field(default=20, ge=1)
    #: Adam learning rate for the continuous embedding update (paper ~1e-3..1e-2).
    lr: float = Field(default=0.001, gt=0.0)
    #: Number of optimisation (Adam) steps. Bounded by the shared ``max_steps``:
    #: the loop runs ``min(num_steps, max_steps)`` updates so a small ``max_steps``
    #: (tests pass 1) still caps wall-clock. Defaults to ``max_steps`` when unset.
    num_steps: Optional[int] = Field(default=None, ge=1)
    #: Continuous optimiser. ``"adam"`` (default) is the paper's choice; ``"sgd"``
    #: is the plain-gradient-descent baseline. Both run on the same loss/seam.
    optim_method: str = Field(default="adam")
    #: Loss-improvement threshold for early-stop convergence. When a step improves
    #: the loss by less than this for ``patience`` steps in a row, the loop stops
    #: (the continuous loss is smooth, so it plateaus cleanly). ``0`` ⇒ never.
    convergence_tol: float = Field(default=1e-6, ge=0.0)
    #: Consecutive sub-``convergence_tol`` steps before early stop.
    patience: int = Field(default=3, ge=1)

    def steps(self) -> int:
        """Effective number of Adam steps: ``min(num_steps or max_steps, max_steps)``."""
        want = self.num_steps if self.num_steps is not None else self.max_steps
        return max(1, min(int(want), int(self.max_steps)))


# --------------------------------------------------------------------------- #
# The embedding-level model seam (distinct from the discrete WhiteBoxModel seam)
# --------------------------------------------------------------------------- #


@runtime_checkable
class EmbeddingModel(Protocol):
    """The embedding-level white-box seam the continuous attack optimises through.

    Distinct from :class:`~injectkit.attackers.whitebox_base.WhiteBoxModel` (which
    exposes gradients w.r.t. *one-hot tokens* for discrete GCG): the continuous
    attack needs the *input-embedding* forward path. A real implementation wraps an
    in-process ``transformers`` causal-LM (see :class:`HFEmbeddingModel`); tests
    inject a pure-Python toy model with the same three methods — no ``torch``, no
    download.

    Tensors are typed ``Any`` so this protocol imports nothing heavy at load.
    """

    #: Stable identifier for metadata (e.g. "hf:meta-llama/Llama-2-7b").
    name: str
    #: Width ``d`` of the embedding space (the soft-prompt vectors are ``ℝ^{k×d}``).
    embedding_dim: int

    def embed(self, ids: Sequence[int]) -> Any:
        """Map token ids to their input embeddings — a ``[len(ids), d]`` tensor."""
        ...

    def loss_from_embeds(self, input_embeds: Any, target_ids: Sequence[int]) -> Any:
        """Affirmative-target NLL of ``target_ids`` given ``input_embeds``.

        ``input_embeds`` is the ``[soft_prompt ⊕ prompt]`` embedding matrix. Lower
        loss ⇒ the model is closer to emitting the (benign) target string. The
        attack minimises this over the trainable soft-prompt rows. On the real seam
        the return value is a differentiable scalar (``torch`` autograd); on a
        pure-Python seam it is a plain float and the optimiser uses the seam's
        :meth:`grad_from_embeds` (or a finite-difference fallback).
        """
        ...

    def token_ids(self, text: str) -> Sequence[int]:
        """Encode ``text`` to token ids (for embedding the static prompt)."""
        ...


# --------------------------------------------------------------------------- #
# Production seam: a thin adapter over an in-process HF causal-LM (lazy torch).
# --------------------------------------------------------------------------- #


class HFEmbeddingModel:
    """Adapter exposing the :class:`EmbeddingModel` seam over an HF causal-LM.

    Wraps an in-process ``transformers`` model + tokenizer and implements the
    embedding-level forward path the continuous attack needs:

    * :meth:`embed` → ``model.get_input_embeddings()(ids)``.
    * :meth:`loss_from_embeds` → a forward with ``inputs_embeds=`` and a
      cross-entropy on the target token positions only (the affirmative-target
      NLL), returning a **differentiable** scalar so ``torch.optim.Adam`` updates
      the soft-prompt rows by real autograd.

    ``torch`` + ``transformers`` are **lazy-imported** on first use (never at
    module load), with the shared friendly error if absent. White-box ⇒ HF-only and
    compute-heavy; a real run wants a GPU (DEFERRED-NO-GPU for the headline number).

    Args:
        model: An in-process ``transformers`` causal-LM (provides
            ``get_input_embeddings`` and a ``inputs_embeds=`` forward).
        tokenizer: The model's tokenizer (encodes the static prompt / target).
        name: Stable identifier for metadata.
        device: Optional device string; defaults to the model's own device.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        name: str = "",
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name or getattr(model, "name_or_path", "") or "hf-embedding-model"
        self._device = device
        self._embedding_dim: Optional[int] = None

    # -- lazy torch -------------------------------------------------------- #

    @staticmethod
    def _torch() -> Any:
        from ..attackers.whitebox_base import import_torch_transformers

        torch, _ = import_torch_transformers()
        return torch

    @property
    def device(self) -> Any:
        if self._device is not None:
            return self._device
        try:
            return next(self.model.parameters()).device
        except Exception:  # pragma: no cover - defensive (CPU stub HF)
            return "cpu"

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            emb = self.model.get_input_embeddings()
            self._embedding_dim = int(emb.weight.shape[1])
        return self._embedding_dim

    # -- seam -------------------------------------------------------------- #

    def token_ids(self, text: str) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return list(ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def embed(self, ids: Sequence[int]) -> Any:
        torch = self._torch()
        emb = self.model.get_input_embeddings()
        id_tensor = torch.tensor([list(ids)], device=self.device, dtype=torch.long)
        # [1, len, d] -> [len, d]
        return emb(id_tensor)[0]

    def embedding_matrix(self) -> Any:
        """The full ``[V, d]`` input-embedding matrix (for ASETF nearest-token)."""
        return self.model.get_input_embeddings().weight

    def loss_from_embeds(self, input_embeds: Any, target_ids: Sequence[int]) -> Any:
        """Cross-entropy NLL of ``target_ids`` given ``input_embeds`` (differentiable).

        Builds ``[input_embeds ⊕ embed(target_ids)]``, runs one forward with
        ``inputs_embeds=``, and computes CE on the **target positions only** — i.e.
        the model's loss for producing the affirmative target after the
        soft-prompted prompt. Returns the autograd scalar so the caller's Adam step
        backprops into the soft-prompt rows.
        """
        torch = self._torch()
        target_ids = list(target_ids)
        if not target_ids:
            # Degenerate: no target ⇒ no loss signal. Return a zero-grad scalar.
            return input_embeds.sum() * 0.0

        target_embeds = self.embed(target_ids)  # [T, d]
        full = torch.cat([input_embeds, target_embeds], dim=0).unsqueeze(0)  # [1, P+T, d]
        out = self.model(inputs_embeds=full)
        logits = out.logits[0]  # [P+T, V]
        # The logit at position i predicts token i+1; the first target token is
        # predicted by the last prompt position. Slice the (P+T) logits so the
        # T target ids are scored against the (P-1 .. P+T-2) prediction logits.
        n_prompt = input_embeds.shape[0]
        pred = logits[n_prompt - 1 : n_prompt - 1 + len(target_ids)]  # [T, V]
        tgt = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        return torch.nn.functional.cross_entropy(pred, tgt)


# --------------------------------------------------------------------------- #
# The optimised artifact + the optimiser
# --------------------------------------------------------------------------- #


class SoftPrompt:
    """The optimised continuous artifact: ``k`` soft-prompt embedding rows + meta.

    ``vectors`` is the ``[k, d]`` matrix (a ``torch.Tensor`` on the real seam, a
    nested ``list[list[float]]`` on the pure-Python seam). It is a *continuous*
    object — there is no token round-trip (see module docstring). :meth:`tolist`
    gives a JSON-serialisable view for the artifacts emitter; :func:`asetf_translate`
    optionally projects it back to discrete tokens for transfer.
    """

    def __init__(self, vectors: Any, *, k: int, dim: int) -> None:
        self.vectors = vectors
        self.k = k
        self.dim = dim

    def tolist(self) -> List[List[float]]:
        """A nested-list (JSON-serialisable) view of the soft-prompt vectors."""
        vecs = self.vectors
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [[float(x) for x in row] for row in vecs]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SoftPrompt(k={self.k}, dim={self.dim})"


@register("embedding")
class EmbeddingAttack(Attack):
    """Continuous embedding / soft-prompt attack (arXiv:2402.09063), dense-only.

    Optimises ``k`` trainable embedding vectors prepended to the rendered prompt's
    embeddings via Adam on the affirmative-target NLL — a *continuous* gradient
    attack with no discrete projection (the capability ceiling, ROADMAP §6.6).
    :meth:`run` adapts the supplied model to the :class:`EmbeddingModel` seam, runs
    the optimiser, and projects the trajectory onto a v0.4
    :class:`~injectkit.whitebox.base.AttackResult` (best soft prompt as
    ``optimized_obj``, kind ``"embeddings"``; per-step loss curve; success flag).

    ``supported_arch`` is the dense default — like every gradient family, the
    continuous embedding attack is scoped to dense transformers for v0.4–v1.0
    (MoE routing is non-differentiable; ROADMAP §6.14).
    """

    name = "embedding"
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
        """Optimise a benign-marker soft prompt via continuous embedding descent.

        Args:
            model: A white-box model. If it already satisfies the
                :class:`EmbeddingModel` seam (``embed`` / ``loss_from_embeds`` /
                ``token_ids``) it is used directly (the offline test path); a real
                ``transformers`` causal-LM is wrapped in :class:`HFEmbeddingModel`.
            tokenizer: The model's tokenizer (used when wrapping a raw HF model).
            messages: Chat turns; the soft prompt is prepended to the rendered
                last-user prompt's embeddings.
            target: The BENIGN string to emit (the per-run marker). When empty, the
                AdvPrefix benign target is derived (objective stays the canary).
            cfg: An :class:`EmbeddingConfig` (or any :class:`AttackConfig`, coerced
                to embedding defaults plus the shared knobs).
            defense: Optional defense (recorded on the result; adaptive in-loop
                coupling such as LatentBreak-vs-perplexity is a later-chunk
                deliverable, ROADMAP §6.13).

        Returns:
            An :class:`~injectkit.whitebox.base.AttackResult` whose ``optimized_obj``
            is the best :class:`SoftPrompt` (kind ``"embeddings"``), with the
            per-step loss curve and the success flag.
        """
        ecfg = cfg if isinstance(cfg, EmbeddingConfig) else _as_embedding_config(cfg)
        emodel = _as_embedding_model(model, tokenizer)
        prompt = _last_user_content(messages)

        if not target:
            target = advprefix_target(
                getattr(emodel, "name", "") or getattr(model, "name", "") or "",
                trigger=ecfg.trigger,
            )

        prompt_ids = list(emodel.token_ids(prompt))
        target_ids = list(emodel.token_ids(target))

        soft, losses, succeeded = _optimize_soft_prompt(
            emodel, prompt_ids, target_ids, ecfg
        )

        best_loss = min(losses) if losses else float("inf")
        defense_id = getattr(defense, "name", "") if defense is not None else ""

        # The "input actually sent" is the prompt — the soft prompt is a continuous
        # prefix in embedding space with no text form (that is the whole point of a
        # capability-ceiling attack). We surface it via optimized_obj, not best_input.
        return AttackResult(
            attack_name=self.name,
            best_input=prompt,
            best_loss=best_loss,
            per_step_losses=losses,
            optimized_obj=soft,
            optimized_obj_kind="embeddings",
            succeeded=succeeded,
            queries=len(losses),
            defense_id=defense_id,
            stamp={"paper": PAPER_CLAIM, "k": soft.k, "optim": ecfg.optim_method},
        )


# --------------------------------------------------------------------------- #
# Optimisation core — seam-agnostic Adam on the soft-prompt rows.
# --------------------------------------------------------------------------- #


def _optimize_soft_prompt(
    emodel: Any,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    cfg: EmbeddingConfig,
) -> "tuple[SoftPrompt, list[float], bool]":
    """Run the continuous embedding optimisation, returning ``(soft, losses, ok)``.

    Dispatches on whether the seam is torch-backed (real autograd + ``torch.optim``)
    or pure-Python (a from-scratch Adam over the seam's scalar loss). Both paths
    implement the *same* contract: ``k`` trainable soft-prompt rows, Adam (or SGD)
    on the affirmative-target loss, a per-step loss curve, and early-stop on
    convergence. The objective is the benign marker either way.
    """
    if _is_torch_backed(emodel):
        return _optimize_torch(emodel, prompt_ids, target_ids, cfg)
    return _optimize_python(emodel, prompt_ids, target_ids, cfg)


def _is_torch_backed(emodel: Any) -> bool:
    """True iff the seam's ``embed`` returns a torch tensor (real autograd path)."""
    if isinstance(emodel, HFEmbeddingModel):
        return True
    return bool(getattr(emodel, "torch_backed", False))


def _optimize_torch(
    emodel: Any,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    cfg: EmbeddingConfig,
) -> "tuple[SoftPrompt, list[float], bool]":
    """Real-autograd path: ``torch.optim`` Adam/SGD on a ``[k, d]`` leaf tensor.

    Initialises the soft prompt from a small Gaussian (seeded), then for each step
    builds ``[soft ⊕ embed(prompt)]``, calls ``loss_from_embeds`` (differentiable),
    backprops, and steps the optimiser. DEFERRED-NO-GPU only for the *headline ASR/
    wall-clock number* — this code path itself is exercised on a tiny CPU HF model.
    """
    torch = HFEmbeddingModel._torch()
    torch.manual_seed(int(cfg.seed))

    dim = int(emodel.embedding_dim)
    k = int(cfg.k)
    # Seed the soft prompt from the prompt's own embeddings where possible (a warm
    # start the paper uses), else a small Gaussian. Detach so it is a leaf we own.
    with torch.no_grad():
        prompt_embeds = emodel.embed(list(prompt_ids)) if prompt_ids else None
        if prompt_embeds is not None and prompt_embeds.shape[0] >= 1:
            reps = (k + prompt_embeds.shape[0] - 1) // prompt_embeds.shape[0]
            init = prompt_embeds.repeat(reps, 1)[:k].clone()
        else:
            init = torch.randn(k, dim) * 0.01
    soft = init.detach().clone().requires_grad_(True)

    opt = _build_torch_optimizer(torch, [soft], cfg)
    prompt_embeds = emodel.embed(list(prompt_ids)) if prompt_ids else soft.new_zeros(0, dim)

    losses: List[float] = []
    best_vectors = soft.detach().clone()
    best_loss = math.inf
    stalls = 0
    succeeded = False

    for _ in range(cfg.steps()):
        opt.zero_grad()
        input_embeds = torch.cat([soft, prompt_embeds.detach()], dim=0)
        loss = emodel.loss_from_embeds(input_embeds, list(target_ids))
        loss.backward()
        opt.step()

        loss_val = float(loss.detach())
        losses.append(loss_val)
        if loss_val + cfg.convergence_tol < best_loss:
            if best_loss - loss_val < cfg.convergence_tol:
                stalls += 1
            else:
                stalls = 0
            best_loss = loss_val
            best_vectors = soft.detach().clone()
        else:
            stalls += 1
        # A continuous attack "succeeds" when the target NLL is driven near zero
        # (the model would greedily emit the benign target). The threshold mirrors
        # the paper's convergence regime; the real ASR judge runs in the bench layer.
        if loss_val <= _SUCCESS_NLL:
            succeeded = True
            break
        if cfg.convergence_tol > 0.0 and stalls >= cfg.patience:
            break

    soft_prompt = SoftPrompt(best_vectors, k=k, dim=dim)
    return soft_prompt, losses, succeeded


def _build_torch_optimizer(torch: Any, params: list, cfg: EmbeddingConfig) -> Any:
    """Construct the torch optimiser named by ``cfg.optim_method`` (adam|sgd)."""
    method = cfg.optim_method.lower()
    if method == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    return torch.optim.Adam(params, lr=cfg.lr)


def _optimize_python(
    emodel: Any,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    cfg: EmbeddingConfig,
) -> "tuple[SoftPrompt, list[float], bool]":
    """Pure-Python path: a from-scratch Adam on the seam's scalar loss (no torch).

    Used by the offline test seam (and any pure-Python ``EmbeddingModel``). It runs
    the identical optimisation contract — ``k×d`` trainable soft-prompt rows, Adam
    (or SGD) on ``loss_from_embeds``, per-step loss curve, convergence early-stop —
    using the seam's own gradient (``grad_from_embeds`` if provided, else a central
    finite-difference) so the loop genuinely *descends* the seam's loss on CPU.
    """
    rng = _SeededRng(int(cfg.seed))
    dim = int(emodel.embedding_dim)
    k = int(cfg.k)

    # k×d soft-prompt rows, small random init (seeded, reproducible).
    soft = [[rng.gauss() * 0.01 for _ in range(dim)] for _ in range(k)]
    prompt_embeds = list(emodel.embed(list(prompt_ids))) if prompt_ids else []

    adam = _Adam(k, dim, lr=cfg.lr, method=cfg.optim_method)
    losses: List[float] = []
    best = [row[:] for row in soft]
    best_loss = math.inf
    stalls = 0
    succeeded = False

    for _ in range(cfg.steps()):
        loss = float(emodel.loss_from_embeds(soft + prompt_embeds, list(target_ids)))
        grad = _grad_from_seam(emodel, soft, prompt_embeds, list(target_ids), loss)
        adam.step(soft, grad)

        losses.append(loss)
        if loss < best_loss - cfg.convergence_tol:
            best_loss = loss
            best = [row[:] for row in soft]
            stalls = 0
        else:
            stalls += 1
        if loss <= _SUCCESS_NLL:
            succeeded = True
            break
        if cfg.convergence_tol > 0.0 and stalls >= cfg.patience:
            break

    return SoftPrompt(best, k=k, dim=dim), losses, succeeded


#: Affirmative-target NLL at/below which the continuous attack is treated as
#: converged (the soft prompt would greedily elicit the benign target). The real
#: ASR judge runs in the bench layer; this is the optimisation-side success flag.
_SUCCESS_NLL = 0.05


def _grad_from_seam(
    emodel: Any,
    soft: List[List[float]],
    prompt_embeds: list,
    target_ids: Sequence[int],
    loss: float,
) -> List[List[float]]:
    """Gradient of the seam loss w.r.t. the soft-prompt rows (analytic or FD).

    Prefers an analytic ``emodel.grad_from_embeds(input_embeds, target_ids)`` when
    the seam provides one (the test toy model does, for speed + determinism);
    otherwise falls back to a central finite-difference over the ``k×d`` soft rows
    so the loop descends *any* differentiable scalar loss the seam exposes.
    """
    grad_fn = getattr(emodel, "grad_from_embeds", None)
    if callable(grad_fn):
        g = grad_fn(soft + prompt_embeds, list(target_ids))
        # The seam returns grad over all rows; keep only the soft-prompt rows.
        return [list(row) for row in g[: len(soft)]]

    eps = 1e-4
    grad = [[0.0] * len(soft[0]) for _ in range(len(soft))]
    for i in range(len(soft)):
        for j in range(len(soft[0])):
            orig = soft[i][j]
            soft[i][j] = orig + eps
            lp = float(emodel.loss_from_embeds(soft + prompt_embeds, list(target_ids)))
            soft[i][j] = orig - eps
            lm = float(emodel.loss_from_embeds(soft + prompt_embeds, list(target_ids)))
            soft[i][j] = orig
            grad[i][j] = (lp - lm) / (2.0 * eps)
    return grad


class _Adam:
    """A minimal, dependency-free Adam (and SGD) optimiser over a ``k×d`` grid.

    Mirrors ``torch.optim.Adam`` (β1=0.9, β2=0.999, ε=1e-8) so the pure-Python CPU
    path and the torch path implement the *same* update rule; ``method="sgd"``
    selects plain gradient descent. In-place updates ``params``.
    """

    def __init__(
        self,
        k: int,
        dim: int,
        *,
        lr: float,
        method: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.method = method.lower()
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = [[0.0] * dim for _ in range(k)]
        self.v = [[0.0] * dim for _ in range(k)]

    def step(self, params: List[List[float]], grad: List[List[float]]) -> None:
        if self.method == "sgd":
            for i in range(len(params)):
                for j in range(len(params[0])):
                    params[i][j] -= self.lr * grad[i][j]
            return
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for i in range(len(params)):
            for j in range(len(params[0])):
                g = grad[i][j]
                self.m[i][j] = self.b1 * self.m[i][j] + (1 - self.b1) * g
                self.v[i][j] = self.b2 * self.v[i][j] + (1 - self.b2) * g * g
                mhat = self.m[i][j] / bc1
                vhat = self.v[i][j] / bc2
                params[i][j] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


class _SeededRng:
    """A tiny seeded Gaussian source (``random.Random``) for reproducible init."""

    def __init__(self, seed: int) -> None:
        import random

        self._r = random.Random(seed)

    def gauss(self) -> float:
        return self._r.gauss(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Optional ASETF extension — translate soft embeddings to discrete tokens.
# --------------------------------------------------------------------------- #


def asetf_translate(
    soft: SoftPrompt,
    embedding_matrix: Any,
    *,
    metric: str = "cosine",
) -> List[int]:
    """ASETF: map each soft-prompt vector to its nearest discrete token (transfer).

    **ASETF — arXiv:2402.16006** ("ASETF: A Novel Method for Jailbreak Attack on
    LLMs through Translate Suffix Embeddings"). The continuous embedding attack is
    a capability ceiling that does **not** transfer (it lives in embedding space);
    ASETF projects each optimised soft vector onto the nearest row of the model's
    input-embedding matrix to recover a **discrete, transferable token sequence**.

    This re-introduces a round-trip the continuous attack itself avoids — the
    translated tokens' embeddings are only *approximately* the soft vectors — so the
    returned ids are a transfer approximation, not the optimised object.

    Args:
        soft: The optimised :class:`SoftPrompt` (``k`` vectors of width ``d``).
        embedding_matrix: The model's ``[V, d]`` input-embedding matrix — a torch
            tensor (``HFEmbeddingModel.embedding_matrix()``) or a nested list.
        metric: ``"cosine"`` (default; magnitude-invariant) or ``"l2"``.

    Returns:
        ``k`` token ids — the nearest-neighbour discrete translation.
    """
    vectors = soft.tolist()
    matrix = _matrix_to_lists(embedding_matrix)
    out: List[int] = []
    for vec in vectors:
        out.append(_nearest_row(vec, matrix, metric))
    return out


def _nearest_row(vec: Sequence[float], matrix: Sequence[Sequence[float]], metric: str) -> int:
    """Index of the row of ``matrix`` nearest to ``vec`` under ``metric``."""
    best_idx = 0
    best_score = math.inf
    vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
    for idx, row in enumerate(matrix):
        if metric == "l2":
            score = sum((a - b) ** 2 for a, b in zip(vec, row))
        else:  # cosine distance = 1 - cos_sim
            dot = sum(a * b for a, b in zip(vec, row))
            rnorm = math.sqrt(sum(x * x for x in row)) or 1.0
            score = 1.0 - dot / (vnorm * rnorm)
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _matrix_to_lists(matrix: Any) -> List[List[float]]:
    """Coerce a torch tensor / nested sequence embedding matrix to nested lists."""
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return [[float(x) for x in row] for row in matrix]


# --------------------------------------------------------------------------- #
# Adapters + small helpers.
# --------------------------------------------------------------------------- #


def _as_embedding_model(model: Any, tokenizer: Any) -> Any:
    """Return an :class:`EmbeddingModel` seam for ``model``.

    If ``model`` already exposes the seam (``embed`` + ``loss_from_embeds`` +
    ``token_ids`` + ``embedding_dim``) it is used directly — the offline test path
    and any pre-adapted seam. Otherwise it is assumed to be a raw ``transformers``
    causal-LM and wrapped in :class:`HFEmbeddingModel` (lazy torch).
    """
    if _satisfies_embedding_seam(model):
        return model
    return HFEmbeddingModel(model, tokenizer, name=getattr(model, "name", ""))


def _satisfies_embedding_seam(model: Any) -> bool:
    """Duck-type check for the :class:`EmbeddingModel` seam (no torch import)."""
    return all(
        callable(getattr(model, attr, None))
        for attr in ("embed", "loss_from_embeds", "token_ids")
    ) and hasattr(model, "embedding_dim")


def _as_embedding_config(cfg: AttackConfig) -> EmbeddingConfig:
    """Coerce a base :class:`AttackConfig` to an :class:`EmbeddingConfig`.

    Carries the shared knobs (steps/target/trigger/seed) and fills embedding
    defaults for the soft-prompt-specific fields.
    """
    return EmbeddingConfig(
        max_steps=cfg.max_steps,
        target=cfg.target,
        trigger=cfg.trigger,
        seed=cfg.seed,
    )


def _last_user_content(messages: list[dict]) -> str:
    """Content of the last ``user`` turn (or the last turn, or ``""``)."""
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
    """First-class continuous-embedding entrypoint (ROADMAP §6.6).

        from injectkit.whitebox import embedding
        result = embedding.run(model, tok, messages, target, EmbeddingConfig(k=8))

    A thin functional wrapper over :meth:`EmbeddingAttack.run` with a defaulted
    config. Cites arXiv:2402.09063 (Schwinn et al., NeurIPS 2024).
    """
    return EmbeddingAttack().run(
        model, tokenizer, messages, target, cfg or EmbeddingConfig(), defense
    )
