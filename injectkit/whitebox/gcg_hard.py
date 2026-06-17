"""nanoGCG-parity-and-beyond hardening for the white-box GCG optimiser.

CHUNK 3-gcg-advprefix (ROADMAP §6.1). The shipped GCG inner loop
(:class:`injectkit.attackers.gcg.GCGSuffixAttacker`, driven through the offline
``StubWhiteBoxModel`` seam) proves the *contract*. This module adds the
production-grade, nanoGCG-parity machinery that a real GPU run needs, with two
MANDATORY correctness traps the literature shows are the usual silent-failure
sources:

1. **filter_ids retokenization drop** — GCG samples candidate token swaps, but a
   candidate suffix is only valid if it *survives a tokenizer round-trip*:
   ``encode(decode(ids)) == ids``. Candidates that don't (the tokenizer would
   re-segment the text into different ids than the ones we optimised) are
   DROPPED — otherwise the loss we computed is for a token sequence the model
   never actually sees, and the "attack" is an artefact. See :func:`filter_ids`.

2. **tokenizer-agnostic chat-template slice location** — the optim/target spans
   must be located in the *rendered chat prompt* per the model's own template,
   **never** with hard-coded offsets (which silently break across
   Llama-3/Qwen/Gemma/Mistral/Phi). :func:`locate_optim_slice` builds the
   sequence by *concatenating separately-encoded* segments (before / optim /
   after / target), so the optim slice is exact by construction and template-
   independent. See :class:`PromptSlices`.

Plus the nanoGCG knob set: one-hot gradient
(:func:`token_gradients_onehot`), ``top_k`` (default 256) candidate sampling,
``search_width`` (default 512) batch candidate evaluation
(:func:`sample_candidates`), an :class:`AttackBuffer` of the best suffixes, and a
``probe_sampling`` placeholder (documented, off by default).

ETHICS — the optimisation target is ALWAYS the benign per-run canary marker; no
harmful objective is ever set. ``torch`` is lazy-imported so importing this
module (and running the offline slice/round-trip tests) needs neither torch nor a
model download. The tensor ops are exercised on CPU with a tiny model
(GPT-2 / Pythia-160M) for the golden-loss regression; the 8B ASR-parity run is
DEFERRED-NO-GPU (see ``docs/REPRODUCE.md``).

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

__all__ = [
    "PromptSlices",
    "locate_optim_slice",
    "filter_ids",
    "round_trips",
    "token_gradients_onehot",
    "sample_candidates",
    "AttackBuffer",
    "ProbeSamplingConfig",
]


# A control character extremely unlikely to occur in a benign prompt; used as the
# placeholder we substitute the optim string with when rendering the chat
# template, so we can split the rendered string at a known, template-independent
# point. NUL is never produced by a tokenizer's chat template itself.
_OPTIM_PLACEHOLDER = "\x00\x00INJECTKIT_OPTIM\x00\x00"


# --------------------------------------------------------------------------- #
# Correctness trap #2: tokenizer-agnostic chat-template slice location
# --------------------------------------------------------------------------- #


@dataclass
class PromptSlices:
    """The token ids of a GCG prompt, split into its four contiguous segments.

    Built by :func:`locate_optim_slice` by encoding each segment *separately* and
    concatenating, so ``optim_slice`` is exact **by construction** — never derived
    from a hard-coded offset or a fragile re-tokenisation of a joined string. The
    full input the model sees is ``before_ids + optim_ids + after_ids`` (the
    "prompt"); ``target_ids`` is the benign string the loss is computed against.

    Attributes:
        before_ids: Tokens of the rendered prompt *before* the optim span (system
            turn, chat headers, the fixed user text, ...).
        optim_ids: Tokens of the optimisable suffix (what GCG mutates).
        after_ids: Tokens *after* the optim span (turn terminator, the assistant
            generation-prompt header, ...).
        target_ids: Tokens of the BENIGN target string the loss is measured on.
    """

    before_ids: list[int]
    optim_ids: list[int]
    after_ids: list[int]
    target_ids: list[int]

    @property
    def optim_slice(self) -> slice:
        """The slice of the *input* sequence covering the optimisable tokens."""
        start = len(self.before_ids)
        return slice(start, start + len(self.optim_ids))

    @property
    def target_slice(self) -> slice:
        """The slice of the *full* sequence (input+target) covering the target."""
        start = len(self.before_ids) + len(self.optim_ids) + len(self.after_ids)
        return slice(start, start + len(self.target_ids))

    @property
    def input_ids(self) -> list[int]:
        """The prompt the model conditions on: before + optim + after."""
        return [*self.before_ids, *self.optim_ids, *self.after_ids]

    @property
    def full_ids(self) -> list[int]:
        """The full teacher-forced sequence: prompt + target."""
        return [*self.input_ids, *self.target_ids]


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` to ids with NO special tokens added (segment-wise encode).

    Special tokens (BOS/role headers) are part of the *rendered template string*,
    so they are encoded as ordinary text here; adding them again would double them.
    Falls back to a ``token_ids`` seam method (the offline ``StubWhiteBoxModel``)
    when the object isn't a real HF tokenizer.
    """
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        try:
            return list(enc(text, add_special_tokens=False))
        except TypeError:
            # Some tokenizers/seam objects don't accept add_special_tokens.
            return list(enc(text))
    # Offline white-box seam (StubWhiteBoxModel) exposes token_ids().
    seam = getattr(tokenizer, "token_ids", None)
    if callable(seam):
        return list(seam(text))
    raise TypeError(
        f"{tokenizer!r} is not a tokenizer: no .encode or .token_ids method."
    )


def _render_chat(
    tokenizer: Any,
    messages: list[dict],
    *,
    chat_template: Optional[str],
    add_generation_prompt: bool,
) -> str:
    """Render ``messages`` to a prompt *string* via the model's chat template.

    Uses ``tokenizer.apply_chat_template(..., tokenize=False)`` so the result is
    text we can split deterministically. ``chat_template`` overrides the
    tokenizer's own template (used by the offline round-trip test to wear a
    bundled family template on a plain tokenizer). When the tokenizer has no chat
    template at all (a base LM like GPT-2), we fall back to concatenating the
    message contents — the slice logic is identical either way.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    has_tpl = chat_template is not None or getattr(tokenizer, "chat_template", None)
    if callable(apply) and has_tpl:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        return apply(messages, **kwargs)
    # Base LM / seam with no template: degrade to plain content concatenation.
    return "".join(str(m.get("content", "")) for m in messages)


def locate_optim_slice(
    tokenizer: Any,
    messages: list[dict],
    optim_str: str,
    target: str,
    *,
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> PromptSlices:
    """Locate the optim span inside the rendered chat prompt — no hard offsets.

    The tokenizer-agnostic technique (nanoGCG's, generalised across families):

    1. Substitute ``optim_str`` in the *last user turn* with a unique NUL
       placeholder and render the full chat template to a STRING (``tokenize=
       False``) — so the model's real role headers / BOS / generation prompt are
       all present exactly as at inference time.
    2. Split the rendered string at the placeholder into ``before`` / ``after``.
    3. Encode ``before``, ``optim_str``, ``after`` and ``target`` **separately**
       and concatenate the id lists. Because the segments are encoded
       independently, the optim slice is ``[len(before): len(before)+len(optim)]``
       *exactly* — no offset is hard-coded and no fragile re-tokenisation of a
       joined string is needed.

    This is robust across Llama-3 (header-id), Qwen/Phi (ChatML), Gemma
    (start_of_turn) and Mistral ([INST]) because step 1 keys off the model's own
    template, not a per-family constant.

    Args:
        tokenizer: An HF tokenizer (or an offline seam exposing ``token_ids``).
        messages: The chat turns; the optim string is inserted into the last
            ``user`` turn (or the last turn). A turn may already contain
            ``optim_str`` as a literal; otherwise it is appended.
        optim_str: The optimisable suffix string (GCG mutates its tokens).
        target: The BENIGN target string the loss is computed against.
        chat_template: Optional Jinja template override (offline test fixtures).
        add_generation_prompt: Append the assistant generation prompt (the real
            inference condition). Default True.

    Returns:
        A :class:`PromptSlices` with the four segment id lists.
    """
    msgs = _inject_placeholder(messages, optim_str)
    rendered = _render_chat(
        tokenizer,
        msgs,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    if _OPTIM_PLACEHOLDER in rendered:
        idx = rendered.index(_OPTIM_PLACEHOLDER)
        before_text = rendered[:idx]
        after_text = rendered[idx + len(_OPTIM_PLACEHOLDER):]
    else:
        # The template dropped/transformed our placeholder (rare). Degrade to
        # "optim at the very end of the rendered prompt" — still no hard offset.
        before_text, after_text = rendered, ""

    return PromptSlices(
        before_ids=_encode(tokenizer, before_text),
        optim_ids=_encode(tokenizer, optim_str),
        after_ids=_encode(tokenizer, after_text),
        target_ids=_encode(tokenizer, target),
    )


def _inject_placeholder(messages: list[dict], optim_str: str) -> list[dict]:
    """Return a copy of ``messages`` with the optim span marked by a placeholder.

    Replaces a literal occurrence of ``optim_str`` in the last user turn with the
    NUL placeholder; if absent, appends the placeholder to that turn's content.
    The last ``user`` turn is preferred, else the last turn overall.
    """
    out = [dict(m) for m in messages] or [{"role": "user", "content": ""}]
    target_idx = len(out) - 1
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            target_idx = i
            break
    content = str(out[target_idx].get("content", ""))
    if optim_str and optim_str in content:
        content = content.replace(optim_str, _OPTIM_PLACEHOLDER, 1)
    else:
        sep = " " if content and not content.endswith(" ") else ""
        content = f"{content}{sep}{_OPTIM_PLACEHOLDER}"
    out[target_idx] = {**out[target_idx], "content": content}
    return out


# --------------------------------------------------------------------------- #
# Correctness trap #1: filter_ids retokenization drop
# --------------------------------------------------------------------------- #


def round_trips(ids: Sequence[int], tokenizer: Any) -> bool:
    """True iff ``ids`` survive a tokenizer round-trip: ``encode(decode(ids))==ids``.

    The GCG invariant: a candidate suffix's *token ids* must be exactly what the
    tokenizer would produce from the candidate's *text*. If not, the model at
    inference time sees a different token sequence than the one whose loss we
    minimised — the candidate is an optimisation artefact and must be dropped.
    """
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return True  # Can't check (seam without decode) -> don't spuriously drop.
    text = decode(list(ids))
    reencoded = _encode(tokenizer, text)
    return list(reencoded) == list(ids)


def filter_ids(candidate_ids: Any, tokenizer: Any) -> list[int]:
    """Return the row-indices of ``candidate_ids`` that survive retokenisation.

    ``candidate_ids`` is a ``[search_width, optim_len]`` batch of candidate suffix
    token-id rows (a tensor in production, a list-of-lists offline). A row is kept
    iff :func:`round_trips` holds for it. nanoGCG drops the non-round-tripping
    rows entirely (rather than projecting them), because a projected candidate's
    loss no longer corresponds to its text.

    Returns:
        The list of kept row indices (in order). Never empty unless *every* row
        fails — callers treat an empty result as "resample this step".
    """
    rows = _as_rows(candidate_ids)
    return [i for i, row in enumerate(rows) if round_trips(row, tokenizer)]


def _as_rows(candidate_ids: Any) -> list[list[int]]:
    """Coerce a tensor / nested sequence of candidate ids to a list of int rows."""
    tolist = getattr(candidate_ids, "tolist", None)
    if callable(tolist):
        candidate_ids = tolist()
    return [[int(x) for x in row] for row in candidate_ids]


# --------------------------------------------------------------------------- #
# nanoGCG one-hot gradient + candidate sampling
# --------------------------------------------------------------------------- #


def token_gradients_onehot(
    model: Any,
    full_ids: Sequence[int],
    optim_slice: slice,
    target_slice: slice,
) -> Any:
    """nanoGCG one-hot gradient of the target NLL w.r.t. the optim tokens.

    Builds a ``[optim_len, vocab]`` one-hot matrix ``X`` with ``requires_grad``,
    forms the input embeddings as ``X @ E`` (``E`` = the embedding matrix) spliced
    into the model's embedded sequence at ``optim_slice``, runs a forward pass,
    computes the cross-entropy of the ``target_slice`` logits against the target
    ids, and back-props to ``X.grad``. ``X.grad[i]`` is then the per-vocab
    gradient for optim slot ``i`` — its most-negative entries are GCG's candidate
    replacement tokens.

    ``torch`` is imported lazily here (never at module import). Requires a real HF
    model exposing ``get_input_embeddings()`` and a logits-returning ``forward``;
    this is the GPU/CPU production path. The offline ``StubWhiteBoxModel`` instead
    uses its own ``token_gradients`` seam (the v0.3 loop), so this function is only
    reached with a real model (CPU tiny-model golden-loss test, or GPU).

    Returns:
        The ``[optim_len, vocab]`` gradient tensor (on the model's device/dtype).
    """
    import torch  # lazy: keeps module import torch-free

    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight  # [vocab, d_model]
    device = embed_weights.device
    vocab = embed_weights.shape[0]

    ids = torch.tensor([list(full_ids)], device=device)  # [1, seq]
    optim_ids = ids[0, optim_slice]

    one_hot = torch.zeros(
        optim_ids.shape[0], vocab, device=device, dtype=embed_weights.dtype
    )
    one_hot.scatter_(1, optim_ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)

    optim_embeds = one_hot @ embed_weights  # [optim_len, d_model]

    base_embeds = embed_layer(ids).detach()  # [1, seq, d_model]
    full_embeds = torch.cat(
        [
            base_embeds[:, : optim_slice.start, :],
            optim_embeds.unsqueeze(0),
            base_embeds[:, optim_slice.stop :, :],
        ],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits  # [1, seq, vocab]

    # Shift: logits at position t predict token t+1, so the target logits start
    # one position before target_slice.start (teacher forcing).
    shift = target_slice.start - 1
    pred = logits[0, shift : target_slice.stop - 1, :]
    tgt = ids[0, target_slice]
    loss = torch.nn.functional.cross_entropy(pred, tgt)

    (grad,) = torch.autograd.grad(loss, [one_hot])
    return grad  # [optim_len, vocab]


def sample_candidates(
    grad: Any,
    optim_ids: Sequence[int],
    *,
    top_k: int = 256,
    search_width: int = 512,
    seed: int = 0,
    not_allowed_ids: Optional[Sequence[int]] = None,
) -> Any:
    """Sample a ``[search_width, optim_len]`` batch of single-token-swap candidates.

    nanoGCG's GCGSampler: for each candidate, pick one optim slot uniformly and
    replace its token with one of that slot's ``top_k`` most-negative-gradient
    ids. Returns a batch tensor (production) so the model can score all
    ``search_width`` candidates in one forward pass.

    ``torch`` is lazy-imported. With a list-of-lists ``grad`` (offline) this still
    works and returns a list-of-lists, so the sampler is unit-testable without
    torch. ``not_allowed_ids`` (e.g. non-ASCII / special tokens) are masked out of
    the top-k before sampling.

    Args:
        grad: ``[optim_len, vocab]`` gradient (tensor or nested list).
        optim_ids: Current optim token ids (length ``optim_len``).
        top_k: Per-slot candidate pool size (nanoGCG default 256).
        search_width: Number of candidates to draw (nanoGCG default 512).
        seed: RNG seed for reproducibility.
        not_allowed_ids: Vocab ids forbidden as replacements (masked from top-k).

    Returns:
        A ``[search_width, optim_len]`` candidate batch (tensor or nested list).
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None and hasattr(grad, "topk"):
        return _sample_candidates_torch(
            torch, grad, optim_ids, top_k, search_width, seed, not_allowed_ids
        )
    return _sample_candidates_py(
        grad, optim_ids, top_k, search_width, seed, not_allowed_ids
    )


def _topk_indices_py(row: Sequence[float], k: int, banned: set[int]) -> list[int]:
    """Indices of the ``k`` smallest (most-negative-gradient) entries, banned-masked."""
    order = sorted(range(len(row)), key=lambda i: row[i])
    out = [i for i in order if i not in banned]
    return out[: max(1, k)]


def _sample_candidates_py(
    grad: Any,
    optim_ids: Sequence[int],
    top_k: int,
    search_width: int,
    seed: int,
    not_allowed_ids: Optional[Sequence[int]],
) -> list[list[int]]:
    """Pure-Python sampler (offline/test path; deterministic given ``seed``)."""
    import random

    rng = random.Random(seed)
    banned = set(int(x) for x in (not_allowed_ids or ()))
    rows = [list(r) for r in grad]
    optim = [int(x) for x in optim_ids]
    n = len(optim)
    topk_per_slot = [_topk_indices_py(rows[s], top_k, banned) for s in range(n)]

    candidates: list[list[int]] = []
    for _ in range(search_width):
        slot = rng.randrange(n) if n else 0
        pool = topk_per_slot[slot] if n else []
        new = list(optim)
        if pool:
            new[slot] = pool[rng.randrange(len(pool))]
        candidates.append(new)
    return candidates


def _sample_candidates_torch(
    torch: Any,
    grad: Any,
    optim_ids: Sequence[int],
    top_k: int,
    search_width: int,
    seed: int,
    not_allowed_ids: Optional[Sequence[int]],
) -> Any:
    """Vectorised torch sampler (production path)."""
    g = grad.clone()
    if not_allowed_ids:
        g[:, list(not_allowed_ids)] = float("inf")
    n = g.shape[0]
    # Most-negative gradient = most-promising replacement -> smallest values.
    topk = (-g).topk(min(top_k, g.shape[1]), dim=1).indices  # [optim_len, top_k]

    gen = torch.Generator(device=g.device).manual_seed(seed)
    base = torch.tensor(list(optim_ids), device=g.device).repeat(search_width, 1)
    slots = torch.randint(0, max(1, n), (search_width,), generator=gen, device=g.device)
    picks = torch.randint(
        0, topk.shape[1], (search_width,), generator=gen, device=g.device
    )
    new_tokens = topk[slots, picks]
    base[torch.arange(search_width, device=g.device), slots] = new_tokens
    return base


# --------------------------------------------------------------------------- #
# Attack buffer
# --------------------------------------------------------------------------- #


@dataclass(order=True)
class _BufferItem:
    loss: float
    ids: list[int] = field(compare=False)


class AttackBuffer:
    """nanoGCG attack buffer: the ``size`` lowest-loss optim suffixes seen so far.

    GCG with ``buffer_size > 0`` keeps a small pool of the best candidates and, at
    each step, optimises from one of them (rather than always the single best),
    which empirically escapes plateaus. ``add`` keeps the buffer sorted ascending
    by loss and capped at ``size``; :meth:`best` returns the lowest-loss ids.

    A ``size`` of 0 still keeps the single best (so the caller always has a result).
    """

    def __init__(self, size: int = 0) -> None:
        self.size = max(0, int(size))
        self._items: list[_BufferItem] = []

    def add(self, loss: float, optim_ids: Sequence[int]) -> None:
        """Insert a candidate, keeping the buffer sorted and capped."""
        item = _BufferItem(float(loss), [int(x) for x in optim_ids])
        self._items.append(item)
        self._items.sort()
        cap = self.size if self.size > 0 else 1
        del self._items[cap:]

    def best(self) -> Optional[list[int]]:
        """The lowest-loss optim ids, or ``None`` if the buffer is empty."""
        return list(self._items[0].ids) if self._items else None

    def best_loss(self) -> float:
        """The lowest loss in the buffer (``inf`` if empty)."""
        return self._items[0].loss if self._items else float("inf")

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# probe_sampling placeholder (nanoGCG knob parity)
# --------------------------------------------------------------------------- #


@dataclass
class ProbeSamplingConfig:
    """Placeholder config for nanoGCG ``probe_sampling`` (knob-parity, off default).

    Probe sampling (nanoGCG / "Accelerating Greedy Coordinate Gradient", arXiv:
    2410.15362) uses a cheap *draft* model to pre-filter the ``search_width``
    candidate batch down to a small probe set before scoring with the expensive
    target model, cutting forward passes. injectkit carries the knob for parity
    and records it in the stamp; the draft-model filtering loop is a later-chunk
    GPU deliverable and is DEFERRED-NO-GPU here. With ``enabled=False`` (default)
    the optimiser scores the full batch exactly as plain GCG.

    Attributes:
        enabled: Turn probe sampling on (requires a draft model; not run on CPU).
        draft_model: Friendly zoo name of the cheap draft model (when enabled).
        sampling_factor: Candidate-batch reduction factor R (nanoGCG default 16).
    """

    enabled: bool = False
    draft_model: Optional[str] = None
    sampling_factor: int = 16
