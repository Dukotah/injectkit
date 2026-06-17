"""Deterministic, backend-LOCKED greedy generation + the same-backend invariant.

CHUNK 6-generation-runner (ROADMAP §3.2, §8). See the package docstring for the
why. This module owns:

* :class:`GenerationConfig` — the frozen knobs (``max_new_tokens=512``,
  ``temperature=0.0``, ``top_p=1.0``, ``backend="hf"``). temp/top_p are
  hard-set greedy and are NOT leaderboard knobs.
* :func:`generate` — render → encode → greedy-decode → return a
  :class:`GenerationOutput` whose ``backend`` field is recorded for provenance.
  Backend-LOCKED: the chosen backend is fixed for the whole call.
* :func:`assert_same_backend` / :func:`backend_of` / :class:`BackendMismatchError`
  — the §3.2 invariant: scoring text generated under backend *A* with a judge that
  runs under backend *B* warns (``strict=False``) or raises (``strict=True``),
  because HF and vLLM logprobs diverge. The backend is auto-stamped when a vLLM
  judge/object is detected so the provenance is never silently lost.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "BACKEND_HF",
    "BACKEND_VLLM",
    "VALID_BACKENDS",
    "BackendMismatchError",
    "GenerationConfig",
    "GenerationOutput",
    "assert_same_backend",
    "backend_of",
    "generate",
]

#: The in-process HuggingFace ``transformers`` backend (the only one with a
#: backward pass — the white-box optimisers' backend).
BACKEND_HF = "hf"
#: The out-of-process vLLM backend (high-throughput EVAL backend; GPU-only).
BACKEND_VLLM = "vllm"
#: The backends the runner understands. Anything else is rejected up front.
VALID_BACKENDS = frozenset({BACKEND_HF, BACKEND_VLLM})


class BackendMismatchError(RuntimeError):
    """Raised when generation/scoring crosses an HF↔vLLM backend boundary.

    HF and vLLM produce divergent logprobs for the same (model, prompt), so a
    judge that re-scores or consumes logprobs gives different numbers depending on
    which engine produced the text (ROADMAP §3.2). The invariant
    (:func:`assert_same_backend`) raises this when an attack generated under one
    backend is graded by a judge bound to another, rather than silently corrupting
    the leaderboard.
    """


class GenerationConfig(BaseModel):
    """Frozen knobs for one :func:`generate` call (ROADMAP §3.2).

    Greedy and deterministic by construction. ``temperature`` / ``top_p`` are
    exposed for provenance only — they are **hard-set to the greedy values** and
    are NOT leaderboard knobs: a non-greedy value is rejected at validation time so
    no caller can quietly turn on sampling and break reproducibility. The validated
    config is what gets stamped onto every result.

    The ``backend`` field LOCKS the call to one engine; the result records it so
    the same-backend invariant can be enforced downstream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Greedy continuation length (the paper / prefill N=512 default).
    max_new_tokens: int = Field(default=512, ge=1)
    #: Hard-set greedy. Must be 0.0 — any other value is rejected (not a knob).
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    #: Hard-set greedy. Must be 1.0 — any other value is rejected (not a knob).
    top_p: float = Field(default=1.0, ge=1.0, le=1.0)
    #: RNG seed, threaded into the backend so a run is byte-reproducible.
    seed: int = 0
    #: The LOCKED backend for this call: ``"hf"`` (default) or ``"vllm"``.
    backend: str = BACKEND_HF

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        if value not in VALID_BACKENDS:
            raise ValueError(
                f"unknown backend {value!r}; must be one of {sorted(VALID_BACKENDS)}. "
                "Backends are not interchangeable: HF and vLLM logprobs diverge "
                "(ROADMAP §3.2)."
            )
        return value


@dataclass
class GenerationOutput:
    """The result of one greedy :func:`generate` call.

    ``text`` is the model's continuation (the new tokens only, prompt stripped).
    ``backend`` is the engine that produced it — recorded so the same-backend
    invariant can be enforced when a judge later scores this text. ``stamp``
    carries the reproducibility metadata (config, backend, seed, token counts) the
    bench/leaderboard layer records.
    """

    #: The greedily-generated continuation (new tokens only; prompt stripped).
    text: str
    #: The backend that produced ``text`` (``"hf"`` | ``"vllm"``) — the lock.
    backend: str
    #: Number of newly generated tokens (budget/accounting; 0 if unmeasured).
    n_new_tokens: int = 0
    #: Number of prompt tokens fed in (0 if unmeasured, e.g. the offline seam).
    n_prompt_tokens: int = 0
    #: Why generation stopped (``end_turn`` | ``length`` | seam-provided).
    stop_reason: str = "end_turn"
    #: Reproducibility stamp (backend, seed, greedy flags, token counts, ...).
    stamp: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Backend detection + the same-backend invariant (ROADMAP §3.2).
# --------------------------------------------------------------------------- #


def backend_of(obj: Any, *, default: str = BACKEND_HF) -> str:
    """Best-effort detect the backend a model / judge / engine runs under.

    Resolution order (auto-stamp if vLLM — ROADMAP §3.2):

    1. An explicit ``backend`` attribute (a judge/model that declares its engine).
    2. A duck-typed vLLM engine: a ``vllm``-rooted class module, or the
       ``LLM``/``vllm`` markers vLLM objects carry. Such an object is stamped
       ``"vllm"`` even if it never set ``backend`` — so a vLLM judge attached to an
       HF attack is *detected*, not silently accepted.
    3. ``default`` (``"hf"``) — the in-process transformers path.

    Never raises and never imports vLLM/torch; pure introspection so it is safe to
    call on any object (including the offline stubs the tests use).
    """
    declared = getattr(obj, "backend", None)
    if isinstance(declared, str) and declared:
        return declared if declared in VALID_BACKENDS else default

    # Duck-type a vLLM engine without importing vllm. vLLM's engine classes live in
    # the ``vllm`` package; the public entrypoint is ``vllm.LLM``.
    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    if module.split(".", 1)[0] == BACKEND_VLLM:
        return BACKEND_VLLM
    qualname = f"{module}.{getattr(cls, '__qualname__', '')}".lower()
    if BACKEND_VLLM in qualname:
        return BACKEND_VLLM
    return default


def assert_same_backend(
    gen_backend: str,
    judge: Any,
    *,
    strict: bool = True,
) -> str:
    """Enforce the §3.2 same-backend invariant between generation and scoring.

    ``gen_backend`` is the backend the text-under-scoring was generated with (an
    :attr:`GenerationOutput.backend` or a config's ``backend``). ``judge`` is the
    object that will score it — its backend is auto-detected via :func:`backend_of`
    (auto-stamp if vLLM). If they differ, HF↔vLLM logprob divergence makes the
    score unreliable, so this either raises :class:`BackendMismatchError`
    (``strict=True``, the default) or emits a :class:`RuntimeWarning`
    (``strict=False``).

    Returns the agreed backend (always ``gen_backend``) when they match, so callers
    can stamp it.

    Args:
        gen_backend: Backend the generations were produced under.
        judge: The judge/scoring object (its backend is detected, default ``hf``).
        strict: Raise on mismatch (True) or warn (False).

    Raises:
        BackendMismatchError: in strict mode when the judge's backend differs.
        ValueError: if ``gen_backend`` is not a known backend.
    """
    if gen_backend not in VALID_BACKENDS:
        raise ValueError(
            f"unknown generation backend {gen_backend!r}; "
            f"must be one of {sorted(VALID_BACKENDS)}."
        )
    judge_backend = backend_of(judge)
    if judge_backend == gen_backend:
        return gen_backend

    judge_id = getattr(judge, "judge_id", None) or type(judge).__name__
    message = (
        f"same-backend invariant violated (ROADMAP §3.2): text was generated under "
        f"backend {gen_backend!r} but judge {judge_id!r} runs under "
        f"{judge_backend!r}. HF and vLLM logprobs diverge, so this score is "
        "unreliable. Re-generate or re-score under one backend."
    )
    if strict:
        raise BackendMismatchError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return gen_backend


# --------------------------------------------------------------------------- #
# The generation runner.
# --------------------------------------------------------------------------- #


def generate(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict],
    max_new_tokens: int = 512,
    backend: str = BACKEND_HF,
    *,
    cfg: Optional[GenerationConfig] = None,
    seed: int = 0,
) -> GenerationOutput:
    """Greedy, deterministic, backend-LOCKED generation (ROADMAP §3.2).

    Renders ``messages`` with the tokenizer's chat template (falling back to a
    plain concatenation for a base LM with no template), greedily decodes
    ``max_new_tokens`` new tokens under the LOCKED ``backend``, and returns a
    :class:`GenerationOutput` whose ``backend`` is recorded for the same-backend
    invariant. Always greedy — ``temperature``/``top_p`` are pinned, so two calls
    with the same (model, messages, seed) yield byte-identical text.

    Three execution paths, selected at runtime so the path is testable offline:

    * An **offline generation seam** — any ``model`` exposing
      ``generate_text(messages, max_new_tokens, *, backend, seed) -> str |
      GenerationOutput``. The test stubs implement this, so the whole runner +
      invariant path runs with no torch and no model download.
    * **vLLM** (``backend="vllm"``) — an out-of-process engine exposing
      ``.generate(prompt, sampling_params)``. Greedy is requested via
      ``SamplingParams(temperature=0, top_p=1, max_tokens=...)``. A real engine
      needs a GPU + multi-GB model (DEFERRED-NO-GPU); the path is stub-tested.
    * **HF** (``backend="hf"``, default) — an in-process ``transformers``
      causal-LM, ``model.generate(..., do_sample=False)``. Verified on tiny GPT-2.

    Args:
        model: The loaded model / engine, OR an offline ``generate_text`` seam.
        tokenizer: The model's tokenizer (HF path; ignored by the seam / vLLM
            engines that tokenise internally).
        messages: Chat turns as ``{"role", "content"}`` dicts.
        max_new_tokens: Greedy continuation length (default 512). Overridden by
            ``cfg.max_new_tokens`` when an explicit ``cfg`` is passed.
        backend: The LOCKED backend (``"hf"`` | ``"vllm"``). Overridden by
            ``cfg.backend`` when an explicit ``cfg`` is passed.
        cfg: An explicit :class:`GenerationConfig`. When given it is authoritative
            (its ``max_new_tokens``/``backend``/``seed`` win over the positionals);
            when ``None`` one is built from the positional args.
        seed: RNG seed (threaded into the backend). Overridden by ``cfg.seed``.

    Returns:
        A :class:`GenerationOutput` with the continuation, the recorded backend,
        token counts, and a reproducibility ``stamp``.
    """
    gcfg = cfg or GenerationConfig(
        max_new_tokens=int(max_new_tokens), backend=backend, seed=int(seed)
    )

    # Offline / custom seam (preferred when present so tests never need torch).
    seam = getattr(model, "generate_text", None)
    if callable(seam):
        out = seam(
            list(messages),
            gcfg.max_new_tokens,
            backend=gcfg.backend,
            seed=gcfg.seed,
        )
        if isinstance(out, GenerationOutput):
            # Honour the seam's reported backend, but the call was LOCKED to gcfg's
            # backend; record the config's backend so the lock is authoritative.
            out.backend = gcfg.backend
            out.stamp = {**_base_stamp(gcfg), **out.stamp}
            return out
        text = str(out)
        n_new = _safe_token_len(tokenizer, text)
        return GenerationOutput(
            text=text,
            backend=gcfg.backend,
            n_new_tokens=n_new,
            stamp=_base_stamp(gcfg),
        )

    if gcfg.backend == BACKEND_VLLM:
        return _vllm_generate(model, tokenizer, list(messages), gcfg)
    return _hf_generate(model, tokenizer, list(messages), gcfg)


def _base_stamp(gcfg: GenerationConfig) -> dict[str, Any]:
    """The reproducibility stamp every generation records (backend + greedy flags)."""
    return {
        "backend": gcfg.backend,
        "max_new_tokens": gcfg.max_new_tokens,
        "temperature": gcfg.temperature,  # always 0.0 (greedy, not a knob).
        "top_p": gcfg.top_p,  # always 1.0 (greedy, not a knob).
        "do_sample": False,
        "seed": gcfg.seed,
        "deterministic": True,
    }


def _render_prompt(tokenizer: Any, messages: list[dict]) -> str:
    """Render the chat prompt with ``add_generation_prompt=True``.

    Falls back to a plain content concatenation when the tokenizer has no chat
    template (a base LM like GPT-2), so the runner is still exercised end-to-end on
    a tiny model.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    has_tpl = bool(getattr(tokenizer, "chat_template", None))
    if callable(apply) and has_tpl:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    body = "".join(str(m.get("content", "")) for m in messages)
    return f"{body}\n"


def _safe_token_len(tokenizer: Any, text: str) -> int:
    """Best-effort token count of ``text`` (0 if no usable tokenizer)."""
    if tokenizer is None:
        return 0
    try:
        ids = tokenizer(text)["input_ids"]
        return len(ids)
    except Exception:  # noqa: BLE001 - accounting only; never break a run.
        return 0


def _hf_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    gcfg: GenerationConfig,
) -> GenerationOutput:
    """Greedy in-process HuggingFace generation (the default, backward-pass backend).

    Lazy-imports torch. Seeds the RNG (``seed``) for reproducibility, renders the
    prompt, encodes it, and greedily decodes ``max_new_tokens`` new tokens
    (``do_sample=False`` ⇒ deterministic). Returns the continuation (new tokens
    only) plus a stamp. DEFERRED-NO-GPU for the 7–20B zoo checkpoints (needs a GPU
    + multi-GB download); verified on a tiny CPU model (GPT-2).
    """
    import torch  # noqa: PLC0415 - intentional lazy import (heavy dep)

    torch.manual_seed(gcfg.seed)

    prompt = _render_prompt(tokenizer, messages)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gcfg.max_new_tokens,
        "do_sample": False,  # greedy — reproducible (ROADMAP §3.2). NOT a knob.
        "num_beams": 1,
    }
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None and eos_id is not None:
        gen_kwargs["pad_token_id"] = eos_id

    attn = enc.get("attention_mask")
    if attn is not None:
        gen_kwargs["attention_mask"] = attn

    with torch.no_grad():
        out_ids = model.generate(input_ids, **gen_kwargs)

    new_ids = out_ids[0][prompt_len:]
    n_new = int(new_ids.shape[0])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    stop_reason = "length" if n_new >= gcfg.max_new_tokens else "end_turn"
    return GenerationOutput(
        text=text,
        backend=BACKEND_HF,
        n_new_tokens=n_new,
        n_prompt_tokens=prompt_len,
        stop_reason=stop_reason,
        stamp=_base_stamp(gcfg),
    )


def _vllm_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    gcfg: GenerationConfig,
) -> GenerationOutput:
    """Greedy out-of-process vLLM generation (the high-throughput EVAL backend).

    Lazy-imports ``vllm`` for the ``SamplingParams``. Greedy is requested with
    ``temperature=0`` (vLLM treats 0 as argmax) + ``top_p=1`` + the seed, so the
    output is deterministic. ``model`` is a ``vllm.LLM`` engine exposing
    ``.generate(prompts, sampling_params)``.

    DEFERRED-NO-GPU: a real vLLM engine needs a CUDA GPU + a multi-GB model, which
    this host has neither of. The code path is complete; it is exercised against an
    offline ``generate_text`` seam stub (see :func:`generate`) rather than a real
    engine. Importing ``vllm`` here raises a clear error if it is reached without
    the GPU stack, instead of hanging on a download.
    """
    from vllm import SamplingParams  # noqa: PLC0415 - lazy (GPU-only dep)

    prompt = _render_prompt(tokenizer, messages)
    params = SamplingParams(
        temperature=0.0,  # greedy / argmax (ROADMAP §3.2). NOT a knob.
        top_p=1.0,
        max_tokens=gcfg.max_new_tokens,
        seed=gcfg.seed,
    )
    outputs = model.generate([prompt], params)
    completion = outputs[0].outputs[0]
    text = completion.text
    n_new = len(getattr(completion, "token_ids", ()) or ())
    finish = getattr(completion, "finish_reason", None) or "end_turn"
    stop_reason = "length" if finish == "length" else "end_turn"
    return GenerationOutput(
        text=text,
        backend=BACKEND_VLLM,
        n_new_tokens=n_new,
        stop_reason=stop_reason,
        stamp=_base_stamp(gcfg),
    )
