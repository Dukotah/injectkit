"""The v0.4 generation runner — deterministic, backend-LOCKED greedy decoding.

CHUNK 6-generation-runner (ROADMAP §3.2, §8). One function, :func:`generate`, is
the single seam every white-box attack and judge generates through. It is
**greedy and deterministic** (``temperature=0.0``, ``top_p=1.0``, fixed seed) so a
run is reproducible token-for-token, and it is **backend-LOCKED**: the backend the
text was generated under (``hf`` | ``vllm``) is recorded in the result stamp, and
the same-backend invariant (:func:`assert_same_backend`) refuses to let a judge
score generations produced under a *different* backend.

Why the backend lock exists (ROADMAP §3.2)
-----------------------------------------
HF and vLLM do not produce identical logprobs for the same model + prompt
(different kernels, different numerical paths, KV-cache quantisation, batching).
A judge that consumes logprobs — or that *re-scores* a continuation under its own
backend — therefore gives subtly different numbers depending on which engine
produced the text. Mixing an HF-generated attack with a vLLM judge (or vice-versa)
silently corrupts the leaderboard. The invariant makes that mismatch a loud
warning/error instead of a quiet divergence, and the backend is auto-stamped so
the provenance is never lost (auto-stamp if vLLM).

``temperature`` / ``top_p`` are **hard-set to greedy**, not leaderboard knobs: the
:class:`GenerationConfig` exposes them for documentation/provenance, but
:func:`generate` ignores any non-greedy value and always decodes greedily so two
runs of the same (model, prompt, seed) are byte-identical.

Backends
--------
* ``hf`` — in-process HuggingFace ``transformers`` causal-LM, ``model.generate(
  ..., do_sample=False)``. The only backend with a backward pass, so the only one
  the white-box optimisers can use; verified on a tiny CPU model (GPT-2).
* ``vllm`` — an out-of-process vLLM engine (an object exposing ``.generate``). The
  high-throughput EVAL backend. Loading a real vLLM engine needs a GPU + a
  multi-GB model (DEFERRED-NO-GPU); the runner's vLLM code path + auto-stamp +
  same-backend invariant are exercised against an offline stub.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from .runner import (
    BACKEND_HF,
    BACKEND_VLLM,
    VALID_BACKENDS,
    BackendMismatchError,
    GenerationConfig,
    GenerationOutput,
    assert_same_backend,
    backend_of,
    generate,
)

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
