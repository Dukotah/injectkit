"""Tests for the generation runner + same-backend invariant (CHUNK 6).

Covers ``injectkit.generate.runner`` (ROADMAP §3.2): the deterministic, greedy,
backend-LOCKED :func:`generate`, the :class:`GenerationConfig` (greedy hard-set,
temp/top_p NOT knobs), and the same-backend invariant — an HF-attack generation
scored by a vLLM judge must warn/raise, and the backend is auto-stamped (vLLM).

Most tests drive an OFFLINE ``generate_text`` seam so the whole path runs with NO
torch and NO model download. One test exercises the REAL HF greedy path on a TINY
CPU model (GPT-2), skipping if torch/transformers/network are unavailable; it never
hangs and never touches a large model. The vLLM real-engine load is DEFERRED-NO-GPU
(no CUDA + multi-GB download here) — the vLLM code path is covered via a stub +
the invariant.
"""

from __future__ import annotations

import warnings

import pytest

from injectkit.generate import (
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


# --------------------------------------------------------------------------- #
# Offline generation seams + stub judges.
# --------------------------------------------------------------------------- #


class _Seam:
    """Offline generation seam: records calls, returns a scripted continuation."""

    def __init__(self, name="llama-3.1-8b", emit=" the answer"):
        self.name = name
        self.emit = emit
        self.calls: list[dict] = []

    def generate_text(self, messages, max_new_tokens, *, backend, seed):
        self.calls.append(
            {"max_new_tokens": max_new_tokens, "backend": backend, "seed": seed}
        )
        return self.emit


class _RichSeam:
    """Seam returning a full GenerationOutput (token counts + stop reason)."""

    def generate_text(self, messages, max_new_tokens, *, backend, seed):
        return GenerationOutput(
            text=" rich continuation",
            backend="hf",  # deliberately set; the runner LOCKS to the config.
            n_new_tokens=3,
            n_prompt_tokens=7,
            stop_reason="length",
            stamp={"seam": True},
        )


class _HFJudge:
    judge_id = "clean_cls"
    backend = "hf"


class _VLLMJudge:
    judge_id = "clean_cls_vllm"
    backend = "vllm"


class _UndeclaredJudge:
    """A judge with no backend attribute — defaults to hf."""

    judge_id = "substring"


# A duck-typed vLLM engine: a class whose module looks like it lives in ``vllm``.
class _FakeVLLMEngine:
    pass


_FakeVLLMEngine.__module__ = "vllm.engine.fake"


_MSGS = [{"role": "user", "content": "hello"}]


# --------------------------------------------------------------------------- #
# GenerationConfig — greedy hard-set, temp/top_p not knobs.
# --------------------------------------------------------------------------- #


def test_config_defaults_are_greedy_and_512():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 512
    assert cfg.temperature == 0.0
    assert cfg.top_p == 1.0
    assert cfg.backend == BACKEND_HF
    assert cfg.seed == 0


def test_config_is_frozen():
    cfg = GenerationConfig()
    with pytest.raises(Exception):
        cfg.max_new_tokens = 8


def test_config_rejects_non_greedy_temp_and_top_p():
    # temp/top_p are HARD-SET greedy, not leaderboard knobs.
    with pytest.raises(Exception):
        GenerationConfig(temperature=0.7)
    with pytest.raises(Exception):
        GenerationConfig(top_p=0.9)


def test_config_rejects_unknown_backend():
    with pytest.raises(Exception):
        GenerationConfig(backend="sglang")


def test_config_rejects_extra_and_bad_bounds():
    with pytest.raises(Exception):
        GenerationConfig(max_new_tokens=0)  # ge=1.
    with pytest.raises(Exception):
        GenerationConfig(bogus=1)  # type: ignore[call-arg]  # extra=forbid.


# --------------------------------------------------------------------------- #
# generate() — offline seam, backend recorded in stamp.
# --------------------------------------------------------------------------- #


def test_generate_offline_seam_records_backend_in_stamp():
    seam = _Seam()
    out = generate(seam, None, _MSGS, max_new_tokens=16, backend="hf")
    assert isinstance(out, GenerationOutput)
    assert out.text == " the answer"
    assert out.backend == "hf"
    # Backend recorded in the reproducibility stamp (the chunk done-check).
    assert out.stamp["backend"] == "hf"
    assert out.stamp["max_new_tokens"] == 16
    assert out.stamp["do_sample"] is False
    assert out.stamp["temperature"] == 0.0
    assert out.stamp["top_p"] == 1.0
    assert out.stamp["deterministic"] is True
    # The seam saw the greedy N and the locked backend.
    assert seam.calls[0]["max_new_tokens"] == 16
    assert seam.calls[0]["backend"] == "hf"


def test_generate_default_n_is_512():
    seam = _Seam()
    generate(seam, None, _MSGS)
    assert seam.calls[0]["max_new_tokens"] == 512  # paper / prefill N=512.


def test_generate_vllm_backend_recorded_and_auto_stamped():
    seam = _Seam()
    out = generate(seam, None, _MSGS, max_new_tokens=8, backend="vllm")
    assert out.backend == "vllm"
    assert out.stamp["backend"] == "vllm"  # auto-stamp if vLLM.
    assert seam.calls[0]["backend"] == "vllm"


def test_generate_rich_output_backend_locked_to_config():
    # The seam reports its own backend, but the call is LOCKED to the config's.
    out = generate(_RichSeam(), None, _MSGS, max_new_tokens=8, backend="vllm")
    assert out.backend == "vllm"  # config wins over the seam's self-report.
    assert out.stamp["backend"] == "vllm"
    assert out.stamp["seam"] is True  # seam's own stamp preserved.
    assert out.n_new_tokens == 3
    assert out.stop_reason == "length"


def test_generate_explicit_cfg_wins_over_positionals():
    seam = _Seam()
    cfg = GenerationConfig(max_new_tokens=4, backend="vllm", seed=99)
    generate(seam, None, _MSGS, max_new_tokens=999, backend="hf", cfg=cfg)
    assert seam.calls[0]["max_new_tokens"] == 4
    assert seam.calls[0]["backend"] == "vllm"
    assert seam.calls[0]["seed"] == 99


def test_generate_seed_threaded_to_seam():
    seam = _Seam()
    generate(seam, None, _MSGS, seed=7)
    assert seam.calls[0]["seed"] == 7


# --------------------------------------------------------------------------- #
# backend_of — detection / auto-stamp.
# --------------------------------------------------------------------------- #


def test_backend_of_declared_attribute():
    assert backend_of(_HFJudge()) == "hf"
    assert backend_of(_VLLMJudge()) == "vllm"


def test_backend_of_defaults_to_hf():
    assert backend_of(_UndeclaredJudge()) == "hf"
    assert backend_of(object()) == "hf"


def test_backend_of_duck_types_vllm_engine():
    # No backend attribute set, but the class module roots at ``vllm`` ⇒ vllm.
    assert backend_of(_FakeVLLMEngine()) == "vllm"


def test_backend_of_custom_default():
    assert backend_of(object(), default="vllm") == "vllm"


# --------------------------------------------------------------------------- #
# THE same-backend invariant (ROADMAP §3.2) — the headline done-check.
# --------------------------------------------------------------------------- #


def test_same_backend_invariant():
    """HF-attack generation + vLLM judge must RAISE (the chunk's named test)."""
    # An attack generated under HF...
    out = generate(_Seam(), None, _MSGS, max_new_tokens=8, backend="hf")
    assert out.backend == "hf"
    # ...scored by a vLLM judge must raise (logprobs diverge; §3.2).
    with pytest.raises(BackendMismatchError):
        assert_same_backend(out.backend, _VLLMJudge(), strict=True)


def test_same_backend_invariant_vllm_attack_hf_judge_raises():
    out = generate(_Seam(), None, _MSGS, max_new_tokens=8, backend="vllm")
    with pytest.raises(BackendMismatchError):
        assert_same_backend(out.backend, _HFJudge(), strict=True)


def test_same_backend_invariant_matched_backends_ok():
    out = generate(_Seam(), None, _MSGS, max_new_tokens=8, backend="hf")
    # Same backend ⇒ no raise, returns the agreed backend.
    assert assert_same_backend(out.backend, _HFJudge(), strict=True) == "hf"


def test_same_backend_invariant_undeclared_judge_is_hf():
    # A judge with no backend attr defaults to hf, so an HF attack is fine...
    assert assert_same_backend("hf", _UndeclaredJudge(), strict=True) == "hf"
    # ...but a vLLM attack scored by that (hf-default) judge raises.
    with pytest.raises(BackendMismatchError):
        assert_same_backend("vllm", _UndeclaredJudge(), strict=True)


def test_same_backend_invariant_auto_stamps_duck_typed_vllm_judge():
    # A vLLM judge that never set ``backend`` is still detected ⇒ mismatch raises.
    with pytest.raises(BackendMismatchError):
        assert_same_backend("hf", _FakeVLLMEngine(), strict=True)


def test_same_backend_invariant_warns_when_not_strict():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = assert_same_backend("hf", _VLLMJudge(), strict=False)
    assert result == "hf"  # warns but does not raise.
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_same_backend_invariant_rejects_unknown_gen_backend():
    with pytest.raises(ValueError):
        assert_same_backend("sglang", _HFJudge())


def test_valid_backends_constant():
    assert VALID_BACKENDS == frozenset({"hf", "vllm"})
    assert BACKEND_HF == "hf"
    assert BACKEND_VLLM == "vllm"


# --------------------------------------------------------------------------- #
# Determinism with a fixed seed (the chunk done-check) — offline seam.
# --------------------------------------------------------------------------- #


def test_determinism_fixed_seed_offline():
    a = generate(_Seam(), None, _MSGS, max_new_tokens=8, backend="hf", seed=0)
    b = generate(_Seam(), None, _MSGS, max_new_tokens=8, backend="hf", seed=0)
    assert a.text == b.text
    assert a.stamp == b.stamp


# --------------------------------------------------------------------------- #
# REAL HF greedy path on a TINY CPU model (proves the production path).
# --------------------------------------------------------------------------- #


def test_generate_real_tiny_model_greedy_and_deterministic():
    """Greedy N=512-capable generation on GPT-2 (CPU), determinism w/ fixed seed.

    Proves the production render → ``model.generate(do_sample=False)`` path runs
    without a seam, on a tiny model, and that a fixed seed yields byte-identical
    text across two calls. Uses a small N so it is fast (the path is identical for
    N=512). Skips (never fails/hangs) if torch/transformers/network are missing;
    never touches a large model. The 7-20B zoo / 24GB-GPU loads are DEFERRED-NO-GPU.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # noqa: BLE001 - offline/network -> skip, never fail.
        pytest.skip(f"could not load tiny model gpt2: {exc}")
    model.eval()

    msgs = [{"role": "user", "content": "Tell me about birds."}]
    a = generate(model, tok, msgs, max_new_tokens=8, backend="hf", seed=0)
    b = generate(model, tok, msgs, max_new_tokens=8, backend="hf", seed=0)

    assert isinstance(a, GenerationOutput)
    assert a.backend == "hf"
    assert a.stamp["backend"] == "hf"
    assert a.stamp["do_sample"] is False
    assert a.n_new_tokens > 0
    assert a.n_prompt_tokens > 0
    # Greedy + fixed seed ⇒ byte-identical (the determinism done-check).
    assert a.text == b.text


def test_generate_real_tiny_model_n512_runs():
    """The N=512 greedy budget actually runs on a tiny CPU model (done-check).

    Proves ``max_new_tokens=512`` works end-to-end on GPT-2 (CPU). Kept separate
    and skipped when torch is absent so it never hangs; GPT-2's small context still
    completes quickly on CPU.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # noqa: BLE001 - offline/network -> skip, never fail.
        pytest.skip(f"could not load tiny model gpt2: {exc}")
    model.eval()

    out = generate(
        model,
        tok,
        [{"role": "user", "content": "Count."}],
        max_new_tokens=512,
        backend="hf",
        seed=0,
    )
    assert out.backend == "hf"
    assert 0 < out.n_new_tokens <= 512
    assert out.stamp["max_new_tokens"] == 512
