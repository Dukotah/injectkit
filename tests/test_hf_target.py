"""Unit tests for the HuggingFace Transformers in-process target adapter.

Fully offline: ``torch`` and ``transformers`` are NEVER imported for real and no
model is ever downloaded or loaded. Tests either inject fake ``tokenizer`` +
``model_obj`` objects directly, or monkeypatch the lazy-imported modules in
``sys.modules`` so the load path picks up fakes. The fakes use a trivial
character-level tokenization so we can assert on the exact prompt the model
sees, the chat-template vs transcript fallback, prompt-stripping, system/context
handling, config construction, and every error path.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import pytest

from injectkit.models import TargetConfig, TargetResponse
from injectkit.targets.base import Target
from injectkit.targets.conversational import (
    ChatMessage,
    ConversationalTarget,
    as_conversational,
)
from injectkit.targets.hf import DEFAULT_MODEL, HFTarget, MissingDependencyError


# --------------------------------------------------------------------------- #
# Fake torch / transformers
# --------------------------------------------------------------------------- #
class _FakeNoGrad:
    def __enter__(self) -> "_FakeNoGrad":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeTorch:
    """Minimal stand-in for ``torch`` covering what the adapter touches."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeTokenizer:
    """Char-level fake tokenizer with an optional chat template.

    ``__call__`` maps a string to per-character ordinals; ``decode`` maps the
    ordinals back to characters. This lets a test assert that the model received
    exactly the rendered prompt and that only the generated suffix is returned.
    """

    def __init__(self, chat_template: Optional[str] = None) -> None:
        self.chat_template = chat_template
        self.pad_token_id: Optional[int] = None
        self.eos_token_id = 7
        self.applied: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        self.applied.append(messages)
        body = "|".join(f"{m['role']}={m['content']}" for m in messages)
        suffix = "|gen" if add_generation_prompt else ""
        return f"<TMPL>{body}{suffix}"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
        ids = [ord(c) for c in text]
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(t)) for t in token_ids)


class _FakeModel:
    """Fake causal-LM whose ``generate`` appends a scripted reply string.

    It echoes the prompt tokens (as a real model output does) followed by the
    character ordinals of ``reply``, so the adapter's prompt-stripping returns
    exactly ``reply``.
    """

    def __init__(self, reply: str = "INJECTOK-abc") -> None:
        self.reply = reply
        self.eval_called = False
        self.moved_to: Optional[str] = None
        self.generate_kwargs: list[dict[str, Any]] = []

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self

    def to(self, device: str) -> "_FakeModel":
        self.moved_to = device
        return self

    def generate(self, input_ids: Any, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs.append(kwargs)
        prompt_ids = list(input_ids[0])
        new_ids = [ord(c) for c in self.reply]
        return [prompt_ids + new_ids]


class _FakeTransformers:
    """Stand-in for the ``transformers`` module's auto classes."""

    def __init__(self, tokenizer: _FakeTokenizer, model: _FakeModel) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self.tokenizer_loads: list[str] = []
        self.model_loads: list[str] = []

        outer = self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str) -> _FakeTokenizer:
                outer.tokenizer_loads.append(model_id)
                return outer._tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id: str) -> _FakeModel:
                outer.model_loads.append(model_id)
                return outer._model

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch):
    """Install fake torch + transformers into sys.modules for the lazy import.

    Returns a small namespace exposing the fakes so tests can assert on them.
    """

    def _install(
        reply: str = "INJECTOK-abc",
        chat_template: Optional[str] = "TEMPLATE",
    ):
        torch = _FakeTorch()
        tokenizer = _FakeTokenizer(chat_template=chat_template)
        model = _FakeModel(reply=reply)
        transformers = _FakeTransformers(tokenizer, model)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        return type(
            "HF",
            (),
            {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
                "transformers": transformers,
            },
        )

    return _install


# --------------------------------------------------------------------------- #
# Construction / protocol conformance
# --------------------------------------------------------------------------- #
def test_defaults_and_name() -> None:
    t = HFTarget()
    assert t.model == DEFAULT_MODEL
    assert t.name == f"hf:{DEFAULT_MODEL}"
    assert t.max_new_tokens == 256


def test_custom_name() -> None:
    t = HFTarget(model="gpt2", name="local-gpt2")
    assert t.name == "local-gpt2"


def test_satisfies_both_protocols() -> None:
    t = HFTarget()
    assert isinstance(t, Target)
    assert isinstance(t, ConversationalTarget)


def test_as_conversational_returns_self() -> None:
    t = HFTarget()
    assert as_conversational(t) is t


def test_from_config_reads_fields() -> None:
    cfg = TargetConfig(
        kind="hf",
        name="myhf",
        model="gpt2",
        system="be terse",
        max_tokens=64,
        extra={"device": "cpu", "seed": 5},
    )
    t = HFTarget.from_config(cfg)
    assert t.model == "gpt2"
    assert t.system == "be terse"
    assert t.max_new_tokens == 64
    assert t.device == "cpu"
    assert t.seed == 5
    assert t.name == "myhf"


def test_from_config_generic_name_not_used() -> None:
    t = HFTarget.from_config(TargetConfig(kind="hf", model="gpt2"))
    assert t.name == "hf:gpt2"


def test_construction_does_not_import_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building the target must never import torch/transformers.
    import builtins

    real_import = builtins.__import__

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "transformers"}:
            raise AssertionError(f"unexpected import of {name} at construction")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    HFTarget(model="gpt2")  # must not raise


# --------------------------------------------------------------------------- #
# Injected tokenizer/model (no import at all)
# --------------------------------------------------------------------------- #
def test_send_with_injected_objects_strips_prompt() -> None:
    tok = _FakeTokenizer(chat_template=None)  # force transcript path
    model = _FakeModel(reply="INJECTOK-xyz")
    t = HFTarget(model="stub", tokenizer=tok, model_obj=model)
    resp = t.send("emit INJECTOK-xyz")
    assert isinstance(resp, TargetResponse)
    assert resp.error is None
    # Only the generated continuation is returned, not the echoed prompt.
    assert resp.text == "INJECTOK-xyz"
    assert resp.refused is False
    assert resp.stop_reason == "stop"
    assert resp.model == "stub"


# --------------------------------------------------------------------------- #
# Lazy-load happy path + prompt rendering
# --------------------------------------------------------------------------- #
def test_send_loads_model_lazily(fake_hf) -> None:
    hf = fake_hf(reply="INJECTOK-1")
    t = HFTarget(model="gpt2")
    resp = t.send("hi")
    assert resp.text == "INJECTOK-1"
    assert hf.transformers.tokenizer_loads == ["gpt2"]
    assert hf.transformers.model_loads == ["gpt2"]
    assert hf.model.eval_called is True


def test_load_happens_once(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2")
    t.send("one")
    t.send("two")
    # Tokenizer + model loaded exactly once and cached.
    assert hf.transformers.tokenizer_loads == ["gpt2"]
    assert hf.transformers.model_loads == ["gpt2"]


def test_chat_template_used_when_available(fake_hf) -> None:
    hf = fake_hf(chat_template="TEMPLATE")
    t = HFTarget(model="gpt2", system="sys-default")
    t.send("hello there")
    # apply_chat_template was called with system + user messages.
    applied = hf.tokenizer.applied[-1]
    assert applied == [
        {"role": "system", "content": "sys-default"},
        {"role": "user", "content": "hello there"},
    ]
    # The rendered prompt reached the tokenizer __call__ (visible via raw prompt).
    # Confirm template rendering shape via the model's seen prompt length being
    # nonzero is implicit; assert the prompt text recorded in raw.
    resp = t.send("again")
    assert resp.raw["prompt"].startswith("<TMPL>")


def test_transcript_fallback_when_no_chat_template(fake_hf) -> None:
    fake_hf(chat_template=None)
    t = HFTarget(model="gpt2")
    resp = t.send("do thing")
    prompt = resp.raw["prompt"]
    assert "User: do thing" in prompt
    assert prompt.rstrip().endswith("Assistant:")


def test_deterministic_generation_kwargs_and_seed(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2", max_new_tokens=42, seed=11)
    t.send("hi")
    kwargs = hf.model.generate_kwargs[-1]
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 42
    assert hf.torch.seeds == [11]


def test_pad_token_falls_back_to_eos(fake_hf) -> None:
    hf = fake_hf()
    # _FakeTokenizer has pad_token_id None and eos_token_id 7.
    t = HFTarget(model="gpt2")
    t.send("hi")
    assert hf.model.generate_kwargs[-1]["pad_token_id"] == 7


# --------------------------------------------------------------------------- #
# System / context handling
# --------------------------------------------------------------------------- #
def test_per_attack_system_overrides_default(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2", system="default-sys")
    t.send("hi", system="attack-sys")
    applied = hf.tokenizer.applied[-1]
    assert applied[0] == {"role": "system", "content": "attack-sys"}


def test_no_system_omits_system_message(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2")  # no default system
    t.send("hi")
    applied = hf.tokenizer.applied[-1]
    assert all(m["role"] != "system" for m in applied)


def test_context_is_fenced(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2")
    t.send("do the thing", context="retrieved doc body")
    user_msg = hf.tokenizer.applied[-1][-1]["content"]
    assert "untrusted_context" in user_msg
    assert "retrieved doc body" in user_msg
    assert "do the thing" in user_msg


# --------------------------------------------------------------------------- #
# chat() multi-turn
# --------------------------------------------------------------------------- #
def test_chat_forwards_full_history(fake_hf) -> None:
    hf = fake_hf(reply="ok")
    t = HFTarget(model="gpt2")
    messages = [
        ChatMessage(role="user", content="turn one"),
        ChatMessage(role="assistant", content="reply one"),
        ChatMessage(role="user", content="turn two"),
    ]
    resp = t.chat(messages, system="sys")
    assert resp.text == "ok"
    applied = hf.tokenizer.applied[-1]
    assert applied == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "turn two"},
    ]


def test_chat_transcript_fallback_includes_all_roles(fake_hf) -> None:
    fake_hf(chat_template=None, reply="ok")
    t = HFTarget(model="gpt2")
    messages = [
        ChatMessage(role="user", content="t1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="t2"),
    ]
    resp = t.chat(messages)
    prompt = resp.raw["prompt"]
    assert "User: t1" in prompt
    assert "Assistant: a1" in prompt
    assert "User: t2" in prompt


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def test_device_moves_model(fake_hf) -> None:
    hf = fake_hf()
    t = HFTarget(model="gpt2", device="cpu")
    t.send("hi")
    assert hf.model.moved_to == "cpu"


# --------------------------------------------------------------------------- #
# Error paths — never raise, always return error response
# --------------------------------------------------------------------------- #
def test_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "transformers"}:
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    resp = HFTarget(model="gpt2").send("hi")
    assert resp.error is not None
    assert "transformers" in resp.error
    assert "injectkit[hf]" in resp.error
    assert resp.text == ""


def test_load_failure_is_captured(fake_hf, monkeypatch: pytest.MonkeyPatch) -> None:
    hf = fake_hf()

    def boom(model_id: str) -> Any:
        raise OSError("model not found on hub")

    monkeypatch.setattr(
        hf.transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(boom)
    )
    resp = HFTarget(model="missing/model").send("hi")
    assert resp.error is not None
    assert "failed to load HF model" in resp.error
    assert "missing/model" in resp.error
    assert resp.text == ""


def test_generation_failure_is_captured(fake_hf) -> None:
    hf = fake_hf()

    def boom(input_ids: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cuda oom")

    hf.model.generate = boom  # type: ignore[assignment]
    resp = HFTarget(model="gpt2").send("hi")
    assert resp.error is not None
    assert "HF generation failed" in resp.error
    assert "cuda oom" in resp.error
    assert resp.text == ""


def test_empty_reply_is_not_an_error() -> None:
    tok = _FakeTokenizer(chat_template=None)
    model = _FakeModel(reply="")
    t = HFTarget(model="stub", tokenizer=tok, model_obj=model)
    resp = t.send("hi")
    assert resp.error is None
    assert resp.text == ""


def test_missing_dependency_error_type() -> None:
    # MissingDependencyError is a RuntimeError so engine error handling treats it
    # the same as other adapters.
    assert issubclass(MissingDependencyError, RuntimeError)
