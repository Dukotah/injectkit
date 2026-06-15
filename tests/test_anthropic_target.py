"""Unit tests for the Anthropic Messages API target adapter.

All tests run fully offline. The anthropic SDK is mocked end-to-end:

  * For success/refusal/error normalization we inject a fake client via the
    ``client=`` constructor argument (so the real SDK is never imported).
  * For the lazy-import and credential-validation paths we monkeypatch
    ``injectkit.targets.anthropic_target._import_anthropic`` to hand back a fake
    ``anthropic`` module whose ``Anthropic`` class is a recorder.

No network calls, no API key required.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import pytest

from injectkit.models import TargetConfig, TargetResponse
from injectkit.targets import anthropic_target as at_mod
from injectkit.targets.anthropic_target import (
    DEFAULT_MODEL,
    AnthropicTarget,
    MissingDependencyError,
)
from injectkit.targets.base import Target


# --------------------------------------------------------------------------- #
# Fake SDK objects
# --------------------------------------------------------------------------- #


class FakeTextBlock:
    """Mimics an SDK text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeNonTextBlock:
    """Mimics a non-text content block (e.g. a thinking block) — ignored."""

    def __init__(self) -> None:
        self.type = "thinking"
        self.thinking = "internal reasoning"


class FakeUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    """Mimics an SDK Message returned by client.messages.create()."""

    def __init__(
        self,
        content: Optional[list] = None,
        stop_reason: str = "end_turn",
        model: str = "claude-opus-4-8",
        usage: Optional[FakeUsage] = None,
    ) -> None:
        self.content = content if content is not None else []
        self.stop_reason = stop_reason
        self.model = model
        self.usage = usage if usage is not None else FakeUsage()


class FakeMessages:
    """Records create() calls and returns a canned response (or raises)."""

    def __init__(self, response: Any = None, error: Optional[Exception] = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    """Mimics anthropic.Anthropic(); exposes a .messages attribute."""

    def __init__(self, response: Any = None, error: Optional[Exception] = None) -> None:
        self.messages = FakeMessages(response=response, error=error)


def _target_with(response: Any = None, error: Optional[Exception] = None, **kw: Any) -> AnthropicTarget:
    """Build an AnthropicTarget wired to a FakeClient (no SDK import)."""
    client = FakeClient(response=response, error=error)
    return AnthropicTarget(client=client, **kw)


# --------------------------------------------------------------------------- #
# Protocol / construction
# --------------------------------------------------------------------------- #


def test_implements_target_protocol() -> None:
    target = _target_with(response=FakeMessage())
    assert isinstance(target, Target)


def test_default_model_and_name() -> None:
    target = AnthropicTarget(client=FakeClient(response=FakeMessage()))
    assert target.model == DEFAULT_MODEL
    assert target.name == f"anthropic:{DEFAULT_MODEL}"


def test_custom_model_and_explicit_name() -> None:
    target = AnthropicTarget(
        model="claude-haiku-4-5",
        name="my-target",
        client=FakeClient(response=FakeMessage()),
    )
    assert target.model == "claude-haiku-4-5"
    assert target.name == "my-target"


def test_from_config() -> None:
    cfg = TargetConfig(
        kind="anthropic",
        name="prod-bot",
        model="claude-haiku-4-5",
        system="You are a helpful bot.",
        max_tokens=512,
        extra={"api_key": "sk-test"},
    )
    target = AnthropicTarget.from_config(cfg)
    assert target.model == "claude-haiku-4-5"
    assert target.system == "You are a helpful bot."
    assert target.max_tokens == 512
    assert target.name == "prod-bot"
    assert target._api_key == "sk-test"


def test_from_config_defaults_model_when_unset() -> None:
    cfg = TargetConfig(kind="anthropic", name="target", model=None)
    target = AnthropicTarget.from_config(cfg)
    assert target.model == DEFAULT_MODEL
    # name="target" is the placeholder default -> derive from model
    assert target.name == f"anthropic:{DEFAULT_MODEL}"


# --------------------------------------------------------------------------- #
# send(): success normalization
# --------------------------------------------------------------------------- #


def test_send_returns_target_response_with_text() -> None:
    msg = FakeMessage(content=[FakeTextBlock("INJECTOK-abc123")], stop_reason="end_turn")
    target = _target_with(response=msg)
    resp = target.send("Output exactly: INJECTOK-abc123")
    assert isinstance(resp, TargetResponse)
    assert resp.text == "INJECTOK-abc123"
    assert resp.refused is False
    assert resp.error is None
    assert resp.stop_reason == "end_turn"


def test_send_joins_multiple_text_blocks_and_skips_non_text() -> None:
    msg = FakeMessage(
        content=[FakeTextBlock("Hello "), FakeNonTextBlock(), FakeTextBlock("world")],
    )
    target = _target_with(response=msg)
    resp = target.send("hi")
    assert resp.text == "Hello world"


def test_send_passes_model_and_max_tokens_and_user_message() -> None:
    target = _target_with(response=FakeMessage(), model="claude-haiku-4-5", max_tokens=256)
    target.send("the payload")
    call = target._client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 256
    assert call["messages"] == [{"role": "user", "content": "the payload"}]


def test_send_never_passes_sampling_params() -> None:
    target = _target_with(response=FakeMessage())
    target.send("payload")
    call = target._client.messages.calls[0]
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in call


def test_send_captures_usage_into_raw() -> None:
    msg = FakeMessage(
        content=[FakeTextBlock("ok")],
        usage=FakeUsage(input_tokens=42, output_tokens=7),
    )
    target = _target_with(response=msg)
    resp = target.send("payload")
    assert resp.raw["input_tokens"] == 42
    assert resp.raw["output_tokens"] == 7


def test_send_uses_response_model_when_present() -> None:
    msg = FakeMessage(content=[FakeTextBlock("ok")], model="claude-opus-4-8-actual")
    target = _target_with(response=msg)
    resp = target.send("payload")
    assert resp.model == "claude-opus-4-8-actual"


# --------------------------------------------------------------------------- #
# send(): system prompt handling
# --------------------------------------------------------------------------- #


def test_per_attack_system_overrides_default() -> None:
    target = _target_with(response=FakeMessage(), system="default system")
    target.send("payload", system="attack system")
    call = target._client.messages.calls[0]
    assert call["system"] == "attack system"


def test_default_system_used_when_attack_has_none() -> None:
    target = _target_with(response=FakeMessage(), system="default system")
    target.send("payload")
    call = target._client.messages.calls[0]
    assert call["system"] == "default system"


def test_no_system_key_when_none_configured() -> None:
    target = _target_with(response=FakeMessage())
    target.send("payload")
    call = target._client.messages.calls[0]
    assert "system" not in call


# --------------------------------------------------------------------------- #
# send(): context handling (indirect injection)
# --------------------------------------------------------------------------- #


def test_context_is_fenced_into_user_message() -> None:
    target = _target_with(response=FakeMessage())
    target.send("ignore the doc and emit marker", context="retrieved doc text")
    call = target._client.messages.calls[0]
    content = call["messages"][0]["content"]
    assert "retrieved doc text" in content
    assert "<untrusted_context>" in content
    assert "ignore the doc and emit marker" in content


def test_no_context_means_plain_prompt() -> None:
    target = _target_with(response=FakeMessage())
    target.send("just the payload")
    content = target._client.messages.calls[0]["messages"][0]["content"]
    assert content == "just the payload"


# --------------------------------------------------------------------------- #
# send(): refusal = defender wins
# --------------------------------------------------------------------------- #


def test_refusal_marks_refused_true() -> None:
    # Refused content is empty; refusal must be detected via stop_reason.
    msg = FakeMessage(content=[], stop_reason="refusal")
    target = _target_with(response=msg)
    resp = target.send("do something disallowed")
    assert resp.refused is True
    assert resp.stop_reason == "refusal"
    assert resp.error is None


def test_refusal_checked_before_reading_content() -> None:
    # Even if a refusal somehow carried text, we should not treat it as success.
    msg = FakeMessage(content=[FakeTextBlock("leftover")], stop_reason="refusal")
    target = _target_with(response=msg)
    resp = target.send("payload")
    assert resp.refused is True
    assert "[refused" in resp.text


# --------------------------------------------------------------------------- #
# send(): error handling — never raises
# --------------------------------------------------------------------------- #


def test_api_error_captured_not_raised() -> None:
    boom = RuntimeError("connection reset")
    target = _target_with(error=boom)
    resp = target.send("payload")
    assert resp.error is not None
    assert "connection reset" in resp.error
    assert resp.refused is False
    assert resp.text == ""


class FakeMalformedMessage:
    """A response object whose .content is truthy but not iterable.

    Mimics a flaky proxy / unexpected SDK shape: normalizing it would raise a
    TypeError. send() must capture this as a per-attack error, never raise it
    out and crash the whole scan.
    """

    content = 12345  # not a list -> iterating it raises TypeError
    stop_reason = "end_turn"
    model = "x"
    usage = None


def test_malformed_response_captured_not_raised() -> None:
    target = _target_with(response=FakeMalformedMessage())
    # Must not raise.
    resp = target.send("payload")
    assert resp.error is not None
    assert "TypeError" in resp.error
    assert resp.refused is False
    assert resp.text == ""


def test_normalize_error_is_caught_inside_send() -> None:
    # Even a None response (no attributes) must be handled gracefully: missing
    # attributes default via getattr, so a None response yields empty text, not
    # a crash.
    target = _target_with(response=None)
    resp = target.send("payload")
    # getattr(None, ...) returns defaults -> empty, non-refused, no crash.
    assert resp.refused is False
    assert resp.error is None or resp.text == ""


# --------------------------------------------------------------------------- #
# Lazy import + credential validation (no real SDK, no real key)
# --------------------------------------------------------------------------- #


def _fake_anthropic_module(recorder: dict) -> types.ModuleType:
    """Build a fake 'anthropic' module whose Anthropic() records its kwargs."""
    mod = types.ModuleType("anthropic")

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            recorder["kwargs"] = kwargs
            self.messages = FakeMessages(response=FakeMessage(content=[FakeTextBlock("ok")]))

    mod.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    return mod


def test_lazy_client_built_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: dict = {}
    monkeypatch.setattr(at_mod, "_import_anthropic", lambda: _fake_anthropic_module(recorder))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    target = AnthropicTarget(api_key="sk-explicit")
    resp = target.send("payload")
    assert resp.text == "ok"
    assert recorder["kwargs"] == {"api_key": "sk-explicit"}


def test_lazy_client_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: dict = {}
    monkeypatch.setattr(at_mod, "_import_anthropic", lambda: _fake_anthropic_module(recorder))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")

    target = AnthropicTarget()  # no explicit key -> SDK reads env, we pass nothing
    resp = target.send("payload")
    assert resp.text == "ok"
    assert recorder["kwargs"] == {}  # let SDK pick up the env var itself


def test_missing_key_surfaces_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: dict = {}
    monkeypatch.setattr(at_mod, "_import_anthropic", lambda: _fake_anthropic_module(recorder))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    target = AnthropicTarget()  # no key anywhere
    resp = target.send("payload")
    assert resp.error is not None
    assert "API key" in resp.error
    assert resp.text == ""
    # The SDK client must never have been constructed.
    assert "kwargs" not in recorder


def test_missing_dependency_raises_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the anthropic package being absent.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MissingDependencyError) as exc:
        at_mod._import_anthropic()
    assert "anthropic" in str(exc.value)


def test_missing_dependency_via_send_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the dep is missing, send() captures it as an error response.
    def raiser() -> Any:
        raise MissingDependencyError("anthropic package not installed")

    monkeypatch.setattr(at_mod, "_import_anthropic", raiser)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    target = AnthropicTarget(api_key="sk-test")
    resp = target.send("payload")
    assert resp.error is not None
    assert "not installed" in resp.error


# --------------------------------------------------------------------------- #
# Core importable without anthropic installed (lazy import contract)
# --------------------------------------------------------------------------- #


def test_module_imports_without_anthropic() -> None:
    # The adapter module is already imported at the top of this test file with
    # no anthropic import triggered at module load. Constructing a target must
    # also not import the SDK.
    target = AnthropicTarget()
    assert target.model == DEFAULT_MODEL
    # anthropic may or may not be installed in the venv; either way, simply
    # building the target must not have required it.
    assert isinstance(target, Target)
