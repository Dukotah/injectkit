"""Unit tests for the Ollama local-server target adapter.

Fully offline: the lazy-imported ``requests`` module is monkeypatched with a
fake that records the POST and returns a scripted response, so no network calls
happen. Covers single-shot ``send``, multi-turn ``chat``, system-prompt and
context handling, config construction, the ``/api/chat`` payload shape, and
every error path (missing dependency, unreachable server, non-2xx status,
non-JSON body, missing reply field).
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import pytest

from injectkit.models import TargetConfig, TargetResponse
from injectkit.targets.conversational import (
    ChatMessage,
    ConversationalTarget,
    as_conversational,
)
from injectkit.targets.base import Target
from injectkit.targets.ollama import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaTarget,
)


# --------------------------------------------------------------------------- #
# Fake requests module + response
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        json_raises: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_raises = json_raises

    def json(self) -> Any:
        if self._json_raises:
            raise ValueError("no json could be decoded")
        return self._payload


class _FakeRequests:
    """Minimal stand-in for the ``requests`` module."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: Optional[_FakeResponse] = None
        self.raise_exc: Optional[Exception] = None

    def post(self, url: str, json=None, timeout=None) -> _FakeResponse:  # noqa: A002
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.response is not None, "test did not script a response"
        return self.response


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> _FakeRequests:
    """Install a fake ``requests`` module so the lazy import picks it up."""
    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def _ok_response(content: str = "hello", **extra: Any) -> _FakeResponse:
    payload = {"model": "llama3", "message": {"role": "assistant", "content": content}}
    payload.update(extra)
    return _FakeResponse(status_code=200, payload=payload)


# --------------------------------------------------------------------------- #
# Construction / protocol conformance
# --------------------------------------------------------------------------- #
def test_defaults_and_name() -> None:
    t = OllamaTarget()
    assert t.model == DEFAULT_MODEL
    assert t.host == DEFAULT_HOST
    assert t.name == f"ollama:{DEFAULT_MODEL}"


def test_custom_name_and_host_strips_trailing_slash() -> None:
    t = OllamaTarget(model="mistral", host="http://box:11434/", name="local")
    assert t.host == "http://box:11434"
    assert t.name == "local"


def test_satisfies_both_protocols() -> None:
    t = OllamaTarget()
    assert isinstance(t, Target)
    assert isinstance(t, ConversationalTarget)


def test_as_conversational_returns_self() -> None:
    # OllamaTarget already has chat(), so as_conversational must not wrap it.
    t = OllamaTarget()
    assert as_conversational(t) is t


def test_from_config_reads_fields() -> None:
    cfg = TargetConfig(
        kind="ollama",
        name="myollama",
        model="qwen2",
        system="be terse",
        url="http://host:1234",
        timeout_s=7.0,
        extra={"options": {"temperature": 0}},
    )
    t = OllamaTarget.from_config(cfg)
    assert t.model == "qwen2"
    assert t.host == "http://host:1234"
    assert t.system == "be terse"
    assert t.timeout_s == 7.0
    assert t.options == {"temperature": 0}
    assert t.name == "myollama"


def test_from_config_host_falls_back_to_extra_then_default() -> None:
    t_extra = OllamaTarget.from_config(
        TargetConfig(kind="ollama", extra={"host": "http://h:1"})
    )
    assert t_extra.host == "http://h:1"
    t_default = OllamaTarget.from_config(TargetConfig(kind="ollama"))
    assert t_default.host == DEFAULT_HOST


def test_from_config_generic_name_not_used() -> None:
    # The sentinel default "target" should not override "ollama:<model>".
    t = OllamaTarget.from_config(TargetConfig(kind="ollama", model="m"))
    assert t.name == "ollama:m"


# --------------------------------------------------------------------------- #
# send() happy path + payload shape
# --------------------------------------------------------------------------- #
def test_send_success_extracts_content(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response("INJECTOK-abc123")
    t = OllamaTarget(model="llama3")
    resp = t.send("emit INJECTOK-abc123")
    assert isinstance(resp, TargetResponse)
    assert resp.error is None
    assert resp.text == "INJECTOK-abc123"
    assert resp.refused is False
    assert resp.model == "llama3"


def test_send_posts_to_api_chat_with_stream_false(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget(host="http://localhost:11434", timeout_s=5.0)
    t.send("hi")
    call = fake_requests.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == 5.0
    body = call["json"]
    assert body["model"] == DEFAULT_MODEL
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "options" not in body


def test_send_includes_system_message(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget(system="default-sys")
    t.send("hi")
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert msgs[0] == {"role": "system", "content": "default-sys"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_send_per_attack_system_overrides_default(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget(system="default-sys")
    t.send("hi", system="attack-sys")
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert msgs[0] == {"role": "system", "content": "attack-sys"}


def test_send_context_is_fenced(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget()
    t.send("do the thing", context="retrieved doc body")
    user_content = fake_requests.calls[0]["json"]["messages"][0]["content"]
    assert "untrusted_context" in user_content
    assert "retrieved doc body" in user_content
    assert "do the thing" in user_content


def test_send_options_forwarded(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget(options={"temperature": 0, "seed": 1})
    t.send("hi")
    assert fake_requests.calls[0]["json"]["options"] == {"temperature": 0, "seed": 1}


def test_send_surfaces_done_reason_as_stop_reason(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response("x", done_reason="length")
    resp = OllamaTarget().send("hi")
    assert resp.stop_reason == "length"


# --------------------------------------------------------------------------- #
# chat() multi-turn
# --------------------------------------------------------------------------- #
def test_chat_forwards_full_history(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response("ok")
    t = OllamaTarget()
    messages = [
        ChatMessage(role="user", content="turn one"),
        ChatMessage(role="assistant", content="reply one"),
        ChatMessage(role="user", content="turn two"),
    ]
    resp = t.chat(messages, system="sys")
    assert resp.text == "ok"
    sent = fake_requests.calls[0]["json"]["messages"]
    assert sent == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "turn two"},
    ]


def test_chat_without_system_omits_system_message(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response()
    t = OllamaTarget()  # no default system
    t.chat([ChatMessage(role="user", content="hi")])
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert all(m["role"] != "system" for m in msgs)


# --------------------------------------------------------------------------- #
# Error paths — never raise, always return error response
# --------------------------------------------------------------------------- #
def test_missing_requests_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy import to fail by making `import requests` raise.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "requests":
            raise ImportError("no module named requests")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "requests", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    resp = OllamaTarget().send("hi")
    assert resp.error is not None
    assert "requests" in resp.error
    assert resp.text == ""


def test_unreachable_server(fake_requests: _FakeRequests) -> None:
    fake_requests.raise_exc = ConnectionError("connection refused")
    resp = OllamaTarget(host="http://localhost:11434").send("hi")
    assert resp.error is not None
    assert "could not reach Ollama" in resp.error
    assert "ollama serve" in resp.error
    assert resp.text == ""


def test_non_2xx_status(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(status_code=500, text="boom")
    resp = OllamaTarget().send("hi")
    assert resp.error == "Ollama returned HTTP 500"
    assert resp.stop_reason == "http_500"
    assert resp.raw["response_text"] == "boom"


def test_non_json_body(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200, text="not json", json_raises=True
    )
    resp = OllamaTarget().send("hi")
    assert resp.error is not None
    assert "not valid JSON" in resp.error


def test_missing_message_content(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200, payload={"model": "llama3", "done": True}
    )
    resp = OllamaTarget().send("hi")
    assert resp.error is not None
    assert "message.content" in resp.error


def test_empty_content_is_not_an_error(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200,
        payload={"model": "llama3", "message": {"role": "assistant", "content": ""}},
    )
    resp = OllamaTarget().send("hi")
    assert resp.error is None
    assert resp.text == ""
