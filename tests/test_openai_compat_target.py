"""Unit tests for the OpenAI-compatible local-server target adapter.

Fully offline: the lazy-imported ``requests`` module is monkeypatched with a
fake that records the POST and returns a scripted response, so no network calls
happen. Covers single-shot ``send``, multi-turn ``chat``, system-prompt and
context handling, auth header, config construction, the ``/chat/completions``
payload shape, and every error path (missing dependency, unreachable server,
non-2xx status, non-JSON body, missing reply field).
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
from injectkit.targets.openai_compat import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatTarget,
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

    def post(
        self, url: str, json=None, headers=None, timeout=None  # noqa: A002
    ) -> _FakeResponse:
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
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
    payload = {
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    payload.update(extra)
    return _FakeResponse(status_code=200, payload=payload)


# --------------------------------------------------------------------------- #
# Construction / protocol conformance
# --------------------------------------------------------------------------- #
def test_defaults_and_name() -> None:
    t = OpenAICompatTarget()
    assert t.model == DEFAULT_MODEL
    assert t.base_url == DEFAULT_BASE_URL
    assert t.api_key is None
    assert t.name == f"openai-compat:{DEFAULT_MODEL}"


def test_custom_name_and_base_url_strips_trailing_slash() -> None:
    t = OpenAICompatTarget(
        model="mixtral", base_url="http://box:8000/v1/", name="vllm"
    )
    assert t.base_url == "http://box:8000/v1"
    assert t.name == "vllm"


def test_satisfies_both_protocols() -> None:
    t = OpenAICompatTarget()
    assert isinstance(t, Target)
    assert isinstance(t, ConversationalTarget)


def test_as_conversational_returns_self() -> None:
    t = OpenAICompatTarget()
    assert as_conversational(t) is t


def test_from_config_reads_fields() -> None:
    cfg = TargetConfig(
        kind="openai-compat",
        name="myserver",
        model="qwen2",
        system="be terse",
        url="http://host:1234/v1",
        timeout_s=7.0,
        max_tokens=256,
        extra={
            "api_key": "sk-local",
            "extra_body": {"temperature": 0},
            "headers": {"X-Org": "lab"},
        },
    )
    t = OpenAICompatTarget.from_config(cfg)
    assert t.model == "qwen2"
    assert t.base_url == "http://host:1234/v1"
    assert t.system == "be terse"
    assert t.timeout_s == 7.0
    assert t.max_tokens == 256
    assert t.api_key == "sk-local"
    assert t.extra_body == {"temperature": 0}
    assert t.headers == {"X-Org": "lab"}
    assert t.name == "myserver"


def test_from_config_base_url_falls_back_to_extra_then_default() -> None:
    t_extra = OpenAICompatTarget.from_config(
        TargetConfig(kind="openai-compat", extra={"base_url": "http://h:1/v1"})
    )
    assert t_extra.base_url == "http://h:1/v1"
    t_default = OpenAICompatTarget.from_config(TargetConfig(kind="openai-compat"))
    assert t_default.base_url == DEFAULT_BASE_URL


def test_from_config_generic_name_not_used() -> None:
    t = OpenAICompatTarget.from_config(
        TargetConfig(kind="openai-compat", model="m")
    )
    assert t.name == "openai-compat:m"


def test_from_config_merges_headers() -> None:
    cfg = TargetConfig(
        kind="openai-compat",
        headers={"X-Base": "1", "X-Override": "base"},
        extra={"headers": {"X-Override": "extra", "X-Extra": "2"}},
    )
    t = OpenAICompatTarget.from_config(cfg)
    assert t.headers == {"X-Base": "1", "X-Override": "extra", "X-Extra": "2"}


# --------------------------------------------------------------------------- #
# send() happy path + payload shape
# --------------------------------------------------------------------------- #
def test_send_success_extracts_content(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response("INJECTOK-abc123")
    t = OpenAICompatTarget(model="local-model")
    resp = t.send("emit INJECTOK-abc123")
    assert isinstance(resp, TargetResponse)
    assert resp.error is None
    assert resp.text == "INJECTOK-abc123"
    assert resp.refused is False
    assert resp.model == "local-model"
    assert resp.stop_reason == "stop"


def test_send_posts_to_chat_completions_with_stream_false(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget(base_url="http://localhost:8000/v1", timeout_s=5.0)
    t.send("hi")
    call = fake_requests.calls[0]
    assert call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["timeout"] == 5.0
    body = call["json"]
    assert body["model"] == DEFAULT_MODEL
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "max_tokens" not in body


def test_send_no_auth_header_when_keyless(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    OpenAICompatTarget().send("hi")
    headers = fake_requests.calls[0]["headers"]
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_send_bearer_auth_when_key_set(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    OpenAICompatTarget(api_key="sk-xyz").send("hi")
    headers = fake_requests.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer sk-xyz"


def test_send_includes_system_message(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget(system="default-sys")
    t.send("hi")
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert msgs[0] == {"role": "system", "content": "default-sys"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_send_per_attack_system_overrides_default(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget(system="default-sys")
    t.send("hi", system="attack-sys")
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert msgs[0] == {"role": "system", "content": "attack-sys"}


def test_send_context_is_fenced(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget()
    t.send("do the thing", context="retrieved doc body")
    user_content = fake_requests.calls[0]["json"]["messages"][0]["content"]
    assert "untrusted_context" in user_content
    assert "retrieved doc body" in user_content
    assert "do the thing" in user_content


def test_send_max_tokens_and_extra_body_forwarded(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget(max_tokens=128, extra_body={"temperature": 0, "seed": 1})
    t.send("hi")
    body = fake_requests.calls[0]["json"]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0
    assert body["seed"] == 1


def test_extra_body_cannot_clobber_core_fields(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response()
    t = OpenAICompatTarget(
        model="real-model",
        extra_body={"model": "evil", "messages": [], "stream": True, "ok": 1},
    )
    t.send("hi")
    body = fake_requests.calls[0]["json"]
    assert body["model"] == "real-model"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["ok"] == 1


def test_send_surfaces_finish_reason_as_stop_reason(
    fake_requests: _FakeRequests,
) -> None:
    fake_requests.response = _ok_response("x", **{})
    # Override finish_reason to "length".
    fake_requests.response = _FakeResponse(
        status_code=200,
        payload={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "length",
                }
            ]
        },
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.stop_reason == "length"


# --------------------------------------------------------------------------- #
# chat() multi-turn
# --------------------------------------------------------------------------- #
def test_chat_forwards_full_history(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _ok_response("ok")
    t = OpenAICompatTarget()
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
    t = OpenAICompatTarget()  # no default system
    t.chat([ChatMessage(role="user", content="hi")])
    msgs = fake_requests.calls[0]["json"]["messages"]
    assert all(m["role"] != "system" for m in msgs)


# --------------------------------------------------------------------------- #
# Error paths — never raise, always return error response
# --------------------------------------------------------------------------- #
def test_missing_requests_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "requests":
            raise ImportError("no module named requests")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "requests", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    resp = OpenAICompatTarget().send("hi")
    assert resp.error is not None
    assert "requests" in resp.error
    assert resp.text == ""


def test_unreachable_server(fake_requests: _FakeRequests) -> None:
    fake_requests.raise_exc = ConnectionError("connection refused")
    resp = OpenAICompatTarget(base_url="http://localhost:8000/v1").send("hi")
    assert resp.error is not None
    assert "could not reach OpenAI-compatible server" in resp.error
    assert resp.text == ""


def test_non_2xx_status(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(status_code=503, text="boom")
    resp = OpenAICompatTarget().send("hi")
    assert resp.error == "server returned HTTP 503"
    assert resp.stop_reason == "http_503"
    assert resp.raw["response_text"] == "boom"


def test_non_json_body(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200, text="not json", json_raises=True
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.error is not None
    assert "not valid JSON" in resp.error


def test_missing_choices(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200, payload={"model": "x", "object": "chat.completion"}
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.error is not None
    assert "choices[0]" in resp.error


def test_missing_message_content(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200,
        payload={"choices": [{"index": 0, "finish_reason": "stop"}]},
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.error is not None
    assert "message.content" in resp.error


def test_empty_content_is_not_an_error(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200,
        payload={
            "choices": [
                {"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
            ]
        },
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.error is None
    assert resp.text == ""


def test_null_content_is_normalized_to_empty(fake_requests: _FakeRequests) -> None:
    fake_requests.response = _FakeResponse(
        status_code=200,
        payload={
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )
    resp = OpenAICompatTarget().send("hi")
    assert resp.error is None
    assert resp.text == ""
    assert resp.stop_reason == "tool_calls"
