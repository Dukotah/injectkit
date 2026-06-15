"""Unit tests for the generic HTTP chat-endpoint target adapter.

Fully offline: httpx is monkeypatched with a fake client so no network calls
happen. Tests cover template rendering, response-path extraction, the OpenAI/
Anthropic-style defaults, and every error path (transport error, non-2xx,
non-JSON body, unresolved response path, missing url, missing httpx).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from injectkit.models import TargetConfig, TargetResponse
from injectkit.targets.http import (
    HttpTarget,
    extract_path,
    render_template,
)


# --------------------------------------------------------------------------- #
# A fake httpx that records requests and returns a scripted response.
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        # If text not given but payload is, serialize it (mirrors httpx).
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeHTTPError(Exception):
    """Stand-in for httpx.HTTPError."""


class _FakeClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.requests: list[dict] = []
        self.closed = False
        # Scripted behavior, set by the test before sending.
        self.response: Optional[_FakeResponse] = None
        self.raise_error: Optional[Exception] = None

    def request(self, method: str, url: str, headers=None, json=None) -> _FakeResponse:
        self.requests.append(
            {"method": method, "url": url, "headers": headers, "json": json}
        )
        if self.raise_error is not None:
            raise self.raise_error
        assert self.response is not None, "test did not script a response"
        return self.response

    def close(self) -> None:
        self.closed = True


class _FakeHttpxModule:
    """A minimal stand-in for the httpx module."""

    HTTPError = _FakeHTTPError

    def __init__(self) -> None:
        self.created: list[_FakeClient] = []

    def Client(self, timeout: float = 30.0) -> _FakeClient:  # noqa: N802 (mimics httpx)
        client = _FakeClient(timeout=timeout)
        self.created.append(client)
        return client


@pytest.fixture
def fake_httpx(monkeypatch: pytest.MonkeyPatch) -> _FakeHttpxModule:
    """Install a fake 'httpx' module so HttpTarget never touches the network."""
    module = _FakeHttpxModule()
    monkeypatch.setitem(__import__("sys").modules, "httpx", module)
    return module


def _config(**overrides: Any) -> TargetConfig:
    base = dict(kind="http", name="test-http", url="https://example.test/chat")
    base.update(overrides)
    return TargetConfig(**base)


# --------------------------------------------------------------------------- #
# render_template
# --------------------------------------------------------------------------- #
def test_render_template_substitutes_nested_placeholders() -> None:
    template = {
        "messages": [
            {"role": "system", "content": "{system}"},
            {"role": "user", "content": "{prompt}"},
        ],
        "meta": {"ctx": "doc: {context}"},
    }
    values = {"prompt": "hi", "system": "be safe", "context": "trusted"}
    out = render_template(template, values)
    assert out["messages"][0]["content"] == "be safe"
    assert out["messages"][1]["content"] == "hi"
    assert out["meta"]["ctx"] == "doc: trusted"


def test_render_template_drops_sole_placeholder_when_none() -> None:
    template = {"system": "{system}", "prompt": "{prompt}"}
    out = render_template(template, {"prompt": "hello", "system": None, "context": None})
    assert "system" not in out  # dropped entirely, not the literal "None"
    assert out["prompt"] == "hello"


def test_render_template_inline_none_becomes_empty() -> None:
    out = render_template("ctx=[{context}]", {"prompt": "p", "system": None, "context": None})
    assert out == "ctx=[]"


def test_render_template_preserves_json_unsafe_values() -> None:
    # A payload with quotes/newlines/braces must not corrupt the body.
    nasty = 'say "{hi}"\nnewline & {braces}'
    out = render_template({"q": "{prompt}"}, {"prompt": nasty, "system": None, "context": None})
    assert out["q"] == nasty


def test_render_template_does_not_reexpand_substituted_values() -> None:
    # SECURITY: an attack payload is untrusted text. If it literally contains
    # another placeholder token (e.g. "{context}"), that token must survive
    # verbatim and NOT be expanded into the system/context value. Substitution
    # is single-pass, so injected placeholders are never re-scanned.
    vals = {"prompt": "leak this: {context}", "system": "S", "context": "CTXSECRET"}
    out = render_template({"q": "{prompt}", "doc": "{context}"}, vals)
    assert out["q"] == "leak this: {context}"  # NOT "leak this: CTXSECRET"
    assert out["doc"] == "CTXSECRET"  # the real placeholder still expands


def test_render_template_does_not_mutate_input() -> None:
    template = {"content": "{prompt}"}
    render_template(template, {"prompt": "x", "system": None, "context": None})
    assert template == {"content": "{prompt}"}


def test_render_template_passes_scalars_through() -> None:
    template = {"temperature": 0, "stream": False, "n": 1}
    out = render_template(template, {"prompt": "p", "system": None, "context": None})
    assert out == {"temperature": 0, "stream": False, "n": 1}


# --------------------------------------------------------------------------- #
# extract_path
# --------------------------------------------------------------------------- #
def test_extract_path_dot_and_index() -> None:
    data = {"choices": [{"message": {"content": "reply!"}}]}
    assert extract_path(data, "choices.0.message.content") == "reply!"


def test_extract_path_missing_key_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        extract_path({"a": 1}, "b")


def test_extract_path_index_out_of_range_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        extract_path({"choices": []}, "choices.0")


def test_extract_path_descend_into_scalar_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        extract_path({"a": "string"}, "a.b")


# --------------------------------------------------------------------------- #
# HttpTarget construction
# --------------------------------------------------------------------------- #
def test_requires_url() -> None:
    with pytest.raises(ValueError):
        HttpTarget(TargetConfig(kind="http", url=None))


def test_name_defaults_to_url() -> None:
    t = HttpTarget(TargetConfig(kind="http", name="", url="https://x.test/c"))
    assert t.name == "http:https://x.test/c"


# --------------------------------------------------------------------------- #
# send: happy paths
# --------------------------------------------------------------------------- #
def test_send_openai_style_default_template_and_path(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    fake_httpx.created  # not yet created
    # First send creates the client; script its response.
    payload = {"choices": [{"message": {"content": "the answer"}}]}

    # Trigger client creation, then script the response on it.
    def run() -> TargetResponse:
        # client is created inside send(); pre-create via _get_client to script.
        client = target._get_client()
        client.response = _FakeResponse(200, payload)
        return target.send("attack prompt", system="sys")

    resp = run()
    assert resp.error is None
    assert resp.text == "the answer"
    assert resp.stop_reason == "http_200"
    sent = fake_httpx.created[0].requests[0]["json"]
    assert sent["messages"][0]["content"] == "sys"
    assert sent["messages"][1]["content"] == "attack prompt"


def test_send_default_system_used_when_attack_has_none(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config(system="DEFAULT-SYS"))
    client = target._get_client()
    client.response = _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
    target.send("p")  # no system passed
    sent = client.requests[0]["json"]
    assert sent["messages"][0]["content"] == "DEFAULT-SYS"


def test_send_custom_template_and_response_path(fake_httpx: _FakeHttpxModule) -> None:
    cfg = _config(
        request_template={"input": "{prompt}", "ctx": "{context}"},
        response_path="data.reply",
        headers={"Authorization": "Bearer T"},
        method="PUT",
    )
    target = HttpTarget(cfg)
    client = target._get_client()
    client.response = _FakeResponse(200, {"data": {"reply": "custom!"}})
    resp = target.send("PROMPT", context="CTX")
    assert resp.text == "custom!"
    req = client.requests[0]
    assert req["method"] == "PUT"
    assert req["headers"] == {"Authorization": "Bearer T"}
    assert req["json"] == {"input": "PROMPT", "ctx": "CTX"}


def test_send_anthropic_style_default_path(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    client = target._get_client()
    client.response = _FakeResponse(200, {"content": [{"text": "anthropic reply"}]})
    resp = target.send("p")
    assert resp.text == "anthropic reply"


def test_send_non_string_reply_is_json_encoded(fake_httpx: _FakeHttpxModule) -> None:
    cfg = _config(response_path="data")
    target = HttpTarget(cfg)
    client = target._get_client()
    client.response = _FakeResponse(200, {"data": {"nested": [1, 2]}})
    resp = target.send("p")
    assert resp.text == json.dumps({"nested": [1, 2]})


# --------------------------------------------------------------------------- #
# send: error paths (never raise)
# --------------------------------------------------------------------------- #
def test_send_transport_error_returns_error(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    client = target._get_client()
    client.raise_error = _FakeHTTPError("connection refused")
    resp = target.send("p")
    assert resp.text == ""
    assert resp.error is not None
    assert "connection refused" in resp.error


def test_send_non_2xx_returns_error(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    client = target._get_client()
    client.response = _FakeResponse(500, text="boom")
    resp = target.send("p")
    assert resp.error == "HTTP 500 from endpoint"
    assert resp.stop_reason == "http_500"
    assert resp.raw["status_code"] == 500


def test_send_invalid_json_returns_error(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    client = target._get_client()
    client.response = _FakeResponse(200, payload=None, text="not json")
    resp = target.send("p")
    assert resp.error is not None
    assert "not valid JSON" in resp.error


def test_send_unresolved_response_path_returns_error(fake_httpx: _FakeHttpxModule) -> None:
    cfg = _config(response_path="nope.here")
    target = HttpTarget(cfg)
    client = target._get_client()
    client.response = _FakeResponse(200, {"something": "else"})
    resp = target.send("p")
    assert resp.error is not None
    assert "could not extract reply" in resp.error


def test_send_default_paths_exhausted_returns_error(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())  # no response_path
    client = target._get_client()
    client.response = _FakeResponse(200, {"weird": {"shape": 1}})
    resp = target.send("p")
    assert resp.error is not None
    assert "could not extract reply" in resp.error


def test_send_missing_httpx_returns_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make `import httpx` fail inside send().
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    target = HttpTarget(_config())
    resp = target.send("p")
    assert resp.text == ""
    assert resp.error is not None
    assert "httpx" in resp.error


# --------------------------------------------------------------------------- #
# client lifecycle
# --------------------------------------------------------------------------- #
def test_client_is_cached_and_closed(fake_httpx: _FakeHttpxModule) -> None:
    target = HttpTarget(_config())
    c1 = target._get_client()
    c2 = target._get_client()
    assert c1 is c2  # cached
    target.close()
    assert c1.closed is True
    assert target._client is None


def test_context_manager_closes(fake_httpx: _FakeHttpxModule) -> None:
    with HttpTarget(_config()) as target:
        client = target._get_client()
    assert client.closed is True


def test_implements_target_protocol() -> None:
    from injectkit.targets.base import Target

    target = HttpTarget(_config())
    assert isinstance(target, Target)
