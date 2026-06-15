"""Generic HTTP chat-endpoint target adapter.

``HttpTarget`` is the broadest, most-adoptable adapter: it can probe *any*
JSON-over-HTTP chat endpoint — your own FastAPI/Flask chatbot, an OpenAI-style
``/v1/chat/completions`` server, a self-hosted inference gateway, etc. — without
needing a provider-specific SDK.

DEFENSIVE / AUTHORIZED USE ONLY. Point this at an endpoint you own or are
explicitly authorized to test.

How it works
------------
You describe the endpoint declaratively with a :class:`TargetConfig`:

* ``url`` / ``method`` / ``headers`` — where and how to send the request.
* ``request_template`` — a JSON-shaped ``dict`` describing the request body.
  Any string anywhere in the template (nested dicts/lists included) may contain
  the placeholders ``{prompt}``, ``{system}``, and ``{context}``; they are
  substituted with the rendered attack payload, the system prompt, and any
  extra untrusted context for this send. A template of ``None`` falls back to a
  simple OpenAI-style ``{"messages": [...]}`` body.
* ``response_path`` — a dotted path (e.g. ``"choices.0.message.content"``) used
  to pull the reply text out of the JSON response. List indices are written as
  plain integers in the path. ``None`` falls back to a list of common reply
  locations.

Templating is intentionally ergonomic: the placeholders are substituted into
the parsed JSON structure (not the raw string), so values that contain quotes,
newlines, or braces never corrupt the JSON. When a string is *exactly* a single
placeholder (e.g. ``"{system}"``) and the value is ``None``, that key is dropped
from the request rather than sent as the literal text ``"None"``.

The adapter never raises on a normal failed request: transport errors, non-2xx
responses, invalid JSON, and missing response paths all come back as a
:class:`TargetResponse` with ``error`` set so the scan can continue.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Optional

from ..models import TargetConfig, TargetResponse

__all__ = ["HttpTarget", "render_template", "extract_path"]

# A single string that is exactly one placeholder, e.g. "{system}". Used to
# decide whether a None-valued placeholder should drop its key entirely.
_SOLE_PLACEHOLDER = re.compile(r"^\{(prompt|system|context)\}$")

# Matches any one of the supported placeholders. Used for a single-pass
# substitution so that a value injected for one placeholder can never be
# re-scanned and have *its* contents treated as another placeholder.
_ANY_PLACEHOLDER = re.compile(r"\{(prompt|system|context)\}")

# Default request body when no template is supplied: an OpenAI-style chat body.
_DEFAULT_TEMPLATE: dict[str, Any] = {
    "messages": [
        {"role": "system", "content": "{system}"},
        {"role": "user", "content": "{prompt}"},
    ]
}

# Response locations tried (in order) when no response_path is configured. These
# cover the most common chat-endpoint shapes.
_DEFAULT_RESPONSE_PATHS: tuple[str, ...] = (
    "choices.0.message.content",  # OpenAI chat completions
    "choices.0.text",  # OpenAI legacy completions
    "content.0.text",  # Anthropic-style
    "message.content",
    "response",
    "reply",
    "text",
    "output",
)


def render_template(template: Any, values: dict[str, Optional[str]]) -> Any:
    """Recursively substitute ``{prompt}`` / ``{system}`` / ``{context}``.

    Substitution happens on the *parsed* structure, so injected text can safely
    contain quotes, braces, or newlines without breaking the JSON body.

    Rules:
      * Any string is scanned for the placeholders and each is replaced with its
        value (``None`` becomes the empty string within a larger string).
      * If a string is *exactly* one placeholder (e.g. ``"{system}"``) and that
        value is ``None``, the value is returned as ``None`` so the caller can
        drop the key (avoids sending the literal ``"None"``).
      * Dicts and lists are walked recursively. Keys whose value renders to a
        bare ``None`` (sole-placeholder case) are omitted from the result dict.

    Args:
        template: The request_template value (dict/list/str/scalar).
        values: Mapping with keys ``"prompt"``, ``"system"``, ``"context"``.

    Returns:
        A new structure with placeholders substituted; the input is not mutated.
    """
    if isinstance(template, str):
        sole = _SOLE_PLACEHOLDER.match(template)
        if sole is not None:
            # Whole string is a single placeholder -> preserve None so the key
            # can be dropped instead of becoming the text "None".
            return values.get(sole.group(1))
        # Inline placeholders inside a larger string. Substitute every
        # placeholder in a SINGLE pass via re.sub so that a value injected for
        # one placeholder (e.g. an attack payload that literally contains the
        # text "{context}") is never re-scanned and expanded. A None value
        # renders as the empty string within a larger string.
        return _ANY_PLACEHOLDER.sub(
            lambda m: values.get(m.group(1)) or "", template
        )

    if isinstance(template, dict):
        rendered: dict[Any, Any] = {}
        for key, val in template.items():
            new_val = render_template(val, values)
            # Drop keys whose sole-placeholder value was None.
            if new_val is None and isinstance(val, str) and _SOLE_PLACEHOLDER.match(val):
                continue
            rendered[key] = new_val
        return rendered

    if isinstance(template, list):
        return [render_template(item, values) for item in template]

    # Scalars (int/float/bool/None) pass through unchanged.
    return template


def extract_path(data: Any, path: str) -> Any:
    """Follow a dotted ``path`` (e.g. ``"choices.0.message.content"``).

    Numeric segments index into lists; everything else is a dict key. Raises
    :class:`KeyError` if the path cannot be fully resolved so the caller can
    surface a clear error.

    Args:
        data: The parsed JSON response.
        path: Dotted path; list indices are plain integers.

    Returns:
        The value at ``path``.
    """
    current = data
    for segment in path.split("."):
        if isinstance(current, list):
            try:
                idx = int(segment)
            except ValueError as exc:
                raise KeyError(
                    f"path segment {segment!r} is not a list index"
                ) from exc
            try:
                current = current[idx]
            except IndexError as exc:
                raise KeyError(f"list index {idx} out of range") from exc
        elif isinstance(current, dict):
            if segment not in current:
                raise KeyError(f"key {segment!r} not found")
            current = current[segment]
        else:
            raise KeyError(
                f"cannot descend into {type(current).__name__} at segment "
                f"{segment!r}"
            )
    return current


class HttpTarget:
    """Probe any JSON-over-HTTP chat endpoint described by a TargetConfig.

    Implements the :class:`~injectkit.targets.base.Target` protocol. ``httpx`` is
    lazy-imported on first send so importing injectkit's core never requires it.

    Args:
        config: A :class:`TargetConfig` with ``kind == "http"``. ``url`` is
            required. ``system`` on the config is the default system prompt used
            when an attack does not carry its own.
    """

    def __init__(self, config: TargetConfig) -> None:
        if not config.url:
            raise ValueError("HttpTarget requires config.url to be set")
        self.config = config
        self.name = config.name or f"http:{config.url}"
        # Default system prompt applied when an attack doesn't supply one.
        self._default_system = config.system
        # Cached httpx client, created lazily on first send.
        self._client: Any = None

    # ---- httpx is optional and heavy: import only when actually sending ----

    def _get_client(self) -> Any:
        """Return a cached httpx.Client, importing httpx lazily.

        Raises a friendly error if httpx is not installed.
        """
        if self._client is not None:
            return self._client
        try:
            import httpx  # noqa: PLC0415 (intentional lazy import)
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise RuntimeError(
                "The HTTP target requires the 'httpx' package. Install it with "
                "`pip install httpx` (or `pip install injectkit[http]`)."
            ) from exc
        self._client = httpx.Client(timeout=self.config.timeout_s)
        return self._client

    def close(self) -> None:
        """Close the underlying httpx client, if one was created."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def __enter__(self) -> "HttpTarget":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- request construction ----

    def _build_body(
        self,
        prompt: str,
        system: Optional[str],
        context: Optional[str],
    ) -> Any:
        """Render the configured (or default) request_template into a body."""
        template = (
            self.config.request_template
            if self.config.request_template is not None
            else _DEFAULT_TEMPLATE
        )
        values: dict[str, Optional[str]] = {
            "prompt": prompt,
            "system": system if system is not None else self._default_system,
            "context": context,
        }
        # Deep-copy so a dict template is never mutated across sends.
        return render_template(copy.deepcopy(template), values)

    def _extract_reply(self, data: Any) -> str:
        """Pull the reply text out of the parsed JSON response.

        Uses the configured ``response_path`` if present, otherwise tries a set
        of common locations. Raises KeyError if nothing matches.
        """
        if self.config.response_path:
            value = extract_path(data, self.config.response_path)
            return value if isinstance(value, str) else json.dumps(value)

        for candidate in _DEFAULT_RESPONSE_PATHS:
            try:
                value = extract_path(data, candidate)
            except KeyError:
                continue
            return value if isinstance(value, str) else json.dumps(value)

        # Last resort: if the whole response is a string, use it.
        if isinstance(data, str):
            return data
        raise KeyError(
            "could not locate the reply text; set response_path to a dotted "
            "path into the JSON response (e.g. 'choices.0.message.content')"
        )

    # ---- the Target protocol ----

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Send one attack to the HTTP endpoint and normalize the reply.

        Never raises on a normal failure: transport errors, non-2xx statuses,
        non-JSON bodies, and unresolved response paths all return a
        :class:`TargetResponse` with ``error`` set.
        """
        body = self._build_body(prompt, system, context)
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return TargetResponse(text="", error=str(exc))

        # httpx is guaranteed importable here (_get_client succeeded); needed
        # for the HTTPError exception type below.
        import httpx  # noqa: PLC0415 (intentional lazy import)

        try:
            resp = client.request(
                self.config.method or "POST",
                self.config.url,
                headers=self.config.headers or None,
                json=body,
            )
        except httpx.HTTPError as exc:
            return TargetResponse(
                text="",
                error=f"HTTP request failed: {type(exc).__name__}: {exc}",
                raw={"request_body": body},
            )

        status = resp.status_code
        if status >= 400:
            return TargetResponse(
                text="",
                error=f"HTTP {status} from endpoint",
                stop_reason=f"http_{status}",
                raw={
                    "status_code": status,
                    "response_text": resp.text[:1000],
                    "request_body": body,
                },
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            return TargetResponse(
                text="",
                error=f"response was not valid JSON: {exc}",
                stop_reason=f"http_{status}",
                raw={"status_code": status, "response_text": resp.text[:1000]},
            )

        try:
            text = self._extract_reply(data)
        except KeyError as exc:
            return TargetResponse(
                text="",
                error=f"could not extract reply from response: {exc}",
                stop_reason=f"http_{status}",
                raw={"status_code": status, "response_json": data},
            )

        return TargetResponse(
            text=text,
            refused=False,
            stop_reason=f"http_{status}",
            model=self.config.model,
            raw={"status_code": status, "request_body": body, "response_json": data},
        )
