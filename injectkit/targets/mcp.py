"""MCP (Model Context Protocol) target adapter — tool-abuse & exfil testing.

This adapter connects to an MCP server (over stdio or streamable HTTP) using the
official ``mcp`` SDK and *exercises its tools* with an attack payload. The goal
is to surface two classes of injection vulnerability in agent/tool surfaces:

  * **Tool abuse** — does crafted untrusted input cause a tool to be invoked
    (or invoked with attacker-chosen arguments) when it should not be?
  * **Data exfiltration** — does the canary marker leak *out* through a tool
    call argument or come *back* through a tool's output, proving the agent's
    tools can be steered to move data?

Because injectkit's engine has no LLM-in-the-loop for raw MCP servers, this
adapter models the "agent" deterministically: it discovers the server's tools,
then attempts to call each one with arguments seeded from the rendered attack
payload (filling string parameters with the payload so the canary travels into
tool arguments). The complete trace — which tools were discovered, which were
called, the arguments used, and every tool's textual output — is flattened into
:attr:`TargetResponse.text` so the offline heuristic detectors (marker echo,
canary-in-output, regex) can inspect it exactly as they would a chat reply. The
structured trace is also preserved in :attr:`TargetResponse.raw` for reporters.

The ``mcp`` SDK is **lazy-imported** inside this module so importing injectkit's
core never requires it; a clear, friendly error is raised if the adapter is used
without the dependency installed. The SDK is async (anyio-based); ``send`` is a
synchronous Target method, so it drives the async client on a private event loop
via :func:`anyio.run`.

AUTHORIZED USE ONLY: only point this adapter at MCP servers you own or are
explicitly authorized to test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..models import TargetConfig, TargetResponse

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from mcp import ClientSession

__all__ = ["MCPTarget", "MCPDependencyError"]


# Maximum number of tool outputs / argument blobs we fold into the response
# text, to keep the trace bounded for very chatty servers.
_MAX_TRACE_CHARS = 20_000


class MCPDependencyError(RuntimeError):
    """Raised when the MCP adapter is used without the optional ``mcp`` SDK."""


def _require_mcp() -> Any:
    """Lazy-import the ``mcp`` SDK, raising a friendly error if it is missing.

    Returns:
        The imported ``mcp`` top-level module.

    Raises:
        MCPDependencyError: If the ``mcp`` package is not installed.
    """
    try:
        import mcp  # noqa: F401  (imported for side-effect/availability check)

        return mcp
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MCPDependencyError(
            "The MCP target requires the optional 'mcp' SDK, which is not "
            "installed. Install it with:  pip install 'injectkit[mcp]'  "
            "(or  pip install mcp )."
        ) from exc


def _text_from_content(content: Any) -> str:
    """Flatten an MCP tool-result ``content`` list into plain text.

    MCP tool results carry a list of content blocks (text, image, embedded
    resource, ...). For detection we only care about textual data, so we pull
    ``.text`` off any block that has it and join the rest by ``str()``.

    Args:
        content: The ``content`` attribute of a ``CallToolResult`` (a list of
            content blocks), or anything list-like / None.

    Returns:
        The concatenated textual representation of the content blocks.
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        # Embedded resource? Try its nested text, else fall back to repr.
        resource = getattr(block, "resource", None)
        res_text = getattr(resource, "text", None)
        if isinstance(res_text, str):
            parts.append(res_text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _seed_arguments(input_schema: Optional[dict[str, Any]], payload: str) -> dict[str, Any]:
    """Build tool-call arguments from a JSON Schema, seeding strings with the payload.

    We want the attack payload (carrying the per-run canary) to flow into every
    string-typed tool argument, so that a server which blindly trusts its inputs
    will echo the canary back through its tool output (data exfiltration) — and
    so that required parameters are satisfied enough for the call to execute.

    Args:
        input_schema: The tool's ``inputSchema`` (JSON Schema dict), or None.
        payload: The rendered attack payload (canary already substituted).

    Returns:
        A mapping of argument name -> seeded value suitable for ``call_tool``.
    """
    if not isinstance(input_schema, dict):
        # No schema we can reason about — pass the payload under a generic key.
        return {"input": payload}

    props = input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return {"input": payload}

    required = input_schema.get("required") or []
    args: dict[str, Any] = {}
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        jtype = spec.get("type", "string")
        # Only bother filling required params plus string params; required ones
        # must be present, and string ones are our exfil vector.
        if jtype == "string":
            args[name] = payload
        elif name in required:
            args[name] = _default_for_type(jtype)
    # Guarantee at least one argument carries the payload so the canary travels.
    if payload not in args.values():
        # Prefer an existing string slot; else add a generic one.
        str_slots = [n for n, s in props.items() if isinstance(s, dict) and s.get("type", "string") == "string"]
        args[str_slots[0] if str_slots else "input"] = payload
    return args


def _default_for_type(jtype: str) -> Any:
    """Return a benign default value for a non-string JSON Schema type."""
    return {
        "integer": 0,
        "number": 0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(jtype, "")


class MCPTarget:
    """A :class:`~injectkit.targets.base.Target` backed by an MCP server.

    Connects to an MCP server (stdio subprocess or streamable-HTTP URL), lists
    its tools, and calls them with payload-seeded arguments to probe for
    tool-abuse and data-exfiltration injection. The resulting tool-call trace is
    returned as a :class:`TargetResponse` for the detectors to grade.

    Construct directly or via :meth:`from_config`. Exactly one transport must be
    configured: either ``command`` (stdio) or ``url`` (HTTP).

    Args:
        command: Executable to launch the MCP server over stdio (e.g. "python").
        args: Arguments for the stdio command.
        url: Streamable-HTTP endpoint of a running MCP server.
        headers: Optional HTTP headers (e.g. auth) for the URL transport.
        name: Display name shown in reports.
        timeout_s: Per-operation timeout, in seconds.
        env: Optional environment overrides for the stdio subprocess.

    Raises:
        ValueError: If neither or both transports are configured.
    """

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
        url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        name: Optional[str] = None,
        timeout_s: float = 30.0,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        if bool(command) == bool(url):
            raise ValueError(
                "MCPTarget needs exactly one transport: set either `command` "
                "(stdio) or `url` (HTTP), not both and not neither."
            )
        self.command = command
        self.args = list(args or [])
        self.url = url
        self.headers = dict(headers or {})
        self.timeout_s = timeout_s
        self.env = dict(env) if env else None
        if name:
            self.name = name
        elif command:
            self.name = "mcp:" + " ".join([command, *self.args]).strip()
        else:
            self.name = f"mcp:{url}"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: TargetConfig) -> "MCPTarget":
        """Build an :class:`MCPTarget` from a :class:`TargetConfig`.

        Reads ``mcp_command`` / ``mcp_args`` / ``mcp_url`` (plus ``headers``,
        ``name``, ``timeout_s``) off the config. ``extra`` may carry an ``env``
        dict for the stdio transport.
        """
        return cls(
            command=config.mcp_command,
            args=config.mcp_args,
            url=config.mcp_url,
            headers=config.headers,
            name=config.name if config.name and config.name != "target" else None,
            timeout_s=config.timeout_s,
            env=config.extra.get("env") if isinstance(config.extra, dict) else None,
        )

    # ------------------------------------------------------------------ #
    # Target protocol
    # ------------------------------------------------------------------ #

    def send(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[str] = None,
    ) -> TargetResponse:
        """Probe the MCP server's tools with the attack payload.

        The ``prompt`` (and any ``context``) is used to seed tool-call arguments
        so the canary travels into tool inputs; the full discovery/call trace is
        returned as the response text. ``system`` is recorded but has no native
        analogue on a raw MCP server. Never raises on transport/SDK error — the
        error is captured in :attr:`TargetResponse.error`.

        Args:
            prompt: Rendered attack payload (canary already substituted).
            system: Optional system prompt (recorded only; no MCP analogue).
            context: Optional untrusted context; folded into the seed payload so
                indirect-injection canaries also reach tool arguments.

        Returns:
            A :class:`TargetResponse` whose ``text`` is the flattened tool-call
            trace **of what the server returned** (tool outputs only — the
            arguments we sent in are excluded so the detectors do not flag the
            marker we ourselves injected), and whose ``raw`` holds the full
            structured trace (including the sent arguments) for reporters.
        """
        # Untrusted context (e.g. a simulated retrieved document) should also be
        # able to carry the injection into tool arguments, so combine it.
        seed = "\n".join(p for p in (context, prompt) if p)

        try:
            mcp = _require_mcp()
        except MCPDependencyError as exc:
            return TargetResponse(text="", error=str(exc), model=self.name)

        try:
            import anyio
        except ImportError:  # pragma: no cover - anyio ships with mcp
            return TargetResponse(
                text="",
                error="The MCP target requires 'anyio' (installed with mcp).",
                model=self.name,
            )

        try:
            trace = anyio.run(self._run_probe, seed, mcp)
        except Exception as exc:  # noqa: BLE001 - never let transport errors escape
            return TargetResponse(
                text="",
                error=f"{type(exc).__name__}: {exc}",
                model=self.name,
                raw={"transport": "stdio" if self.command else "http"},
            )

        text = self._render_trace(trace)
        return TargetResponse(
            text=text[:_MAX_TRACE_CHARS],
            refused=False,
            stop_reason="tool_use" if trace.get("calls") else "end_turn",
            model=self.name,
            raw=trace,
        )

    # ------------------------------------------------------------------ #
    # Async probe driver
    # ------------------------------------------------------------------ #

    async def _run_probe(self, seed: str, mcp: Any) -> dict[str, Any]:
        """Open a session, list tools, and call each with payload-seeded args.

        Returns a structured trace dict with keys ``tools`` (discovered tool
        names), ``calls`` (per-call records), and ``transport``.
        """
        async with self._open_session(mcp) as session:
            await session.initialize()
            return await self._exercise_tools(session, seed)

    def _open_session(self, mcp: Any) -> Any:
        """Return an async context manager yielding an initialized-able session.

        Picks the stdio or streamable-HTTP transport based on configuration and
        wraps it so callers can ``async with self._open_session(mcp) as session``.
        """
        from contextlib import asynccontextmanager

        from mcp import ClientSession

        timeout = self.timeout_s

        @asynccontextmanager
        async def _ctx() -> Any:
            if self.command:
                from mcp import StdioServerParameters, stdio_client

                params = StdioServerParameters(
                    command=self.command,
                    args=self.args,
                    env=self.env,
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        yield session
            else:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    self.url,
                    headers=self.headers or None,
                    timeout=timeout,
                ) as (read, write, _get_session_id):
                    async with ClientSession(read, write) as session:
                        yield session

        return _ctx()

    async def _exercise_tools(self, session: "ClientSession", seed: str) -> dict[str, Any]:
        """Discover and invoke each tool, seeding string args with ``seed``."""
        trace: dict[str, Any] = {
            "transport": "stdio" if self.command else "http",
            "tools": [],
            "calls": [],
        }

        listed = await session.list_tools()
        tools = list(getattr(listed, "tools", []) or [])
        trace["tools"] = [getattr(t, "name", "?") for t in tools]

        for tool in tools:
            tool_name = getattr(tool, "name", "?")
            schema = getattr(tool, "inputSchema", None)
            arguments = _seed_arguments(schema, seed)
            call_record: dict[str, Any] = {
                "tool": tool_name,
                "arguments": arguments,
                "output": "",
                "is_error": False,
                "error": None,
            }
            try:
                result = await session.call_tool(tool_name, arguments)
                call_record["output"] = _text_from_content(getattr(result, "content", None))
                call_record["is_error"] = bool(getattr(result, "isError", False))
                structured = getattr(result, "structuredContent", None)
                if structured is not None:
                    call_record["structured"] = structured
            except Exception as exc:  # noqa: BLE001 - record, keep probing others
                call_record["error"] = f"{type(exc).__name__}: {exc}"
            trace["calls"].append(call_record)

        return trace

    # ------------------------------------------------------------------ #
    # Trace rendering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_trace(trace: dict[str, Any]) -> str:
        """Flatten a structured probe trace into detector-inspectable text.

        CRITICAL DETECTION INVARIANT: this text is what the offline detectors
        grade, and detection success means *the server reflected the canary/marker
        back out through a tool*. Therefore the request **arguments we sent in are
        deliberately excluded** from this text — they already contain the rendered
        payload (and thus the marker/canary), so echoing them here would make every
        attack a guaranteed false positive (the detector would "find" the marker we
        ourselves injected, even against a server that ignored the input entirely).

        Only what the *server returned* is rendered: the discovered tool names and,
        per call, the tool's textual output, its structured content, and any
        per-call error. A canary that comes *back* in a tool output is genuine
        evidence of exfiltration/tool-abuse; a canary that merely went *in* is not.
        The full arguments are still preserved in :attr:`TargetResponse.raw` for
        reporters that want to show what was sent.
        """
        import json

        lines: list[str] = []
        tools = trace.get("tools") or []
        lines.append(f"MCP tools discovered: {', '.join(tools) if tools else '(none)'}")
        for call in trace.get("calls") or []:
            lines.append("")
            lines.append(f"[tool call] {call.get('tool', '?')}")
            if call.get("error"):
                lines.append(f"  error: {call['error']}")
            else:
                flag = " (isError)" if call.get("is_error") else ""
                lines.append(f"  output{flag}: {call.get('output', '')}")
                if "structured" in call:
                    try:
                        lines.append(f"  structured: {json.dumps(call['structured'], default=str)}")
                    except (TypeError, ValueError):
                        lines.append(f"  structured: {call['structured']}")
        return "\n".join(lines)
