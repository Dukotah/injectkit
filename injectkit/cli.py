"""injectkit command-line interface.

The CLI is the primary way people run injectkit, including as a CI gate. It wires
the configuration loader, the corpus loader, a target adapter, the detectors, the
:class:`~injectkit.engine.Engine`, and a reporter together into three
subcommands:

  * ``scan`` — load the corpus, run every attack against the configured target,
    render a report, and exit non-zero when any finding meets the ``--fail-on``
    severity threshold (the CI gate).
  * ``list`` — list the attacks in the corpus (optionally filtered by technique),
    so users can see what will run without sending anything.
  * ``init`` — write a starter ``.injectkit.yaml`` so users can configure a
    target without reading the docs.

DEFENSIVE / AUTHORIZED USE ONLY. injectkit scans LLM endpoints you own or are
explicitly authorized to test — the "scan your own site" posture. The
authorized-use notice appears in ``--help`` and in every rendered report. Do not
point it at third parties.

Heavy/optional SDKs (anthropic, mcp, httpx) are imported lazily by their
adapters, so importing this module and running ``list``/``init`` never requires
them. A missing optional dependency surfaces as a clear, friendly error only
when a scan actually needs it.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .config import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_JUDGE_MODEL,
    Config,
    load_config,
)
from .corpus import CorpusError, load_corpus
from .engine import Engine, ScanError
from .models import Attack, ScanReport, Severity, SEVERITY_ORDER
from .reporters.base import AUTHORIZED_USE_NOTICE

__all__ = ["main", "build_parser"]


# The authorized-use banner shown in --help (and echoed in every report).
_EPILOG = (
    "DEFENSIVE / AUTHORIZED USE ONLY\n"
    "  injectkit scans LLM endpoints you OWN or are EXPLICITLY AUTHORIZED to "
    "test\n"
    "  (the \"scan your own site\" posture). Do not target third parties. "
    "MIT licensed.\n"
)

# Valid --format values mapped to their reporter classes. Reporters are imported
# lazily inside the factory so `list`/`init` never import rich/etc. unnecessarily
# (rich is core, but keeping the import local keeps the surface tidy and fast).
_REPORT_FORMATS = ("terminal", "json", "markdown", "sarif", "html")

# Valid --target kinds the CLI knows how to construct.
_TARGET_KINDS = ("http", "anthropic", "mcp", "mock")

# Exit codes (documented so CI authors can rely on them).
EXIT_OK = 0  # scan ran; no finding met --fail-on
EXIT_FINDINGS = 1  # scan ran; at least one finding met/exceeded --fail-on
EXIT_ERROR = 2  # the scan could not run (config/corpus/target setup error)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the injectkit CLI."""
    parser = argparse.ArgumentParser(
        prog="injectkit",
        description=(
            "Red-team your own LLM application for prompt injection. "
            "Loads a data-driven attack corpus, sends each attack to a target "
            "you control, and grades whether the injection succeeded."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"injectkit {__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="{scan,list,init,gui}")
    sub.required = True

    _add_scan_parser(sub)
    _add_list_parser(sub)
    _add_init_parser(sub)
    _add_gui_parser(sub)
    return parser


def _add_common_corpus_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by subcommands that load the corpus."""
    p.add_argument(
        "--config",
        metavar="FILE",
        help=f"Path to a {DEFAULT_CONFIG_FILENAME} config file "
        f"(default: {DEFAULT_CONFIG_FILENAME} in the current directory).",
    )
    p.add_argument(
        "--corpus",
        metavar="PATH",
        help="Attack corpus file or directory (default: the bundled corpus).",
    )
    p.add_argument(
        "--technique",
        metavar="NAME",
        action="append",
        help="Only include attacks of this technique (repeatable; also "
        "matches an attack tag). Default: all techniques.",
    )


def _add_scan_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scan",
        help="Scan a target for prompt injection.",
        description="Run the attack corpus against a target and report findings.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_corpus_args(p)
    p.add_argument(
        "--target",
        choices=_TARGET_KINDS,
        help="Target adapter kind. Overrides the config's target.kind.",
    )
    p.add_argument(
        "--url",
        help="Endpoint URL for the http target (or HTTP MCP server URL).",
    )
    p.add_argument(
        "--model",
        help="Model id for the target (anthropic default: claude-opus-4-8).",
    )
    p.add_argument(
        "--system",
        help="Default system prompt to send with attacks that don't carry one.",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        default=None,  # None => 'not specified' so it doesn't clobber the config
        help="Enable the optional LLM judge (needs the 'anthropic' SDK + an API "
        "key). Off by default; the offline heuristics always run.",
    )
    p.add_argument(
        "--judge-model",
        metavar="MODEL",
        help=f"Judge model id (default: {DEFAULT_JUDGE_MODEL}).",
    )
    p.add_argument(
        "--fail-on",
        choices=SEVERITY_ORDER,
        help="Minimum finding severity that makes the scan exit non-zero "
        "(the CI gate). Default: high. Use 'critical' for a lenient gate or "
        "'info' to fail on any finding.",
    )
    p.add_argument(
        "--format",
        choices=_REPORT_FORMATS,
        help="Report format (default: terminal).",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        help="Write the report to this file instead of stdout.",
    )


def _add_list_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "list",
        help="List the attacks in the corpus.",
        description="List corpus attacks (id, technique, severity, name) "
        "without sending anything to a target.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_corpus_args(p)


def _add_init_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "init",
        help=f"Write a starter {DEFAULT_CONFIG_FILENAME}.",
        description=f"Write a commented starter {DEFAULT_CONFIG_FILENAME} you "
        "can edit to point injectkit at your target.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=DEFAULT_CONFIG_FILENAME,
        help=f"Where to write the config (default: ./{DEFAULT_CONFIG_FILENAME}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file.",
    )


def _add_gui_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gui",
        help="Launch the local web GUI in your browser (point-and-shoot).",
        description="Start the injectkit local web GUI (localhost only): pick a "
        "target, choose techniques, click Run scan — no API key needed for the "
        "offline mock target + heuristics.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1, localhost only).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765).",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open a browser window.",
    )


# --------------------------------------------------------------------------- #
# Config assembly from CLI args
# --------------------------------------------------------------------------- #
def _build_cli_overrides(args: argparse.Namespace) -> dict:
    """Turn parsed scan args into the cli_overrides dict load_config expects.

    Only non-None values are forwarded so an absent flag never clobbers a value
    set in the config file (load_config also ignores Nones defensively).
    Target-specific flags are nested under a ``target`` dict.
    """
    target_overrides: dict = {}
    for cli_name, tc_field in (
        ("target", "kind"),
        ("url", "url"),
        ("model", "model"),
        ("system", "system"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            target_overrides[tc_field] = value

    overrides: dict = {"target": target_overrides}
    if getattr(args, "corpus", None) is not None:
        overrides["corpus_path"] = args.corpus
    if getattr(args, "judge", None):
        overrides["use_judge"] = True
    if getattr(args, "judge_model", None) is not None:
        overrides["judge_model"] = args.judge_model
    if getattr(args, "fail_on", None) is not None:
        overrides["fail_on"] = args.fail_on
    if getattr(args, "format", None) is not None:
        overrides["report_format"] = args.format
    if getattr(args, "out", None) is not None:
        overrides["out_path"] = args.out
    if getattr(args, "technique", None):
        overrides["techniques"] = list(args.technique)
    return overrides


def _load_config_for(args: argparse.Namespace) -> Config:
    """Load + merge config for a corpus-using subcommand."""
    return load_config(
        config_path=getattr(args, "config", None),
        cli_overrides=_build_cli_overrides(args),
    )


# --------------------------------------------------------------------------- #
# Corpus loading + filtering
# --------------------------------------------------------------------------- #
def _load_attacks(config: Config) -> list[Attack]:
    """Load the corpus and apply the technique/tag filter from config."""
    attacks = load_corpus(config.resolved_corpus_path())
    return _filter_attacks(attacks, config.techniques)


def _filter_attacks(
    attacks: Sequence[Attack], techniques: Optional[Sequence[str]]
) -> list[Attack]:
    """Keep attacks whose technique (or a tag) matches any requested name.

    Matching is case-insensitive. ``None``/empty means keep everything.
    """
    if not techniques:
        return list(attacks)
    wanted = {t.strip().lower() for t in techniques if t and t.strip()}
    if not wanted:
        return list(attacks)
    kept: list[Attack] = []
    for a in attacks:
        names = {a.technique.lower(), *(tag.lower() for tag in a.tags)}
        if names & wanted:
            kept.append(a)
    return kept


# --------------------------------------------------------------------------- #
# Target + detector + reporter construction
# --------------------------------------------------------------------------- #
def _build_target(config: Config):
    """Construct a Target from the resolved config.

    Adapters lazy-import their SDKs, so the import here is cheap; a missing
    optional dependency surfaces only when the target is actually used.
    """
    kind = (config.target.kind or "anthropic").lower()
    if kind == "http":
        from .targets.http import HttpTarget

        return HttpTarget(config.target)
    if kind == "anthropic":
        from .targets.anthropic_target import AnthropicTarget

        return AnthropicTarget.from_config(config.target)
    if kind == "mcp":
        from .targets.mcp import MCPTarget

        return MCPTarget.from_config(config.target)
    if kind == "mock":
        # An offline, deterministic target for demos/CI smoke tests. Imported
        # from the engine-agnostic helper so it works without a network.
        return _build_mock_target(config)
    raise ScanError(
        f"Unknown target kind {kind!r}. Choose one of: {', '.join(_TARGET_KINDS)}."
    )


def _build_mock_target(config: Config):
    """Build the built-in offline MockTarget (mirrors the test fixture).

    Defined here (not imported from tests) so the shipped CLI can run a fully
    offline self-scan without the test package. Behavior matches the brief's
    MockTarget contract closely enough for a smoke test / demo.
    """
    import re

    from .models import TargetResponse

    trigger = "INJECTOK-"
    default_system = config.target.system

    class _MockTarget:
        name = config.target.name if config.target.name != "target" else "mock"

        def send(self, prompt, system=None, context=None):
            effective_system = system if system is not None else default_system
            combined = "\n".join(p for p in (context, prompt) if p)
            if effective_system and re.search(
                r"system prompt|verbatim|configuration token", prompt, re.I
            ):
                return TargetResponse(
                    text=f"Sure, here is my system prompt:\n{effective_system}",
                    stop_reason="end_turn",
                    model="mock",
                )
            if trigger in combined:
                m = re.search(re.escape(trigger) + r"[A-Za-z0-9_-]+", combined)
                return TargetResponse(
                    text=m.group(0) if m else trigger,
                    stop_reason="end_turn",
                    model="mock",
                )
            return TargetResponse(
                text="Here is a helpful, on-task answer.",
                stop_reason="end_turn",
                model="mock",
            )

    return _MockTarget()


def _build_detectors(config: Config) -> list:
    """Build the detector list: offline heuristics always, judge if enabled."""
    from .evaluators.heuristics import HeuristicDetector

    detectors: list = [HeuristicDetector()]
    if config.use_judge:
        from .evaluators.judge import JudgeDetector

        detectors.append(JudgeDetector(model=config.judge_model))
    return detectors


def _build_reporter(report_format: str):
    """Construct the reporter matching ``report_format``."""
    fmt = (report_format or "terminal").lower()
    if fmt == "terminal":
        from .reporters.terminal import TerminalReporter

        return TerminalReporter()
    if fmt == "json":
        from .reporters.json import JSONReporter

        return JSONReporter()
    if fmt == "markdown":
        from .reporters.markdown import MarkdownReporter

        return MarkdownReporter()
    if fmt == "sarif":
        from .reporters.sarif import SarifReporter

        return SarifReporter()
    if fmt == "html":
        from .reporters.html import HtmlReporter

        return HtmlReporter()
    raise ScanError(
        f"Unknown report format {fmt!r}. Choose one of: "
        f"{', '.join(_REPORT_FORMATS)}."
    )


# --------------------------------------------------------------------------- #
# CI gate
# --------------------------------------------------------------------------- #
def _meets_fail_threshold(report: ScanReport, fail_on: Severity) -> bool:
    """True if any finding's severity is >= the ``fail_on`` threshold."""
    highest = report.highest_severity
    if highest is None:
        return False
    return highest.rank >= fail_on.rank


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def _cmd_scan(args: argparse.Namespace, *, out: object, err: object) -> int:
    """Run the ``scan`` subcommand. Returns the process exit code."""
    try:
        config = _load_config_for(args)
    except (ValueError, OSError) as exc:
        print(f"injectkit: config error: {exc}", file=err)
        return EXIT_ERROR

    try:
        attacks = _load_attacks(config)
    except CorpusError as exc:
        print(f"injectkit: corpus error: {exc}", file=err)
        return EXIT_ERROR
    except OSError as exc:
        print(f"injectkit: could not read corpus: {exc}", file=err)
        return EXIT_ERROR

    if not attacks:
        print(
            "injectkit: no attacks matched (corpus empty or --technique filter "
            "excluded everything).",
            file=err,
        )
        return EXIT_ERROR

    try:
        target = _build_target(config)
        detectors = _build_detectors(config)
    except (ScanError, ValueError) as exc:
        print(f"injectkit: target setup error: {exc}", file=err)
        return EXIT_ERROR

    engine = Engine(
        target,
        detectors,
        use_judge=config.use_judge,
        tool_version=__version__,
    )
    try:
        report = engine.run(attacks)
    except ScanError as exc:
        print(f"injectkit: {exc}", file=err)
        return EXIT_ERROR

    # Surface errored attacks so an unreachable target is never silently read as
    # "defended". An all-errored scan gets a prominent warning.
    if report.all_errored:
        print(
            f"injectkit: WARNING: target unreachable — all {report.total} "
            "attack(s) errored (no usable responses). The scan could NOT be "
            "graded and is not a pass; check the target URL/credentials.",
            file=err,
        )
    elif report.errored > 0:
        print(
            f"injectkit: warning: {report.errored} attack(s) could not reach the "
            "target (errors); those are not counted as defended.",
            file=err,
        )

    # Render and emit the report.
    try:
        reporter = _build_reporter(config.report_format)
        rendered = reporter.render(report)
    except ScanError as exc:
        print(f"injectkit: report error: {exc}", file=err)
        return EXIT_ERROR

    if config.out_path:
        try:
            _write_text(config.out_path, rendered)
        except OSError as exc:
            print(f"injectkit: could not write report: {exc}", file=err)
            return EXIT_ERROR
        # A short status line to stderr so the file path is visible but the
        # report file itself stays clean for machine consumers.
        print(
            f"injectkit: wrote {config.report_format} report to "
            f"{config.out_path}",
            file=err,
        )
    else:
        print(rendered, file=out)

    if _meets_fail_threshold(report, config.fail_on):
        worst = report.highest_severity
        print(
            f"injectkit: FAIL - {report.failed} finding(s); worst severity "
            f"{worst.value if worst else 'n/a'} >= --fail-on {config.fail_on.value}.",
            file=err,
        )
        return EXIT_FINDINGS
    return EXIT_OK


def _cmd_list(args: argparse.Namespace, *, out: object, err: object) -> int:
    """Run the ``list`` subcommand. Returns the process exit code."""
    try:
        config = _load_config_for(args)
        attacks = _load_attacks(config)
    except CorpusError as exc:
        print(f"injectkit: corpus error: {exc}", file=err)
        return EXIT_ERROR
    except (ValueError, OSError) as exc:
        print(f"injectkit: {exc}", file=err)
        return EXIT_ERROR

    if not attacks:
        print("injectkit: no attacks matched.", file=err)
        return EXIT_ERROR

    print(f"{len(attacks)} attack(s) in corpus:", file=out)
    print("", file=out)
    for a in attacks:
        sev = a.severity.value if isinstance(a.severity, Severity) else str(a.severity)
        print(f"  [{sev:<8}] {a.technique:<20} {a.id}", file=out)
        print(f"             {a.name}", file=out)
    print("", file=out)
    print(AUTHORIZED_USE_NOTICE, file=out)
    return EXIT_OK


def _cmd_init(args: argparse.Namespace, *, out: object, err: object) -> int:
    """Run the ``init`` subcommand: write a starter config. Returns exit code."""
    out_path = args.out or DEFAULT_CONFIG_FILENAME
    if os.path.exists(out_path) and not args.force:
        print(
            f"injectkit: {out_path} already exists. Use --force to overwrite.",
            file=err,
        )
        return EXIT_ERROR
    try:
        _write_text(out_path, _STARTER_CONFIG)
    except OSError as exc:
        print(f"injectkit: could not write {out_path}: {exc}", file=err)
        return EXIT_ERROR
    print(f"injectkit: wrote starter config to {out_path}", file=out)
    print(
        "Edit it to point at a target you own or are authorized to test, then "
        "run:  injectkit scan",
        file=out,
    )
    return EXIT_OK


def _cmd_gui(args: argparse.Namespace, *, out: object, err: object) -> int:
    """Run the ``gui`` subcommand: launch the local web UI. Returns exit code.

    Imported lazily so importing the CLI (for ``list``/``init``) never pulls in
    the web server. ``web.serve`` blocks until Ctrl-C; a clean stop returns 0.
    """
    from . import web

    web.serve(args.host, args.port, open_browser=not args.no_open)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, creating parent dirs as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


#: A commented starter .injectkit.yaml written by `injectkit init`.
_STARTER_CONFIG = """\
# injectkit configuration.
#
# DEFENSIVE / AUTHORIZED USE ONLY: only scan an endpoint you OWN or are
# explicitly AUTHORIZED to test. Do not target third parties.
#
# Run a scan with:  injectkit scan
# List the attacks with:  injectkit list

target:
  # kind: one of anthropic | http | mcp | mock
  kind: anthropic
  # Display name shown in the report header.
  name: my-app
  # --- anthropic target ---
  # model: claude-opus-4-8        # default; set ANTHROPIC_API_KEY in your env
  # system: "You are a helpful assistant."  # your app's default system prompt

  # --- http target (uncomment and set kind: http) ---
  # url: https://your-app.example.com/api/chat
  # method: POST
  # headers:
  #   Authorization: "Bearer ${YOUR_TOKEN}"
  # request_template:
  #   messages:
  #     - role: system
  #       content: "{system}"
  #     - role: user
  #       content: "{prompt}"
  # response_path: choices.0.message.content

  # --- mcp target (uncomment and set kind: mcp) ---
  # mcp_command: python
  # mcp_args: ["-m", "your_mcp_server"]
  # or, for an HTTP MCP server:
  # mcp_url: http://localhost:8000/mcp

# Path to a custom corpus file/dir. Omit to use the bundled attack corpus.
# corpus_path: ./my-attacks

# Optional LLM judge for sharper grading (needs the 'anthropic' SDK + key).
use_judge: false
# judge_model: claude-haiku-4-5

# CI gate: exit non-zero if any finding is at least this severe.
# One of: info | low | medium | high | critical
fail_on: high

# Report format: terminal | json | markdown | sarif | html
report_format: terminal
# out_path: injectkit-report.json   # write to a file instead of stdout

# Only run these techniques/tags (omit for all):
# techniques: [direct_injection, system_prompt_leak]
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (also usable via sys.exit).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). Injectable so tests
            can drive the CLI in-process without spawning a subprocess.

    Returns:
        ``0`` clean, ``1`` findings met ``--fail-on``, ``2`` a setup/run error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    out = sys.stdout
    err = sys.stderr

    if args.command == "scan":
        return _cmd_scan(args, out=out, err=err)
    if args.command == "list":
        return _cmd_list(args, out=out, err=err)
    if args.command == "init":
        return _cmd_init(args, out=out, err=err)
    if args.command == "gui":
        return _cmd_gui(args, out=out, err=err)

    # argparse enforces a subcommand (sub.required = True), so this is unreachable
    # in practice; kept as a defensive fallback.
    parser.print_help(err)  # pragma: no cover
    return EXIT_ERROR  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
