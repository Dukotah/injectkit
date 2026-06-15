"""injectkit local web GUI.

A tiny, dependency-free browser front end for injectkit so you can interact with
a scan without the command line. It reuses the exact same engine, corpus, target
adapters, detectors, and HTML reporter the CLI uses — this is a thin web layer,
not a second implementation.

Run it with::

    python -m injectkit.web            # opens http://127.0.0.1:8765 in your browser
    python -m injectkit.web --port 9000 --no-open

Then pick a target (the offline ``mock`` target needs no API key and no network),
choose which attack techniques to run, set the CI fail-on threshold, and click
*Run scan*. The full HTML report renders right in the page.

DEFENSIVE / AUTHORIZED USE ONLY. Only scan endpoints you own or are explicitly
authorized to test. The server binds to localhost only.
"""

from __future__ import annotations

import argparse
import html
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs

from . import __version__
from .config import load_config
from .engine import Engine
from .models import ScanReport

# Reuse the CLI's builders so the GUI and CLI behave identically.
from .cli import _build_detectors, _build_reporter, _build_target, _load_attacks

TECHNIQUES = [
    "direct_injection",
    "indirect_injection",
    "jailbreak",
    "system_prompt_leak",
    "tool_abuse",
    "data_exfiltration",
]
TARGET_KINDS = ["mock", "http", "anthropic", "mcp"]
FAIL_ON = ["info", "low", "medium", "high", "critical"]

# Last rendered HTML report, served at /report and embedded in the results page.
_LAST_REPORT_HTML: Optional[str] = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Scan execution (reuses the CLI pipeline)
# --------------------------------------------------------------------------- #
def run_scan(form: dict[str, list[str]]) -> ScanReport:
    """Build a Config from form fields and run a scan, returning the report."""
    def one(name: str) -> Optional[str]:
        vals = form.get(name)
        val = vals[0].strip() if vals and vals[0].strip() else None
        return val

    target: dict = {"kind": one("kind") or "mock"}
    for f in ("url", "model", "system"):
        if one(f):
            target[f] = one(f)

    techniques = [t for t in form.get("technique", []) if t in TECHNIQUES]

    overrides: dict = {
        "target": target,
        "use_judge": bool(form.get("judge")),
        "fail_on": one("fail_on") or "high",
        "report_format": "html",
    }
    if techniques:
        overrides["techniques"] = techniques

    config = load_config(cli_overrides=overrides)
    attacks = _load_attacks(config)
    if not attacks:
        raise ValueError("No attacks matched your technique selection.")

    target_obj = _build_target(config)
    detectors = _build_detectors(config)
    engine = Engine(
        target_obj,
        detectors,
        use_judge=config.use_judge,
        tool_version=__version__,
    )
    return engine.run(attacks)


# --------------------------------------------------------------------------- #
# HTML pages
# --------------------------------------------------------------------------- #
_STYLE = """
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #0e1116; color: #e6edf3; }
.wrap { max-width: 880px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h1 .v { color: #768390; font-size: 14px; font-weight: 400; }
.sub { color: #adbac7; margin: 0 0 20px; }
.banner { background: #2d2410; border: 1px solid #5c4813; color: #e3b341;
          padding: 10px 14px; border-radius: 8px; font-size: 13px; margin: 0 0 24px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 20px; margin: 0 0 20px; }
label { display: block; font-weight: 600; margin: 14px 0 4px; font-size: 13px; }
input[type=text], select { width: 100%; padding: 8px 10px; border-radius: 6px;
        border: 1px solid #30363d; background: #0e1116; color: #e6edf3; font-size: 14px; }
.row { display: flex; gap: 16px; } .row > div { flex: 1; }
.techs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }
.techs label { font-weight: 400; display: flex; align-items: center; gap: 8px; margin: 0; }
.hint { color: #768390; font-size: 12px; margin: 4px 0 0; }
button { background: #238636; color: #fff; border: 0; padding: 11px 22px;
         border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 22px; }
button:hover { background: #2ea043; }
a { color: #58a6ff; } .chk { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
.summary { display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 18px; }
.stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 18px; min-width: 110px; }
.stat .n { font-size: 26px; font-weight: 700; } .stat .l { color: #768390; font-size: 12px; }
.bad { color: #f85149; } .good { color: #3fb950; } .warn { color: #e3b341; }
.warnbox { background: #2d2410; border: 1px solid #5c4813; color: #e3b341;
           padding: 12px 16px; border-radius: 8px; font-size: 14px; margin: 0 0 18px; }
iframe { width: 100%; height: 1400px; border: 1px solid #30363d; border-radius: 10px; background: #fff; }
.err { background: #2d1416; border: 1px solid #5c1a1f; color: #ff7b72;
       padding: 14px 16px; border-radius: 8px; }
code { background: #0e1116; padding: 1px 5px; border-radius: 4px; }
"""

_BANNER = (
    "&#9888; Defensive / authorized use only — scan only endpoints you own "
    "or are explicitly authorized to test."
)


def _page(body: str) -> bytes:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>injectkit</title><style>{_STYLE}</style></head>"
        f"<body><div class=wrap>{body}</div></body></html>"
    ).encode("utf-8")


def form_page(notice: str = "") -> bytes:
    techs = "".join(
        f"<label><input type=checkbox name=technique value='{t}'> {t}</label>"
        for t in TECHNIQUES
    )
    kinds = "".join(
        f"<option value='{k}'{' selected' if k == 'mock' else ''}>{k}</option>"
        for k in TARGET_KINDS
    )
    fail = "".join(
        f"<option value='{s}'{' selected' if s == 'high' else ''}>{s}</option>"
        for s in FAIL_ON
    )
    return _page(
        f"<h1>injectkit <span class=v>v{__version__}</span></h1>"
        "<p class=sub>Red-team your own LLM app for prompt injection.</p>"
        f"<div class=banner>{_BANNER}</div>"
        f"{notice}"
        "<form method=post action='/scan'><div class=card>"
        "<div class=row>"
        f"<div><label>Target</label><select name=kind>{kinds}</select>"
        "<p class=hint><b>mock</b> = built-in vulnerable demo target (no key, no "
        "network). <b>http</b> = your endpoint URL. <b>anthropic</b> = a Claude "
        "model (needs ANTHROPIC_API_KEY).</p></div>"
        f"<div><label>Fail-on (CI gate)</label><select name=fail_on>{fail}</select>"
        "<p class=hint>Lowest severity that counts as a failed gate.</p></div>"
        "</div>"
        "<label>Endpoint URL <span class=hint>(http target only)</span></label>"
        "<input type=text name=url placeholder='https://your-app.example.com/api/chat'>"
        "<div class=row>"
        "<div><label>Model <span class=hint>(optional)</span></label>"
        "<input type=text name=model placeholder='claude-opus-4-8'></div>"
        "<div><label>System prompt <span class=hint>(optional)</span></label>"
        "<input type=text name=system placeholder=\"You are a helpful assistant.\"></div>"
        "</div>"
        "<label>Techniques <span class=hint>(none = run all 6)</span></label>"
        f"<div class=techs>{techs}</div>"
        "<div class=chk><input type=checkbox name=judge value=1 id=judge>"
        "<label for=judge style='margin:0;font-weight:400'>Use LLM judge "
        "(sharper grading — needs an Anthropic API key; off = fully offline)</label></div>"
        "<button type=submit>Run scan</button>"
        "</div></form>"
        "<p class=hint>Tip: leave everything default and click <b>Run scan</b> to "
        "watch injectkit attack the built-in mock target with zero setup.</p>"
    )


def results_page(report: ScanReport) -> bytes:
    failed = report.failed
    errored = report.errored
    worst = report.highest_severity
    worst_s = worst.value if worst else "none"
    fcls = "bad" if failed else "good"
    ecls = "warn" if errored else ""

    # An all-errored scan never reached the target — say so plainly instead of
    # implying the target defended everything.
    if report.all_errored:
        notice = (
            "<div class=warnbox>&#9888; Target unreachable — all "
            f"{report.total} attack(s) errored (no usable responses). This scan "
            "could not be graded and is <b>not</b> a pass. Check the target URL "
            "or credentials.</div>"
        )
    elif errored:
        notice = (
            f"<div class=warnbox>&#9888; {errored} attack(s) could not reach the "
            "target (errors) and are not counted as defended.</div>"
        )
    else:
        notice = ""

    errored_stat = (
        f"<div class=stat><div class='n {ecls}'>{errored}</div>"
        "<div class=l>errored</div></div>"
        if errored
        else ""
    )

    return _page(
        "<h1>Scan results</h1>"
        f"<p class=sub>Target: <code>{html.escape(report.target_name)}</code>"
        + (f" &middot; <code>{html.escape(report.target_model)}</code>" if report.target_model else "")
        + "</p>"
        f"{notice}"
        "<div class=summary>"
        f"<div class=stat><div class=n>{report.total}</div><div class=l>attacks</div></div>"
        f"<div class=stat><div class='n good'>{report.passed}</div><div class=l>defended</div></div>"
        f"<div class=stat><div class='n {fcls}'>{failed}</div><div class=l>vulnerable</div></div>"
        f"{errored_stat}"
        f"<div class=stat><div class='n {fcls}'>{html.escape(worst_s)}</div><div class=l>worst severity</div></div>"
        "</div>"
        "<p><a href='/'>&larr; New scan</a> &nbsp;&middot;&nbsp; "
        "<a href='/report' target=_blank>open full report in a new tab</a></p>"
        "<iframe src='/report' title='injectkit report'></iframe>"
    )


def error_page(message: str) -> bytes:
    return _page(
        "<h1>Scan failed</h1>"
        f"<div class=err>{html.escape(message)}</div>"
        "<p style='margin-top:18px'><a href='/'>&larr; Back</a></p>"
        "<p class=hint>The <b>mock</b> target always works offline. <b>anthropic</b> "
        "needs <code>ANTHROPIC_API_KEY</code>; <b>http</b> needs a reachable URL.</p>"
    )


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        global _LAST_REPORT_HTML
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(form_page())
        elif path == "/report":
            with _LOCK:
                report = _LAST_REPORT_HTML
            if report is None:
                self._send(form_page("<div class=banner>No scan has run yet.</div>"))
            else:
                self._send(report.encode("utf-8"))
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        global _LAST_REPORT_HTML
        if self.path.split("?", 1)[0] != "/scan":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        form = parse_qs(raw, keep_blank_values=True)
        try:
            report = run_scan(form)
        except Exception as exc:  # surface any failure as a friendly page
            self._send(error_page(f"{type(exc).__name__}: {exc}"))
            return
        reporter = _build_reporter("html")
        with _LOCK:
            _LAST_REPORT_HTML = reporter.render(report)
        self._send(results_page(report))

    def log_message(self, *args) -> None:  # keep the console quiet
        return


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the local GUI server (blocks until Ctrl-C)."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"injectkit GUI running at {url}  (Ctrl-C to stop)")
    print("Defensive / authorized use only — scan only what you own.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m injectkit.web",
        description="Launch the injectkit local web GUI (localhost only).",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default localhost).")
    p.add_argument("--port", type=int, default=8765, help="Port (default 8765).")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open a browser.")
    args = p.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
