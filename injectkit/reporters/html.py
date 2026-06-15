"""Self-contained HTML reporter — the demo / screenshot artifact.

Renders a :class:`~injectkit.models.ScanReport` as a single, standalone HTML
document with all CSS inlined (no external assets, no JS required), so it can be
opened straight from disk, attached to a PR, or published to GitHub Pages.

The report shows:

  * a header with the target name/model, scan timing, and the authorized-use
    notice;
  * a letter grade (A-F) derived from the worst finding and the failure rate;
  * a summary strip (attacks run / defended / vulnerable);
  * a per-technique breakdown table;
  * each finding as a card with severity, confidence, the exact payload sent,
    the response excerpt, and the detector rationale.

Everything user/model-supplied is HTML-escaped. This module is dependency-free
(stdlib only) and imports cleanly anywhere.
"""

from __future__ import annotations

import html
import time
from typing import Optional

from ..models import Finding, ScanReport, Severity
from .base import AUTHORIZED_USE_NOTICE

__all__ = ["HtmlReporter", "grade_report"]

#: Severity -> accent color used for badges and finding borders.
_SEVERITY_COLOR: dict[str, str] = {
    Severity.INFO.value: "#3b82f6",
    Severity.LOW.value: "#10b981",
    Severity.MEDIUM.value: "#f59e0b",
    Severity.HIGH.value: "#f97316",
    Severity.CRITICAL.value: "#ef4444",
}

#: Letter grade -> color, for the big grade dial.
_GRADE_COLOR: dict[str, str] = {
    "A": "#10b981",
    "B": "#84cc16",
    "C": "#f59e0b",
    "D": "#f97316",
    "F": "#ef4444",
    "N/A": "#6b7280",
}


def grade_report(report: ScanReport) -> str:
    """Compute a letter grade (A-F, or N/A) for a scan.

    A clean scan (zero findings) is an ``A``. Otherwise the grade is driven by
    the worst finding's severity, then nudged by how many attacks got through:

      * any CRITICAL finding => F
      * any HIGH finding      => D (or F if >25% of attacks succeeded)
      * worst is MEDIUM       => C
      * worst is LOW          => B
      * worst is INFO only    => B

    The failure rate can pull a borderline grade down one notch so a target that
    fails *most* attacks never scores well even at lower severities.

    A scan where every attack errored (the target was never reached) grades as
    ``N/A`` — an unreachable target is not a defended one and must never read as
    a clean ``A``.
    """
    if report.all_errored:
        return "N/A"
    if not report.findings:
        return "A"

    worst = report.highest_severity or Severity.INFO
    fail_rate = (report.failed / report.total) if report.total else 0.0

    if worst.rank >= Severity.CRITICAL.rank:
        return "F"
    if worst.rank >= Severity.HIGH.rank:
        return "F" if fail_rate > 0.25 else "D"
    if worst.rank >= Severity.MEDIUM.rank:
        return "D" if fail_rate > 0.5 else "C"
    # LOW or INFO worst
    return "C" if fail_rate > 0.5 else "B"


def _fmt_time(ts: Optional[float]) -> str:
    """Format an epoch timestamp as a readable UTC string, or '—' if None."""
    if ts is None:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def _esc(value: object) -> str:
    """HTML-escape any value's string form (quotes included)."""
    return html.escape(str(value), quote=True)


def _errored_card(report: ScanReport) -> str:
    """A summary card for errored attacks, shown only when some errored."""
    if not report.errored:
        return ""
    return (
        '\n    <div class="card"><div class="n" style="color:#b45309">'
        f"{report.errored}</div>"
        '\n      <div class="l">errored</div></div>'
    )


#: URL schemes safe to emit in an ``href``. Anything else (notably
#: ``javascript:`` and ``data:``) is dropped: corpus ``references`` come from
#: community-contributed YAML, and ``html.escape`` does NOT neutralize a
#: ``javascript:`` URL — it contains no HTML metacharacters — so an escaped-but-
#: live ``href="javascript:..."`` would be a clickable XSS vector in a report
#: that gets attached to PRs and published to GitHub Pages.
_SAFE_URL_SCHEMES = ("http://", "https://", "mailto:")


def _safe_href(url: str) -> Optional[str]:
    """Return ``url`` if it uses a safe scheme, else ``None``.

    Accepts absolute ``http(s)``/``mailto`` URLs and scheme-relative/relative
    references (``//host``, ``/path``, ``./path``, ``#frag``); rejects anything
    with a dangerous or unknown scheme (``javascript:``, ``data:``, ``vbscript:``,
    ``file:``, ...). Leading control characters/whitespace are stripped first so
    ``"java\\tscript:..."`` cannot slip past.
    """
    if not url:
        return None
    # Strip ASCII control chars and whitespace browsers ignore inside schemes.
    cleaned = "".join(ch for ch in str(url) if ord(ch) > 0x20).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.startswith(_SAFE_URL_SCHEMES):
        return cleaned
    # No scheme present (no ':' before the first '/', '?', or '#') => relative,
    # which is safe. A ':' before any path separator means an explicit scheme we
    # did not allowlist, so reject it.
    for ch in cleaned:
        if ch == ":":
            return None
        if ch in "/?#":
            break
    return cleaned


def _severity_badge(severity: Severity) -> str:
    """Inline-styled severity pill."""
    sev = Severity.coerce(severity)
    color = _SEVERITY_COLOR.get(sev.value, "#6b7280")
    return (
        f'<span class="badge" style="background:{color}">'
        f"{_esc(sev.value.upper())}</span>"
    )


def _technique_rows(report: ScanReport) -> str:
    """Build the per-technique breakdown table body.

    Counts every attack run per technique (from results) and how many produced a
    finding, plus the worst severity seen for that technique.
    """
    total_by_tech: dict[str, int] = {}
    errored_by_tech: dict[str, int] = {}
    for r in report.results:
        total_by_tech[r.attack.technique] = total_by_tech.get(r.attack.technique, 0) + 1
        if r.response.error is not None:
            errored_by_tech[r.attack.technique] = (
                errored_by_tech.get(r.attack.technique, 0) + 1
            )

    fail_by_tech: dict[str, int] = {}
    worst_by_tech: dict[str, Severity] = {}
    for f in report.findings:
        fail_by_tech[f.technique] = fail_by_tech.get(f.technique, 0) + 1
        sev = Severity.coerce(f.severity)
        if f.technique not in worst_by_tech or sev.rank > worst_by_tech[f.technique].rank:
            worst_by_tech[f.technique] = sev

    # Include techniques that ran even if they produced no findings.
    techniques = sorted(set(total_by_tech) | set(fail_by_tech))
    if not techniques:
        return (
            '<tr><td colspan="4" class="muted">No attacks were run.</td></tr>'
        )

    rows: list[str] = []
    for tech in techniques:
        total = total_by_tech.get(tech, 0)
        failed = fail_by_tech.get(tech, 0)
        errored = errored_by_tech.get(tech, 0)
        worst = worst_by_tech.get(tech)
        worst_cell = _severity_badge(worst) if worst else '<span class="muted">—</span>'
        if failed:
            status_class = "fail"
            status = f"{failed} vulnerable"
            if errored:
                status += f", {errored} errored"
        elif errored and errored == total:
            # No usable response at all for this technique — not a defense.
            status_class = "muted"
            status = f"{errored} errored (unreachable)"
        elif errored:
            status_class = "pass"
            status = f"defended ({errored} errored)"
        else:
            status_class = "pass"
            status = "defended"
        rows.append(
            "<tr>"
            f"<td>{_esc(tech.replace('_', ' '))}</td>"
            f"<td>{total}</td>"
            f'<td class="{status_class}">{_esc(status)}</td>'
            f"<td>{worst_cell}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _finding_card(finding: Finding) -> str:
    """Render one finding as a styled card."""
    sev = Severity.coerce(finding.severity)
    color = _SEVERITY_COLOR.get(sev.value, "#6b7280")

    refs = ""
    if finding.references:
        item_html: list[str] = []
        for u in finding.references:
            href = _safe_href(u)
            if href is not None:
                # Safe scheme: render a clickable link.
                item_html.append(
                    f'<li><a href="{_esc(href)}" rel="noreferrer noopener">'
                    f"{_esc(u)}</a></li>"
                )
            else:
                # Unsafe/unknown scheme: show the text but never make it clickable.
                item_html.append(f"<li>{_esc(u)}</li>")
        items = "\n".join(item_html)
        refs = f'<div class="refs"><strong>References</strong><ul>{items}</ul></div>'

    tags = ""
    if finding.tags:
        chips = "".join(f'<span class="chip">{_esc(t)}</span>' for t in finding.tags)
        tags = f'<div class="tags">{chips}</div>'

    rationale = ""
    if finding.rationale:
        rationale = (
            f'<div class="field"><div class="label">Detector rationale</div>'
            f'<div class="value">{_esc(finding.rationale)}</div></div>'
        )

    return (
        f'<article class="finding" style="border-left-color:{color}">'
        '<header class="finding-head">'
        f"<h3>{_esc(finding.name)}</h3>"
        f"<div>{_severity_badge(sev)}"
        f'<span class="conf">confidence {finding.confidence:.0%}</span></div>'
        "</header>"
        f'<div class="meta"><code>{_esc(finding.attack_id)}</code>'
        f'<span class="muted"> · {_esc(finding.technique.replace("_", " "))}</span></div>'
        f"<p>{_esc(finding.description)}</p>"
        f'<div class="field"><div class="label">Payload sent</div>'
        f"<pre>{_esc(finding.payload)}</pre></div>"
        f'<div class="field"><div class="label">Response excerpt</div>'
        f"<pre>{_esc(finding.response_excerpt)}</pre></div>"
        f"{rationale}{tags}{refs}"
        "</article>"
    )


_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif; line-height: 1.5; color: #1f2937;
  background: #f3f4f6;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }
header.top { display: flex; justify-content: space-between; align-items: flex-start;
  gap: 24px; flex-wrap: wrap; }
h1 { margin: 0 0 4px; font-size: 28px; }
.sub { color: #6b7280; font-size: 14px; margin: 0; }
.notice { margin: 16px 0 28px; padding: 10px 14px; background: #fffbeb;
  border: 1px solid #fde68a; border-radius: 8px; font-size: 13px; color: #92400e; }
.grade { text-align: center; min-width: 120px; }
.grade .letter { font-size: 64px; font-weight: 800; line-height: 1;
  width: 110px; height: 110px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; color: #fff; margin: 0 auto 6px; }
.grade .caption { font-size: 12px; color: #6b7280; }
.summary { display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }
.card { flex: 1 1 160px; background: #fff; border: 1px solid #e5e7eb;
  border-radius: 10px; padding: 16px; text-align: center; }
.card .n { font-size: 30px; font-weight: 700; }
.card .l { font-size: 12px; color: #6b7280; text-transform: uppercase;
  letter-spacing: .04em; }
h2 { font-size: 18px; margin: 32px 0 12px; }
table { width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #f1f5f9;
  font-size: 14px; }
th { background: #f9fafb; font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; color: #6b7280; }
tr:last-child td { border-bottom: 0; }
td.fail { color: #b91c1c; font-weight: 600; }
td.pass { color: #047857; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
  color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .03em; }
.muted { color: #9ca3af; }
.finding { background: #fff; border: 1px solid #e5e7eb; border-left-width: 5px;
  border-radius: 10px; padding: 18px 20px; margin: 16px 0; }
.finding-head { display: flex; justify-content: space-between; align-items: center;
  gap: 12px; flex-wrap: wrap; }
.finding-head h3 { margin: 0; font-size: 17px; }
.conf { margin-left: 10px; font-size: 12px; color: #6b7280; }
.meta { font-size: 13px; margin: 2px 0 8px; }
.meta code { background: #f3f4f6; padding: 1px 6px; border-radius: 5px; }
.field { margin: 12px 0; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  color: #6b7280; margin-bottom: 4px; }
pre { margin: 0; background: #0f172a; color: #e2e8f0; padding: 12px 14px;
  border-radius: 8px; overflow-x: auto; font-size: 13px;
  white-space: pre-wrap; word-break: break-word; }
.tags { margin-top: 10px; }
.chip { display: inline-block; background: #eef2ff; color: #3730a3;
  border-radius: 999px; padding: 2px 9px; font-size: 11px; margin: 2px 4px 0 0; }
.refs { margin-top: 12px; font-size: 13px; }
.refs ul { margin: 4px 0 0; padding-left: 18px; }
.clean { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46;
  padding: 16px; border-radius: 10px; }
.unreachable { margin: 0 0 8px; padding: 12px 16px; background: #fffbeb;
  border: 1px solid #fcd34d; border-radius: 10px; color: #92400e; font-size: 14px; }
footer { margin-top: 40px; font-size: 12px; color: #9ca3af; text-align: center; }
""".strip()


def _render_html(report: ScanReport) -> str:
    """Render the full standalone HTML document for ``report``."""
    grade = grade_report(report)
    grade_color = _GRADE_COLOR.get(grade, "#6b7280")
    grade_caption = "target unreachable" if report.all_errored else "security grade"

    target_line = _esc(report.target_name)
    if report.target_model:
        target_line += f' <span class="muted">({_esc(report.target_model)})</span>'

    # An unreachable-target notice (every attack errored) and/or a partial-error
    # line so a reader never mistakes errors for defended attacks.
    error_notice = ""
    if report.all_errored:
        error_notice = (
            '<div class="unreachable">⚠ Target unreachable — no usable '
            f"responses. All {report.total} attack(s) errored, so this scan "
            "cannot be graded and is <strong>not</strong> a pass.</div>"
        )
    elif report.errored:
        error_notice = (
            f'<div class="unreachable">⚠ {report.errored} attack(s) could not '
            "reach the target (errors) and are not counted as defended; the "
            "grade reflects only the attacks that got a response.</div>"
        )

    if report.findings:
        findings_html = "\n".join(_finding_card(f) for f in report.findings)
    else:
        findings_html = (
            '<div class="clean">No successful injections — the target defended '
            "every attack in this scan.</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>injectkit report — {_esc(report.target_name)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>injectkit report</h1>
      <p class="sub">Target: {target_line}</p>
      <p class="sub">Scanned {_fmt_time(report.started_at)} ·
        {report.duration_s:.2f}s · injectkit {_esc(report.tool_version)}</p>
    </div>
    <div class="grade">
      <div class="letter" style="background:{grade_color}">{grade}</div>
      <div class="caption">{grade_caption}</div>
    </div>
  </header>

  <div class="notice">{_esc(AUTHORIZED_USE_NOTICE)}</div>
{error_notice}
  <div class="summary">
    <div class="card"><div class="n">{report.total}</div>
      <div class="l">attacks run</div></div>
    <div class="card"><div class="n" style="color:#047857">{report.passed}</div>
      <div class="l">defended</div></div>
    <div class="card"><div class="n" style="color:#b91c1c">{report.failed}</div>
      <div class="l">vulnerable</div></div>{_errored_card(report)}
  </div>

  <h2>Technique breakdown</h2>
  <table>
    <thead><tr><th>Technique</th><th>Attacks</th><th>Result</th>
      <th>Worst severity</th></tr></thead>
    <tbody>
{_technique_rows(report)}
    </tbody>
  </table>

  <h2>Findings ({len(report.findings)})</h2>
  {findings_html}

  <footer>Generated by injectkit {_esc(report.tool_version)} — defensive,
    authorized-use-only LLM prompt-injection testing.</footer>
</div>
</body>
</html>
"""


class HtmlReporter:
    """Reporter that emits a single self-contained HTML document.

    Implements the :class:`~injectkit.reporters.base.Reporter` protocol.
    """

    name = "html"
    extension = ".html"

    def render(self, report: ScanReport) -> str:
        """Render ``report`` as a standalone HTML string."""
        return _render_html(report)
