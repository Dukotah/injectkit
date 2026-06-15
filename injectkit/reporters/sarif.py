"""SARIF 2.1.0 reporter.

Renders a :class:`~injectkit.models.ScanReport` as a SARIF 2.1.0 log so that
injectkit findings surface in GitHub code-scanning ("Security" tab). The GitHub
Action uploads this file; code-scanning ingests it and can gate PRs.

SARIF shape produced here:

  * ``$schema`` + ``version`` = "2.1.0"
  * a single ``run`` with one ``tool.driver`` ("injectkit")
  * ``tool.driver.rules`` — one reportingDescriptor per *attack technique that
    produced a finding* (deduped by ruleId), each with a help text and the
    injectkit severity mapped to a SARIF ``security-severity`` property
  * ``results`` — one result per finding, ``level`` mapped from severity, with a
    message, the rule reference, and a (synthetic) location so GitHub renders it

The authorized-use notice is embedded in the run's ``properties`` and in the
tool driver's full description so it travels with every uploaded report.

This module is dependency-free (stdlib ``json`` only) and never touches the
network, so it imports cleanly anywhere.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import Finding, ScanReport, Severity
from .base import AUTHORIZED_USE_NOTICE

__all__ = ["SarifReporter", "build_sarif"]

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/"
    "sarif-schema-2.1.0.json"
)

#: injectkit severity -> SARIF result level. SARIF levels are a small, fixed
#: set ("none" | "note" | "warning" | "error"); we collapse our five severities
#: onto them while keeping the precise value in ``security-severity``.
_SARIF_LEVEL: dict[str, str] = {
    Severity.INFO.value: "note",
    Severity.LOW.value: "note",
    Severity.MEDIUM.value: "warning",
    Severity.HIGH.value: "error",
    Severity.CRITICAL.value: "error",
}

#: injectkit severity -> GitHub ``security-severity`` numeric string (CVSS-like,
#: 0.0-10.0). GitHub uses this to bucket alerts as low/medium/high/critical.
_SECURITY_SEVERITY: dict[str, str] = {
    Severity.INFO.value: "0.0",
    Severity.LOW.value: "3.0",
    Severity.MEDIUM.value: "5.5",
    Severity.HIGH.value: "8.0",
    Severity.CRITICAL.value: "9.5",
}


def _level_for(severity: Severity) -> str:
    """Map an injectkit Severity onto a SARIF result level."""
    return _SARIF_LEVEL.get(Severity.coerce(severity).value, "warning")


def _security_severity_for(severity: Severity) -> str:
    """Map an injectkit Severity onto a GitHub security-severity number."""
    return _SECURITY_SEVERITY.get(Severity.coerce(severity).value, "5.5")


#: Schemes permitted in the rule ``helpUri`` (SARIF consumers like GitHub render
#: it as a clickable link). Reference URLs come from community-contributed
#: corpus YAML, so a hostile ``javascript:``/``data:`` URL is dropped rather
#: than handed to a viewer as a live link.
_SAFE_URI_SCHEMES = ("http://", "https://", "mailto:")


def _safe_uri(url: str) -> str:
    """Return ``url`` if it uses a safe absolute scheme, else ``""``.

    SARIF ``helpUri`` is expected to be an absolute URL; we only emit one when
    it is an ``http(s)``/``mailto`` URL. Anything else (dangerous scheme, or a
    relative/garbage value) yields an empty string, which is omitted downstream.
    """
    if not url:
        return ""
    cleaned = "".join(ch for ch in str(url) if ord(ch) > 0x20).strip()
    return cleaned if cleaned.lower().startswith(_SAFE_URI_SCHEMES) else ""


def _rule_id(finding: Finding) -> str:
    """Stable SARIF ruleId for a finding (one rule per technique)."""
    return f"injectkit/{finding.technique}"


def _build_rules(findings: list[Finding]) -> list[dict[str, Any]]:
    """Build the deduped list of reportingDescriptor rules for the run.

    One rule per technique that produced at least one finding. The rule's
    ``security-severity`` reflects the highest severity seen for that technique.
    """
    by_id: dict[str, Finding] = {}
    worst: dict[str, Severity] = {}
    for f in findings:
        rid = _rule_id(f)
        sev = Severity.coerce(f.severity)
        if rid not in by_id:
            by_id[rid] = f
            worst[rid] = sev
        elif sev.rank > worst[rid].rank:
            worst[rid] = sev

    rules: list[dict[str, Any]] = []
    for rid, f in by_id.items():
        # Only the first safe http(s)/mailto reference becomes the helpUri.
        help_uri = next((u for u in (_safe_uri(r) for r in f.references) if u), "")
        rule: dict[str, Any] = {
            "id": rid,
            "name": f.technique.replace("_", " ").title().replace(" ", ""),
            "shortDescription": {
                "text": f"Prompt injection: {f.technique.replace('_', ' ')}"
            },
            "fullDescription": {
                "text": (
                    f"injectkit detected a successful {f.technique.replace('_', ' ')} "
                    "prompt-injection attack against the target. The model's "
                    "defenses did not prevent the injected instruction from taking "
                    "effect."
                )
            },
            "help": {
                "text": (
                    "Harden the application's prompt handling: isolate untrusted "
                    "input, add output/marker filtering, and constrain tool access. "
                    f"\n\n{AUTHORIZED_USE_NOTICE}"
                )
            },
            "defaultConfiguration": {"level": _level_for(worst[rid])},
            "properties": {
                "security-severity": _security_severity_for(worst[rid]),
                "tags": ["security", "prompt-injection", "llm", f.technique],
            },
        }
        # Omit helpUri entirely unless we have a safe absolute URL.
        if help_uri:
            rule["helpUri"] = help_uri
        rules.append(rule)
    return rules


def _build_result(finding: Finding, rule_index: int) -> dict[str, Any]:
    """Build one SARIF result object for a single finding."""
    message = (
        f"{finding.name} [{Severity.coerce(finding.severity).value}] "
        f"(confidence {finding.confidence:.2f}) — {finding.description}"
    )
    if finding.rationale:
        message += f" Rationale: {finding.rationale}"

    # GitHub requires a physicalLocation to render a result. injectkit findings
    # are not tied to a source line, so we point at a synthetic artifact named
    # after the attack id; this keeps each finding distinct in the UI.
    artifact_uri = f"injectkit-findings/{finding.attack_id}.txt"

    return {
        "ruleId": _rule_id(finding),
        "ruleIndex": rule_index,
        "level": _level_for(finding.severity),
        "message": {"text": message},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": artifact_uri,
                        "uriBaseId": "SRCROOT",
                    },
                    "region": {"startLine": 1, "startColumn": 1},
                }
            }
        ],
        "partialFingerprints": {
            # Stable across runs so GitHub dedupes alerts: technique + attack id.
            "injectkitFindingId": f"{finding.technique}:{finding.attack_id}",
        },
        "properties": {
            "attack_id": finding.attack_id,
            "technique": finding.technique,
            "severity": Severity.coerce(finding.severity).value,
            "confidence": finding.confidence,
            "payload": finding.payload,
            "response_excerpt": finding.response_excerpt,
            "tags": list(finding.tags),
        },
    }


def build_sarif(report: ScanReport) -> dict[str, Any]:
    """Build the SARIF 2.1.0 log document (as a plain dict) for ``report``."""
    rules = _build_rules(report.findings)
    rule_index_by_id = {rule["id"]: i for i, rule in enumerate(rules)}

    results = [
        _build_result(f, rule_index_by_id[_rule_id(f)]) for f in report.findings
    ]

    driver: dict[str, Any] = {
        "name": "injectkit",
        "informationUri": "https://github.com/Dukotah/injectkit",
        "version": report.tool_version,
        "fullName": f"injectkit {report.tool_version}",
        "shortDescription": {
            "text": "Prompt-injection red-team scanner for LLM applications."
        },
        "fullDescription": {
            "text": (
                "injectkit red-teams LLM applications for prompt injection. "
                f"{AUTHORIZED_USE_NOTICE}"
            )
        },
        "rules": rules,
    }

    run: dict[str, Any] = {
        "tool": {"driver": driver},
        "results": results,
        "columnKind": "utf16CodeUnits",
        "properties": {
            "authorized_use_notice": AUTHORIZED_USE_NOTICE,
            "target_name": report.target_name,
            "target_model": report.target_model,
            "total_attacks": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "severity_counts": report.severity_counts(),
        },
    }

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [run],
    }


class SarifReporter:
    """Reporter that emits a SARIF 2.1.0 log for GitHub code-scanning.

    Implements the :class:`~injectkit.reporters.base.Reporter` protocol.
    """

    name = "sarif"
    extension = ".sarif"

    def render(self, report: ScanReport) -> str:
        """Render ``report`` as a pretty-printed SARIF 2.1.0 JSON string."""
        return json.dumps(build_sarif(report), indent=2, ensure_ascii=False)
