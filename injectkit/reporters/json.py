"""JSON reporter — the machine-readable full scan report.

Serializes a :class:`~injectkit.models.ScanReport` (including every
:class:`AttackResult` and every :class:`Finding`) into a stable, well-typed
JSON document. This is the format CI pipelines and downstream tooling consume;
it is intentionally lossless where practical so nothing about a finding is
hidden behind a pretty layout.

The top-level shape::

    {
      "tool": "injectkit",
      "tool_version": "0.1.0",
      "authorized_use_notice": "...",
      "target": {"name": "...", "model": "..."},
      "summary": {"total": 1, "passed": 0, "failed": 1, ...},
      "findings": [ {finding...}, ... ],
      "results": [ {result...}, ... ]
    }
"""

from __future__ import annotations

import json
from typing import Any

from ..models import (
    AttackResult,
    DetectorVerdict,
    Finding,
    ScanReport,
    Severity,
    TargetResponse,
)
from .base import AUTHORIZED_USE_NOTICE

__all__ = ["JSONReporter"]


def _sev(value: Severity) -> str:
    """Normalize a Severity (or str-enum) to its plain string value."""
    return value.value if isinstance(value, Severity) else str(value)


def _verdict_dict(v: DetectorVerdict) -> dict[str, Any]:
    """Serialize one detector verdict."""
    return {
        "detector": v.detector,
        "success": v.success,
        "confidence": v.confidence,
        "rationale": v.rationale,
        "matched_conditions": list(v.matched_conditions),
    }


def _response_dict(r: TargetResponse) -> dict[str, Any]:
    """Serialize a target response. ``raw`` is included best-effort."""
    return {
        "text": r.text,
        "refused": r.refused,
        "stop_reason": r.stop_reason,
        "model": r.model,
        "error": r.error,
        "raw": r.raw,
    }


def _finding_dict(f: Finding) -> dict[str, Any]:
    """Serialize one finding."""
    return {
        "attack_id": f.attack_id,
        "technique": f.technique,
        "name": f.name,
        "severity": _sev(f.severity),
        "confidence": f.confidence,
        "description": f.description,
        "payload": f.payload,
        "response_excerpt": f.response_excerpt,
        "rationale": f.rationale,
        "references": list(f.references),
        "tags": list(f.tags),
    }


def _result_dict(r: AttackResult) -> dict[str, Any]:
    """Serialize one attack result, including the attack metadata."""
    a = r.attack
    return {
        "attack": {
            "id": a.id,
            "technique": a.technique,
            "name": a.name,
            "description": a.description,
            "severity": _sev(a.severity),
            "tags": list(a.tags),
            "references": list(a.references),
            "source_file": a.source_file,
        },
        "canary": r.canary,
        "success": r.success,
        "detected": r.detected,
        "severity": _sev(r.severity),
        "confidence": r.confidence,
        "duration_s": r.duration_s,
        "response": _response_dict(r.response),
        "verdicts": [_verdict_dict(v) for v in r.verdicts],
    }


class JSONReporter:
    """Render a :class:`ScanReport` as a machine-readable JSON document.

    Implements the :class:`~injectkit.reporters.base.Reporter` protocol.
    """

    name = "json"
    extension = ".json"

    def __init__(self, *, indent: int = 2) -> None:
        """Args:
        indent: JSON indentation. Use ``0``/``None`` for compact output.
        """
        self.indent = indent or None

    def to_dict(self, report: ScanReport) -> dict[str, Any]:
        """Build the plain-dict document for ``report`` (no JSON encoding)."""
        highest = report.highest_severity
        return {
            "tool": "injectkit",
            "tool_version": report.tool_version,
            "authorized_use_notice": AUTHORIZED_USE_NOTICE,
            "target": {
                "name": report.target_name,
                "model": report.target_model,
            },
            "timing": {
                "started_at": report.started_at,
                "finished_at": report.finished_at,
                "duration_s": report.duration_s,
            },
            "summary": {
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "highest_severity": _sev(highest) if highest else None,
                "severity_counts": report.severity_counts(),
            },
            "findings": [_finding_dict(f) for f in report.findings],
            "results": [_result_dict(r) for r in report.results],
        }

    def render(self, report: ScanReport) -> str:
        """Render ``report`` to a JSON string."""
        return json.dumps(
            self.to_dict(report),
            indent=self.indent,
            ensure_ascii=False,
            sort_keys=False,
        )
