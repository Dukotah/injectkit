"""injectkit — an open-source red-team scanner for LLM prompt injection.

DEFENSIVE / AUTHORIZED USE ONLY. injectkit scans LLM endpoints that you own
or are explicitly authorized to test. It is the "scan your own site" of prompt
injection: point it at your own chatbot, agent, or model endpoint to find
injection weaknesses before an attacker does. Do not target third parties.

Public API (stable surface other code should import from the package root):

    from injectkit import (
        Attack, AttackResult, Finding, Severity, ScanReport, Verdict,
        TargetResponse, DetectorVerdict, TargetConfig,
    )
"""

from __future__ import annotations

from .models import (
    Attack,
    AttackResult,
    DetectorVerdict,
    Finding,
    ScanReport,
    Severity,
    TargetConfig,
    TargetResponse,
    Verdict,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Attack",
    "AttackResult",
    "DetectorVerdict",
    "Finding",
    "ScanReport",
    "Severity",
    "TargetConfig",
    "TargetResponse",
    "Verdict",
]
