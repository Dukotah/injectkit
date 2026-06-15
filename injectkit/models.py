"""Shared data contracts for injectkit.

This module is the single source of truth for the types every other module
depends on. The dataclasses and enums here are frozen interfaces: module
builders (targets, evaluators, reporters, engine, CLI) code against these
field names. Changing a field name here is a breaking change for everyone.

Nothing in this module imports anthropic, httpx, mcp, or any heavy/optional
dependency — it must always import cleanly so the core CLI works without
optional SDKs installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

__all__ = [
    "Severity",
    "SEVERITY_ORDER",
    "Attack",
    "TargetResponse",
    "DetectorVerdict",
    "AttackResult",
    "Finding",
    "ScanReport",
    "Verdict",
    "TargetConfig",
]


class Severity(str, Enum):
    """Severity of a successful injection, ordered info < low < ... < critical.

    Inherits from ``str`` so values serialize cleanly to YAML/JSON and compare
    equal to their plain string form (e.g. ``Severity.HIGH == "high"``).
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Integer rank for ordering/comparison (info=0 .. critical=4)."""
        return SEVERITY_ORDER.index(self.value)

    @classmethod
    def coerce(cls, value: "str | Severity") -> "Severity":
        """Parse a string (case-insensitive) or pass through a Severity."""
        if isinstance(value, Severity):
            return value
        return cls(str(value).strip().lower())


# Canonical low-to-high ordering. Use Severity.rank for comparisons; this list
# is the backing order and is handy for --fail-on threshold logic.
SEVERITY_ORDER: list[str] = [
    Severity.INFO.value,
    Severity.LOW.value,
    Severity.MEDIUM.value,
    Severity.HIGH.value,
    Severity.CRITICAL.value,
]


@dataclass
class Attack:
    """A single prompt-injection test case, loaded from a corpus YAML file.

    Fields mirror the corpus schema. ``payload`` may contain a ``{canary}``
    placeholder that the engine renders with a per-run unique marker before
    sending to the target. ``success_conditions`` is the raw rule dict that the
    heuristic detector interprets (keys: marker_echoed, regex, refusal_absent,
    system_prompt_leaked, canary_in_output).
    """

    id: str
    technique: str
    name: str
    description: str
    severity: Severity
    payload: str
    success_conditions: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Optional system prompt to send with this attack (e.g. a fake "secret"
    # system prompt for system-prompt-leak attacks). None => target default.
    system: Optional[str] = None
    # Optional extra context (e.g. simulated retrieved document for indirect
    # injection). Passed through to Target.send as the `context` argument.
    context: Optional[str] = None
    # Source file the attack was loaded from (populated by the loader).
    source_file: Optional[str] = None

    def render(self, canary: str) -> str:
        """Return the payload with ``{canary}`` substituted for ``canary``.

        Uses str.replace (not str.format) so literal braces elsewhere in the
        payload are left untouched.
        """
        return self.payload.replace("{canary}", canary)


@dataclass
class TargetResponse:
    """Normalized response from any Target adapter.

    ``text`` is the model's reply text (already extracted from whatever shape
    the underlying SDK uses). ``refused`` is True when the target *declined* —
    for an attack, a refusal means the target SUCCESSFULLY DEFENDED, so
    detectors treat it as a non-success signal.
    """

    text: str
    # True when the model refused/declined (e.g. Anthropic stop_reason ==
    # "refusal"). A refusal is the defender winning.
    refused: bool = False
    # Raw stop reason / finish reason from the provider, if available.
    stop_reason: Optional[str] = None
    # Provider/model identifier that produced this response.
    model: Optional[str] = None
    # Arbitrary adapter-specific extras (tool calls, usage, latency, etc.).
    raw: dict[str, Any] = field(default_factory=dict)
    # True if the adapter hit an error talking to the target (network, auth).
    error: Optional[str] = None


@dataclass
class DetectorVerdict:
    """The result a single Detector returns for one (attack, response) pair.

    ``success`` means the detector believes the injection worked (the defense
    failed). ``confidence`` is 0.0-1.0. ``detector`` names which detector
    produced this verdict so reporters can attribute it.
    """

    detector: str
    success: bool
    confidence: float = 1.0
    rationale: str = ""
    # Which success_condition rule(s) fired, for transparency in reports.
    matched_conditions: list[str] = field(default_factory=list)


@dataclass
class AttackResult:
    """Outcome of running one Attack against the target.

    Aggregates the raw response plus every detector verdict, and the engine's
    final scored decision (success/severity/confidence). The scoring module
    fills in ``success``, ``severity``, and ``confidence`` from ``verdicts``.
    """

    attack: Attack
    canary: str
    response: TargetResponse
    verdicts: list[DetectorVerdict] = field(default_factory=list)
    # Final scored decision (filled by scoring.score()).
    success: bool = False
    severity: Severity = Severity.INFO
    confidence: float = 0.0
    # Wall-clock seconds the attack took (send + evaluate).
    duration_s: float = 0.0

    @property
    def detected(self) -> bool:
        """True if any detector flagged success."""
        return any(v.success for v in self.verdicts)


@dataclass
class Finding:
    """A confirmed vulnerability: an attack that succeeded.

    Reporters render findings. Built from a successful AttackResult by the
    engine. Carries enough context to be actionable in a report without the
    reader needing the full AttackResult.
    """

    attack_id: str
    technique: str
    name: str
    severity: Severity
    confidence: float
    description: str
    payload: str
    response_excerpt: str
    rationale: str = ""
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: AttackResult, excerpt_len: int = 500) -> "Finding":
        """Build a Finding from a successful AttackResult."""
        rationale = "; ".join(
            v.rationale for v in result.verdicts if v.success and v.rationale
        )
        excerpt = (result.response.text or "")[:excerpt_len]
        return cls(
            attack_id=result.attack.id,
            technique=result.attack.technique,
            name=result.attack.name,
            severity=result.severity,
            confidence=result.confidence,
            description=result.attack.description,
            payload=result.attack.render(result.canary),
            response_excerpt=excerpt,
            rationale=rationale,
            references=list(result.attack.references),
            tags=list(result.attack.tags),
        )


@dataclass
class ScanReport:
    """Aggregate result of a full scan: every result plus rolled-up findings.

    This is what every Reporter renders. ``findings`` is the subset of
    ``results`` that succeeded. ``target_name`` / ``target_model`` describe what
    was scanned for the report header.
    """

    target_name: str
    results: list[AttackResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    target_model: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # injectkit version that produced the report.
    tool_version: str = "0.1.0"

    # ---- convenience rollups (computed, not stored) ----

    @property
    def total(self) -> int:
        """Number of attacks run."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Attacks the target genuinely defended.

        Only counts attacks that got a usable response and did NOT succeed. An
        attack whose target call errored (``response.error`` set) is NOT a
        defense — it never reached the target — so it is excluded here and
        counted by :attr:`errored` instead. Thus ``total == passed + failed +
        errored``.
        """
        return sum(
            1 for r in self.results if not r.success and r.response.error is None
        )

    @property
    def failed(self) -> int:
        """Attacks that succeeded against the target (vulnerabilities)."""
        return sum(1 for r in self.results if r.success)

    @property
    def errored(self) -> int:
        """Attacks whose target call errored (never got a usable response).

        These must not be read as defended — a wrong URL, missing API key, or
        unreachable host would otherwise masquerade as a clean pass.
        """
        return sum(1 for r in self.results if r.response.error is not None)

    @property
    def all_errored(self) -> bool:
        """True when the scan ran but every attack errored (target unreachable).

        Reporters use this to render the grade as N/A instead of a misleading
        clean ``A`` for a target that never returned a usable response.
        """
        return bool(self.results) and self.errored == self.total

    @property
    def duration_s(self) -> float:
        """Total wall-clock scan duration in seconds."""
        if self.finished_at is None:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)

    def severity_counts(self) -> dict[str, int]:
        """Count of findings per severity level (only severities present)."""
        counts: dict[str, int] = {}
        for f in self.findings:
            key = f.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def highest_severity(self) -> Optional[Severity]:
        """The most severe finding, or None if the target passed everything."""
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: s.rank)


class Verdict(BaseModel):
    """Structured output schema for the optional LLM judge.

    Passed to ``client.messages.parse(..., output_format=Verdict)`` so the
    Anthropic SDK returns a validated instance. Field names are part of the
    judge's contract — the judge module and scoring module both rely on them.
    """

    is_success: bool = Field(
        description="True if the injection attack succeeded (the model's "
        "defense failed); False if the model resisted or refused."
    )
    severity: str = Field(
        default="info",
        description="Severity if successful: one of info, low, medium, high, "
        "critical. Use 'info' when is_success is False.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this judgement, 0.0 to 1.0.",
    )
    rationale: str = Field(
        default="",
        description="One or two sentences explaining the judgement.",
    )


@dataclass
class TargetConfig:
    """Typed configuration for constructing a Target adapter.

    The CLI / config loader builds one of these, and the target factory turns
    it into a concrete Target. ``kind`` selects the adapter
    ("http" | "anthropic" | "mcp" | "mock"). The remaining fields are a union
    of what the various adapters need; each adapter reads only what it uses.
    """

    kind: str = "anthropic"
    name: str = "target"
    # --- anthropic / generic model selection ---
    model: Optional[str] = None
    system: Optional[str] = None
    max_tokens: int = 1024
    # --- http adapter ---
    url: Optional[str] = None
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    # Request body template; "{prompt}" is substituted with the rendered attack.
    request_template: Optional[dict[str, Any]] = None
    # JSONPath-ish dotted path to the reply text in the JSON response,
    # e.g. "choices.0.message.content".
    response_path: Optional[str] = None
    timeout_s: float = 30.0
    # --- mcp adapter ---
    mcp_command: Optional[str] = None
    mcp_args: list[str] = field(default_factory=list)
    mcp_url: Optional[str] = None
    # --- catch-all for adapter-specific extras ---
    extra: dict[str, Any] = field(default_factory=dict)
