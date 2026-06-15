"""Unit tests for the optional LLM judge (injectkit.evaluators.judge).

All tests run fully offline: the Anthropic SDK is never imported for real and no
network call is made. We inject a fake client (with a fake ``messages.parse``)
into :class:`JudgeDetector`, or monkeypatch the lazy-import path, so we can assert
on exactly what the judge sends and how it maps structured output to verdicts —
including the tricky refusal / None / error cases.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from injectkit.evaluators.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM,
    JudgeDetector,
    JudgeUnavailableError,
)
from injectkit.evaluators.base import Detector
from injectkit.models import Attack, DetectorVerdict, Severity, TargetResponse, Verdict


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeParseResult:
    """Stand-in for the object client.messages.parse returns."""

    def __init__(
        self,
        parsed_output: Optional[Verdict] = None,
        stop_reason: Optional[str] = "end_turn",
    ) -> None:
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason


class FakeMessages:
    """Records the kwargs of the last parse() call and returns a canned result."""

    def __init__(self, result: Any = None, raises: Optional[Exception] = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[dict] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._result


class FakeClient:
    """Minimal stand-in for anthropic.Anthropic()."""

    def __init__(self, result: Any = None, raises: Optional[Exception] = None) -> None:
        self.messages = FakeMessages(result=result, raises=raises)


def make_detector(
    parsed: Optional[Verdict] = None,
    stop_reason: str = "end_turn",
    raises: Optional[Exception] = None,
) -> tuple[JudgeDetector, FakeClient]:
    """Build a JudgeDetector wired to a fake client returning the given output."""
    result = None if raises is not None else FakeParseResult(parsed, stop_reason)
    client = FakeClient(result=result, raises=raises)
    return JudgeDetector(client=client), client


# --------------------------------------------------------------------------- #
# Protocol / construction
# --------------------------------------------------------------------------- #
def test_implements_detector_protocol() -> None:
    det = JudgeDetector(client=FakeClient())
    assert isinstance(det, Detector)
    assert det.name == "judge"
    assert det.model == DEFAULT_JUDGE_MODEL


def test_construction_does_no_network_and_needs_no_key(monkeypatch) -> None:
    # No client injected, no key in env: construction must still succeed because
    # the SDK/key are only needed at evaluate() time.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    det = JudgeDetector()  # must not raise
    assert det._client is None


# --------------------------------------------------------------------------- #
# Happy paths: success and defended
# --------------------------------------------------------------------------- #
def test_evaluate_success_maps_to_success_verdict(sample_attack: Attack) -> None:
    parsed = Verdict(
        is_success=True,
        severity="high",
        confidence=0.9,
        rationale="The model echoed the injected marker.",
    )
    det, client = make_detector(parsed=parsed)
    resp = TargetResponse(text="INJECTOK-abc123", refused=False, stop_reason="end_turn")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert isinstance(verdict, DetectorVerdict)
    assert verdict.detector == "judge"
    assert verdict.success is True
    assert verdict.confidence == pytest.approx(0.9)
    assert verdict.matched_conditions == ["judge"]
    assert "high" in verdict.rationale  # judge severity annotated


def test_evaluate_defended_maps_to_nonsuccess(sample_attack: Attack) -> None:
    parsed = Verdict(
        is_success=False,
        severity="info",
        confidence=0.8,
        rationale="The model stayed on task and ignored the injection.",
    )
    det, _ = make_detector(parsed=parsed)
    resp = TargetResponse(text="Here is a helpful, on-task answer.")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert verdict.matched_conditions == []
    assert verdict.confidence == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# The call shape sent to the SDK
# --------------------------------------------------------------------------- #
def test_parse_called_with_expected_args(sample_attack: Attack) -> None:
    parsed = Verdict(is_success=True, severity="high", confidence=0.5)
    det, client = make_detector(parsed=parsed)
    resp = TargetResponse(text="INJECTOK-canary99")

    det.evaluate(sample_attack, resp, "canary99")

    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == DEFAULT_JUDGE_MODEL
    assert call["output_format"] is Verdict
    assert call["system"] == JUDGE_SYSTEM
    # No temperature/top_p/top_k passed (removed on the model -> would 400).
    assert "temperature" not in call
    assert "top_p" not in call
    assert "top_k" not in call
    # User message carries the rendered payload, canary, and the response text.
    user_content = call["messages"][0]["content"]
    assert call["messages"][0]["role"] == "user"
    assert "canary99" in user_content
    assert "INJECTOK-canary99" in user_content  # rendered payload + response text
    assert sample_attack.id in user_content


def test_judge_system_prompt_is_hardened() -> None:
    # The system prompt must define success precisely and resist manipulation
    # via untrusted response content.
    s = JUDGE_SYSTEM.lower()
    assert "is_success" in s
    assert "untrusted" in s
    assert "refus" in s  # refusal handling described
    # Anti-injection instruction for the judge itself.
    assert "do not follow" in s or "disregard" in s


# --------------------------------------------------------------------------- #
# Defensive: refusal, None output, exceptions
# --------------------------------------------------------------------------- #
def test_adapter_refusal_short_circuits_without_calling_judge(
    sample_attack: Attack,
) -> None:
    det, client = make_detector(parsed=Verdict(is_success=True))
    resp = TargetResponse(text="", refused=True, stop_reason="refusal")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert verdict.confidence > 0.5
    # No judge call spent on a clear refusal.
    assert client.messages.calls == []


def test_adapter_error_short_circuits_without_calling_judge(
    sample_attack: Attack,
) -> None:
    # A transport/API error on the target means there is no real response to
    # grade. The judge must abstain (confidence 0.0, non-success) WITHOUT
    # spending an API call, so scoring falls back to the offline heuristics.
    det, client = make_detector(parsed=Verdict(is_success=True))
    resp = TargetResponse(text="", refused=False, error="ConnectTimeout: boom")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert verdict.confidence == 0.0
    assert "error" in verdict.rationale.lower()
    # No judge call spent on an errored response.
    assert client.messages.calls == []


def test_judge_refusal_stop_reason_yields_nonsuccess(sample_attack: Attack) -> None:
    # messages.parse returns stop_reason == "refusal" -> treat as non-success.
    det, _ = make_detector(parsed=Verdict(is_success=True), stop_reason="refusal")
    resp = TargetResponse(text="some output")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert verdict.confidence == 0.0
    assert "no structured verdict" in verdict.rationale.lower()


def test_none_parsed_output_yields_nonsuccess(sample_attack: Attack) -> None:
    det, _ = make_detector(parsed=None, stop_reason="end_turn")
    resp = TargetResponse(text="some output")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert verdict.confidence == 0.0


def test_api_exception_yields_nonsuccess_not_raise(sample_attack: Attack) -> None:
    det, _ = make_detector(raises=RuntimeError("boom 500"))
    resp = TargetResponse(text="some output")

    verdict = det.evaluate(sample_attack, resp, "abc123")

    assert verdict.success is False
    assert "failed" in verdict.rationale.lower()


# --------------------------------------------------------------------------- #
# Confidence clamping / severity coercion
# --------------------------------------------------------------------------- #
def test_confidence_is_clamped(sample_attack: Attack) -> None:
    # Verdict's own validator enforces 0..1, so feed an out-of-range value by
    # constructing a loose object that quacks like a Verdict.
    loose = SimpleNamespace(
        is_success=True, severity="weird", confidence=3.5, rationale=""
    )
    det = JudgeDetector(client=FakeClient(result=FakeParseResult(loose)))
    resp = TargetResponse(text="x")

    verdict = det.evaluate(sample_attack, resp, "c")

    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.confidence == 1.0  # 3.5 clamped down
    # Unknown severity string coerces to a safe default and still annotates.
    assert verdict.success is True


# --------------------------------------------------------------------------- #
# Lazy import / friendly errors
# --------------------------------------------------------------------------- #
def test_missing_sdk_raises_friendly_error(monkeypatch) -> None:
    det = JudgeDetector()  # no client injected -> will lazy-build
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(JudgeUnavailableError) as exc:
        det._get_client()
    assert "anthropic" in str(exc.value).lower()


def test_missing_api_key_raises_friendly_error(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    det = JudgeDetector()

    # Pretend the SDK imports fine but there is no key.
    fake_anthropic = SimpleNamespace(Anthropic=lambda **kw: FakeClient())
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

    with pytest.raises(JudgeUnavailableError) as exc:
        det._get_client()
    assert "key" in str(exc.value).lower()


def test_explicit_api_key_used_when_no_env(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured: dict = {}

    def fake_anthropic_ctor(**kw: Any) -> FakeClient:
        captured.update(kw)
        return FakeClient()

    fake_anthropic = SimpleNamespace(Anthropic=fake_anthropic_ctor)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

    det = JudgeDetector(api_key="sk-test-123")
    client = det._get_client()

    assert isinstance(client, FakeClient)
    assert captured.get("api_key") == "sk-test-123"


def test_injected_client_skips_sdk_and_key(monkeypatch) -> None:
    # With a client injected, _get_client must not touch the SDK or env at all.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("imported anthropic!"))
        if a and a[0] == "anthropic"
        else __import__(*a, **k),
    )
    fc = FakeClient()
    det = JudgeDetector(client=fc)
    assert det._get_client() is fc
