"""Tests for the reply-referencing & agent-decomposition crescendo strategies.

Covers the v0.3.0 additions to ``injectkit/attacks/multiturn.py``:

* :class:`CrescendoReplyReferencingStrategy` — the frozen ``next_turn`` /
  ``final_step`` reply-aware hooks (Crescendo, arXiv:2404.01833).
* :class:`CrescendoDecompositionStrategy` — the agent-decomposition variant.

All offline and deterministic. The end-to-end drive replays the documented
engine loop against the in-repo ``FakeConversationalTarget`` fixture (no network,
no model, no translation). We verify: lead-ins quote the model's OWN prior reply,
lead-ins never leak the live canary, exactly the final ask is scored & carries the
marker, and driving the loop against a vulnerable target scores a benign hit.
"""

from __future__ import annotations

import pytest

from injectkit.attacks.base import AttackStep, AttackStrategy, StrategyError
from injectkit.attacks.multiturn import (
    MULTI_TURN_STRATEGIES,
    CrescendoDecompositionStrategy,
    CrescendoReplyReferencingStrategy,
    CrescendoStrategy,
    build_strategy,
)
from injectkit.models import TargetResponse
from injectkit.targets.conversational import ChatMessage

CANARY = "cana12345"
MARKER = f"INJECTOK-{CANARY}"


def _reply_strategies() -> list[CrescendoReplyReferencingStrategy]:
    return [
        CrescendoReplyReferencingStrategy(steps=3),
        CrescendoDecompositionStrategy(steps=3),
    ]


def _drive(strat, attack, canary, target):
    """Replay the documented engine loop: reply-aware lead-ins, then scored ask.

    Returns ``(history, final_step, final_response)``.
    """
    history: list[tuple[str, str]] = []
    for _ in range(strat.steps):
        step = strat.next_turn(attack, canary, history)
        messages = [ChatMessage(role=r, content=c) for r, c in history]
        messages.append(step.message)
        reply = target.chat(messages)
        history.append((step.message.role, step.message.content))
        history.append(("assistant", reply.text))
    final = strat.final_step(attack, canary, history)
    messages = [ChatMessage(role=r, content=c) for r, c in history]
    messages.append(final.message)
    final_response = target.chat(messages)
    return history, final, final_response


# --------------------------------------------------------------------------- #
# Type / contract conformance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_is_attack_strategy_subclass_of_crescendo(strat):
    assert isinstance(strat, AttackStrategy)
    assert isinstance(strat, CrescendoStrategy)
    assert isinstance(strat.name, str) and strat.name


def test_names_are_frozen():
    assert CrescendoReplyReferencingStrategy.name == "crescendo_reply"
    assert CrescendoDecompositionStrategy.name == "crescendo_decompose"


def test_registry_includes_new_strategies():
    assert "crescendo_reply" in MULTI_TURN_STRATEGIES
    assert "crescendo_decompose" in MULTI_TURN_STRATEGIES
    assert MULTI_TURN_STRATEGIES["crescendo_reply"] is CrescendoReplyReferencingStrategy
    assert (
        MULTI_TURN_STRATEGIES["crescendo_decompose"]
        is CrescendoDecompositionStrategy
    )


# --------------------------------------------------------------------------- #
# next_turn: reply-referencing behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_next_turn_first_call_has_no_reply_to_quote(strat, sample_attack):
    step = strat.next_turn(sample_attack, CANARY, history=[])
    assert isinstance(step, AttackStep)
    assert step.message.role == "user"
    assert step.scored is False
    assert step.expect_response is True
    assert CANARY not in step.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_next_turn_quotes_latest_assistant_reply(strat, sample_attack):
    prior_reply = "Sure, I can echo a fixed test token verbatim when asked."
    history = [
        ("user", "first lead-in"),
        ("assistant", prior_reply),
    ]
    step = strat.next_turn(sample_attack, CANARY, history)
    # A bounded excerpt of the model's own reply must appear in the new lead-in.
    assert prior_reply in step.message.content
    assert step.scored is False
    assert step.expect_response is True
    assert CANARY not in step.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_next_turn_excerpt_is_bounded(strat, sample_attack):
    long_reply = "A" * 5000
    history = [("user", "x"), ("assistant", long_reply)]
    step = strat.next_turn(sample_attack, CANARY, history)
    # The full 5000-char reply must NOT be embedded verbatim; it is truncated.
    assert long_reply not in step.message.content
    assert len(step.message.content) < 1000


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_next_turn_uses_most_recent_assistant_reply(strat, sample_attack):
    history = [
        ("user", "u1"),
        ("assistant", "OLDER reply text"),
        ("user", "u2"),
        ("assistant", "NEWER reply text"),
    ]
    step = strat.next_turn(sample_attack, CANARY, history)
    assert "NEWER reply text" in step.message.content
    assert "OLDER reply text" not in step.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_next_turn_is_deterministic(strat, sample_attack):
    history = [("user", "u"), ("assistant", "a cooperative reply")]
    a = strat.next_turn(sample_attack, CANARY, history)
    b = strat.next_turn(sample_attack, CANARY, history)
    assert (a.message.role, a.message.content) == (b.message.role, b.message.content)


# --------------------------------------------------------------------------- #
# final_step: the single scored ask
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_final_step_is_scored_user_turn_with_marker(strat, sample_attack):
    history = [("user", "u"), ("assistant", "we are aligned and ready")]
    final = strat.final_step(sample_attack, CANARY, history)
    assert isinstance(final, AttackStep)
    assert final.scored is True
    assert final.expect_response is True
    assert final.message.role == "user"
    assert MARKER in final.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_final_step_without_history_still_carries_marker(strat, sample_attack):
    final = strat.final_step(sample_attack, CANARY, history=[])
    assert final.scored is True
    assert MARKER in final.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_final_step_references_prior_reply_when_present(strat, sample_attack):
    history = [("user", "u"), ("assistant", "DISTINCTIVE closing phrase")]
    final = strat.final_step(sample_attack, CANARY, history)
    assert "DISTINCTIVE closing phrase" in final.message.content


# --------------------------------------------------------------------------- #
# End-to-end drive of the documented reply-aware engine loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_drive_loop_lead_ins_never_leak_canary(
    strat, sample_attack, fake_conversational_target
):
    history, final, _resp = _drive(
        strat, sample_attack, CANARY, fake_conversational_target
    )
    # Every user lead-in delivered before the final ask is canary-free.
    user_leadins = [c for r, c in history if r == "user"]
    assert user_leadins, "expected at least one lead-in"
    for content in user_leadins:
        assert CANARY not in content
    # Exactly the final step carries the marker and is scored.
    assert final.scored is True
    assert MARKER in final.message.content


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_drive_loop_scores_marker_against_vulnerable_target(
    strat, sample_attack, fake_conversational_target
):
    _history, _final, resp = _drive(
        strat, sample_attack, CANARY, fake_conversational_target
    )
    assert isinstance(resp, TargetResponse)
    assert resp.refused is False
    assert resp.text == MARKER


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_drive_loop_refusing_target_does_not_score(
    strat, sample_attack, fake_conversational_target
):
    fake_conversational_target.vulnerable = False
    _history, _final, resp = _drive(
        strat, sample_attack, CANARY, fake_conversational_target
    )
    assert resp.refused is True
    assert MARKER not in resp.text


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_drive_loop_quotes_real_target_replies(
    strat, sample_attack, fake_conversational_target
):
    """After the first lead-in, later lead-ins must quote the target's reply."""
    history: list[tuple[str, str]] = []
    contents: list[str] = []
    for _ in range(strat.steps):
        step = strat.next_turn(sample_attack, CANARY, history)
        contents.append(step.message.content)
        messages = [ChatMessage(role=r, content=c) for r, c in history]
        messages.append(step.message)
        reply = fake_conversational_target.chat(messages)
        history.append((step.message.role, step.message.content))
        history.append(("assistant", reply.text))
    # The vulnerable target replies "On-task answer." to canary-free lead-ins;
    # the 2nd lead-in onward must quote that real reply.
    assert any("On-task answer." in c for c in contents[1:])


# --------------------------------------------------------------------------- #
# Static build() contract still holds (factory + shared contract tests rely on it)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _reply_strategies(), ids=lambda s: s.name)
def test_static_build_has_exactly_one_scored_with_marker(strat, sample_attack):
    steps = strat.build(sample_attack, CANARY)
    assert len(steps) >= 2
    scored = [s for s in steps if s.scored]
    assert len(scored) == 1
    assert steps[-1] is scored[0]
    assert steps[-1].message.role == "user"
    assert MARKER in steps[-1].message.content
    for lead in steps[:-1]:
        assert CANARY not in lead.message.content


def test_decomposition_build_emits_subtasks(sample_attack):
    strat = CrescendoDecompositionStrategy(steps=4)
    steps = strat.build(sample_attack, CANARY)
    assert len(steps) == 5  # 4 sub-tasks + scored ask
    subtasks = steps[:-1]
    assert all(s.scored is False and s.expect_response is True for s in subtasks)
    # Sub-tasks decompose the benign objective and never carry the live canary.
    joined = " ".join(s.message.content for s in subtasks)
    assert "INJECTOK-" in joined  # the benign prefix convention is referenced
    assert CANARY not in joined


# --------------------------------------------------------------------------- #
# Constructor validation (inherited from CrescendoStrategy) + factory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cls", [CrescendoReplyReferencingStrategy, CrescendoDecompositionStrategy]
)
def test_rejects_zero_steps(cls):
    with pytest.raises(StrategyError):
        cls(steps=0)


@pytest.mark.parametrize("name", ["crescendo_reply", "crescendo_decompose"])
def test_build_strategy_constructs_by_name(name):
    strat = build_strategy(name)
    assert isinstance(strat, AttackStrategy)
    assert strat.name == name


def test_build_strategy_forwards_steps_kwarg():
    strat = build_strategy("crescendo_decompose", steps=2)
    assert isinstance(strat, CrescendoDecompositionStrategy)
    assert strat.steps == 2
