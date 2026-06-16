"""Tests for the multi-turn attack strategies (injectkit/attacks/multiturn.py).

All offline and deterministic — strategies are pure (no network, no models), and
the end-to-end drive uses the in-repo FakeConversationalTarget fixture. We verify
the frozen AttackStrategy contract (exactly one scored step, canary preserved in
the scored turn, lead-in turns benign and canary-free) and that delivering the
built sequence to a vulnerable conversational target scores a benign-marker hit.
"""

from __future__ import annotations

import pytest

from injectkit.attacks.base import AttackStep, AttackStrategy
from injectkit.attacks.multiturn import (
    MULTI_TURN_STRATEGIES,
    ContextOverflowStrategy,
    CrescendoStrategy,
    ManyShotStrategy,
    PersonaPrimingStrategy,
    StrategyError,
    build_strategy,
)
from injectkit.models import TargetResponse
from injectkit.targets.conversational import ChatMessage

CANARY = "cana12345"


def _all_strategies() -> list[AttackStrategy]:
    return [
        CrescendoStrategy(),
        ManyShotStrategy(),
        ContextOverflowStrategy(),
        PersonaPrimingStrategy(),
    ]


# --------------------------------------------------------------------------- #
# Contract conformance shared by every strategy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_strategy_satisfies_protocol_and_has_name(strat):
    assert isinstance(strat, AttackStrategy)
    assert isinstance(strat.name, str) and strat.name


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_build_returns_steps_with_exactly_one_scored(strat, sample_attack):
    steps = strat.build(sample_attack, CANARY)
    assert len(steps) >= 2  # multi-turn => at least a lead-in plus the ask
    assert all(isinstance(s, AttackStep) for s in steps)
    scored = [s for s in steps if s.scored]
    assert len(scored) == 1, "exactly one step must be scored"


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_scored_step_is_final_user_turn_with_rendered_canary(strat, sample_attack):
    steps = strat.build(sample_attack, CANARY)
    scored = next(s for s in steps if s.scored)
    # The scored step is always the final turn.
    assert steps[-1] is scored
    assert scored.message.role == "user"
    assert scored.expect_response is True
    # The live canary marker is present only in the scored seed payload turn.
    assert f"INJECTOK-{CANARY}" in scored.message.content


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_lead_in_turns_do_not_leak_live_canary(strat, sample_attack):
    steps = strat.build(sample_attack, CANARY)
    for s in steps[:-1]:
        assert CANARY not in s.message.content, (
            "only the scored final turn may carry the live canary"
        )


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_build_is_deterministic(strat, sample_attack):
    a = strat.build(sample_attack, CANARY)
    b = strat.build(sample_attack, CANARY)
    assert [(s.message.role, s.message.content, s.scored) for s in a] == [
        (s.message.role, s.message.content, s.scored) for s in b
    ]


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_end_to_end_scores_marker_against_vulnerable_target(
    strat, sample_attack, fake_conversational_target
):
    """Deliver the built turns to a vulnerable conversational target.

    The engine contract: send history (expect_response steps) and then score the
    final turn's response. Here we deliver the FULL message list at once (the
    FakeConversationalTarget answers the latest user turn), which is the scored
    turn, and assert the benign marker came back.
    """
    steps = strat.build(sample_attack, CANARY)
    messages = [s.message for s in steps]
    resp = fake_conversational_target.chat(messages)
    assert isinstance(resp, TargetResponse)
    assert resp.refused is False
    assert resp.text == f"INJECTOK-{CANARY}"


@pytest.mark.parametrize("strat", _all_strategies(), ids=lambda s: s.name)
def test_end_to_end_refusing_target_does_not_score(
    strat, sample_attack, fake_conversational_target
):
    fake_conversational_target.vulnerable = False
    steps = strat.build(sample_attack, CANARY)
    resp = fake_conversational_target.chat([s.message for s in steps])
    assert resp.refused is True
    assert f"INJECTOK-{CANARY}" not in resp.text


# --------------------------------------------------------------------------- #
# Crescendo specifics
# --------------------------------------------------------------------------- #


def test_crescendo_step_count_matches_lead_ins(sample_attack):
    strat = CrescendoStrategy(steps=2)
    steps = strat.build(sample_attack, CANARY)
    # 2 lead-ins + 1 scored ask
    assert len(steps) == 3
    assert sum(1 for s in steps if not s.scored) == 2
    for lead in steps[:-1]:
        assert lead.message.role == "user"
        assert lead.expect_response is True  # become real history


def test_crescendo_rejects_zero_steps():
    with pytest.raises(StrategyError):
        CrescendoStrategy(steps=0)


# --------------------------------------------------------------------------- #
# Many-shot specifics
# --------------------------------------------------------------------------- #


def test_many_shot_primes_with_example_markers_not_canary(sample_attack):
    strat = ManyShotStrategy(shots=3)
    steps = strat.build(sample_attack, CANARY)
    # 3 user + 3 assistant scripted pairs + 1 scored ask.
    assert len(steps) == 7
    assistants = [s for s in steps if s.message.role == "assistant"]
    assert len(assistants) == 3
    for a in assistants:
        assert a.scored is False
        assert a.expect_response is False  # scripted fake history
        assert a.message.content.startswith("INJECTOK-EXAMPLE")
        assert CANARY not in a.message.content


def test_many_shot_rejects_zero_shots():
    with pytest.raises(StrategyError):
        ManyShotStrategy(shots=0)


# --------------------------------------------------------------------------- #
# Context-overflow specifics
# --------------------------------------------------------------------------- #


def test_context_overflow_produces_bulky_filler_history(sample_attack):
    strat = ContextOverflowStrategy(filler_turns=3, filler_repeat=5)
    steps = strat.build(sample_attack, CANARY)
    fillers = [
        s for s in steps if s.message.role == "user" and "notes batch" in s.message.content
    ]
    assert len(fillers) == 3
    for f in fillers:
        assert f.scored is False
        assert f.expect_response is False
        # filler_repeat makes the body deterministically long.
        assert f.message.content.count("authorized test scaffolding") == 5


def test_context_overflow_validates_args():
    with pytest.raises(StrategyError):
        ContextOverflowStrategy(filler_turns=0)
    with pytest.raises(StrategyError):
        ContextOverflowStrategy(filler_repeat=0)


# --------------------------------------------------------------------------- #
# Persona-priming specifics
# --------------------------------------------------------------------------- #


def test_persona_priming_spreads_persona_over_turns(sample_attack):
    strat = PersonaPrimingStrategy(persona="DAN")
    steps = strat.build(sample_attack, CANARY)
    priming = steps[:-1]
    assert len(priming) == 2
    for s in priming:
        assert "DAN" in s.message.content
        assert s.scored is False
        assert s.expect_response is True


def test_persona_priming_rejects_empty_persona():
    with pytest.raises(StrategyError):
        PersonaPrimingStrategy(persona="   ")


# --------------------------------------------------------------------------- #
# build_strategy factory
# --------------------------------------------------------------------------- #


def test_build_strategy_constructs_known_strategies():
    for name in MULTI_TURN_STRATEGIES:
        strat = build_strategy(name)
        assert isinstance(strat, AttackStrategy)
        assert strat.name == name


def test_build_strategy_forwards_kwargs():
    strat = build_strategy("crescendo", steps=1)
    assert isinstance(strat, CrescendoStrategy)
    assert strat.steps == 1


def test_build_strategy_unknown_raises():
    with pytest.raises(StrategyError):
        build_strategy("does_not_exist")


def test_build_strategy_bad_kwarg_raises_strategy_error():
    # An unknown constructor kwarg must surface as a StrategyError (the
    # documented failure type), not a raw TypeError that crashes the run.
    with pytest.raises(StrategyError):
        build_strategy("crescendo", not_a_real_param=1)


def test_build_strategy_invalid_value_raises_strategy_error():
    # Constructor-level validation (steps < 1) already raises StrategyError and
    # must propagate unchanged through build_strategy.
    with pytest.raises(StrategyError):
        build_strategy("crescendo", steps=0)
