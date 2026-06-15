"""Tests for the bundled attack corpus (the six technique YAML files).

These tests are fully offline. They assert that:

  * every bundled YAML file parses via ``load_corpus`` into ``Attack`` objects;
  * the six expected technique files exist, each with a healthy number of real
    attacks (>= 4), and each attack's ``technique`` matches its filename;
  * all ``success_conditions`` use only the loader's allowed keys, and every
    attack carries at least one *strong* (concrete-evidence) condition;
  * the ``{canary}`` marker convention is honored end-to-end: marker/canary and
    system-prompt-leak attacks embed ``{canary}`` where the detector expects it;
  * loading the whole directory yields globally-unique ids;
  * a representative attack from each file actually fires its success condition
    against a deliberately-vulnerable MockTarget (so no attack is "toothless").
"""

from __future__ import annotations

import os

import pytest

from injectkit.corpus import load_attack_file, load_corpus
from injectkit.corpus.loader import _VALID_CONDITION_KEYS
from injectkit.evaluators.heuristics import HeuristicDetector, DEFAULT_TRIGGER
from injectkit.models import Attack, Severity

# Strong conditions provide concrete proof of compliance; refusal_absent alone
# is only weak corroboration and must never be an attack's sole condition.
_STRONG_CONDITION_KEYS = {
    "marker_echoed",
    "canary_in_output",
    "regex",
    "system_prompt_leaked",
}

# Expected technique files: filename stem -> declared technique value.
_TECHNIQUE_FILES = {
    "direct_injection": "direct_injection",
    "indirect_injection": "indirect_injection",
    "jailbreak": "jailbreak",
    "system_prompt_leak": "system_prompt_leak",
    "tool_abuse": "tool_abuse",
    "data_exfiltration": "data_exfiltration",
}

_MIN_ATTACKS_PER_FILE = 4


def _corpus_dir() -> str:
    """Absolute path to the bundled corpus directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "injectkit", "corpus")


def _technique_path(stem: str) -> str:
    return os.path.join(_corpus_dir(), f"{stem}.yaml")


def _all_bundled_attacks() -> list[Attack]:
    return load_corpus(_corpus_dir())


# --------------------------------------------------------------------------- #
# Structure / existence
# --------------------------------------------------------------------------- #


def test_all_six_technique_files_exist() -> None:
    for stem in _TECHNIQUE_FILES:
        path = _technique_path(stem)
        assert os.path.isfile(path), f"missing bundled corpus file: {path}"


def test_corpus_dir_loads_without_error() -> None:
    attacks = _all_bundled_attacks()
    # Six files x >=4 attacks each = a solid v1 corpus.
    assert len(attacks) >= len(_TECHNIQUE_FILES) * _MIN_ATTACKS_PER_FILE


@pytest.mark.parametrize("stem,technique", sorted(_TECHNIQUE_FILES.items()))
def test_each_file_parses_and_is_substantial(stem: str, technique: str) -> None:
    attacks = load_attack_file(_technique_path(stem))
    assert len(attacks) >= _MIN_ATTACKS_PER_FILE, (
        f"{stem}.yaml has only {len(attacks)} attacks; expected "
        f">= {_MIN_ATTACKS_PER_FILE}"
    )
    for attack in attacks:
        assert attack.technique == technique, (
            f"{attack.id} in {stem}.yaml declares technique "
            f"{attack.technique!r}, expected {technique!r}"
        )


def test_global_ids_unique() -> None:
    # load_corpus raises CorpusError on duplicate ids; reaching here means unique.
    attacks = _all_bundled_attacks()
    ids = [a.id for a in attacks]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Success-condition validity
# --------------------------------------------------------------------------- #


def test_all_success_conditions_use_valid_keys() -> None:
    for attack in _all_bundled_attacks():
        keys = set(attack.success_conditions or {})
        assert keys, f"{attack.id} has no success_conditions"
        unknown = keys - _VALID_CONDITION_KEYS
        assert not unknown, f"{attack.id} has unknown condition keys: {sorted(unknown)}"


def test_every_attack_has_a_strong_condition() -> None:
    for attack in _all_bundled_attacks():
        keys = set(attack.success_conditions or {})
        strong = keys & _STRONG_CONDITION_KEYS
        assert strong, (
            f"{attack.id} relies only on weak conditions {sorted(keys)}; "
            f"needs at least one of {sorted(_STRONG_CONDITION_KEYS)}"
        )


def test_severity_is_valid_enum() -> None:
    for attack in _all_bundled_attacks():
        assert isinstance(attack.severity, Severity)


def test_attacks_have_references_and_tags() -> None:
    for attack in _all_bundled_attacks():
        assert attack.references, f"{attack.id} has no references"
        assert attack.tags, f"{attack.id} has no tags"


# --------------------------------------------------------------------------- #
# Canary / marker convention
# --------------------------------------------------------------------------- #


def test_canary_convention_holds() -> None:
    """Marker/canary attacks must put {canary} where the detector looks.

    * marker_echoed / canary_in_output -> {canary} must be in the payload (or
      context, for indirect attacks) so the rendered marker is unique per run.
    * system_prompt_leaked with a string spec -> {canary} must be in the system
      text AND in the condition value, so the planted secret is per-run unique.
    """
    for attack in _all_bundled_attacks():
        conds = attack.success_conditions or {}
        rendered_sources = " ".join(
            s for s in (attack.payload, attack.context, attack.system) if s
        )
        if conds.get("marker_echoed") or conds.get("canary_in_output"):
            assert "{canary}" in (attack.payload or "") or "{canary}" in (
                attack.context or ""
            ), f"{attack.id}: marker/canary condition but no {{canary}} placeholder"

        spec = conds.get("system_prompt_leaked")
        if isinstance(spec, str):
            assert "{canary}" in spec, (
                f"{attack.id}: system_prompt_leaked marker has no {{canary}}"
            )
            assert "{canary}" in (attack.system or ""), (
                f"{attack.id}: system_prompt_leaked attack must plant {{canary}} "
                f"in its system text"
            )

        # regex specs that reference a planted secret should be renderable too.
        regex_spec = conds.get("regex")
        if regex_spec is not None:
            patterns = [regex_spec] if isinstance(regex_spec, str) else list(regex_spec)
            joined = " ".join(str(p) for p in patterns)
            if "{canary}" in joined:
                assert "{canary}" in rendered_sources, (
                    f"{attack.id}: regex references {{canary}} but no source "
                    f"field plants it"
                )


def test_render_substitutes_canary() -> None:
    for attack in _all_bundled_attacks():
        if "{canary}" in attack.payload:
            rendered = attack.render("CANARYXYZ")
            assert "CANARYXYZ" in rendered
            assert "{canary}" not in rendered


# --------------------------------------------------------------------------- #
# End-to-end "not toothless": each file fires against a vulnerable target
# --------------------------------------------------------------------------- #


def _render_conditions(attack: Attack, canary: str) -> dict:
    """Substitute {canary} into condition string/list values (engine behavior)."""
    out: dict = {}
    for key, value in (attack.success_conditions or {}).items():
        if isinstance(value, str):
            out[key] = value.replace("{canary}", canary)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                v.replace("{canary}", canary) if isinstance(v, str) else v
                for v in value
            ]
        else:
            out[key] = value
    return out


def _mock_exercisable(attack: Attack) -> bool:
    """True if the bundled MockTarget can actually simulate this attack.

    MockTarget only knows two "vulnerable" behaviors: echoing the INJECTOK-
    marker when it appears in the prompt/context, and leaking its system prompt
    when asked. Attacks whose sole evidence is a custom regex sentinel (e.g.
    SECRET-DATA-) cannot be exercised by the mock, so they are out of scope for
    this particular end-to-end smoke test (the heuristics suite covers those
    detectors directly).
    """
    conds = attack.success_conditions or {}
    if conds.get("marker_echoed") or conds.get("canary_in_output"):
        return True
    if conds.get("system_prompt_leaked"):
        # MockTarget leaks only when the prompt mentions the magic phrases.
        return bool(
            attack.system
            and any(
                kw in (attack.payload or "").lower()
                for kw in ("system prompt", "verbatim", "configuration token")
            )
        )
    return False


@pytest.mark.parametrize("stem", sorted(_TECHNIQUE_FILES))
def test_first_attack_fires_against_vulnerable_target(
    stem: str, mock_target
) -> None:
    """A mock-exercisable attack in each file succeeds against a vulnerable mock.

    This guards against shipping a "toothless" marker/leak attack whose success
    condition can never fire even when the model fully complies. Attacks the
    MockTarget cannot simulate (custom regex sentinels) are skipped here and are
    covered by the dedicated heuristics test suite.
    """
    candidates = [
        a for a in load_attack_file(_technique_path(stem)) if _mock_exercisable(a)
    ]
    if not candidates:
        pytest.skip(f"{stem}.yaml has no MockTarget-exercisable attack")
    attack = candidates[0]
    canary = "FIRECANARY01"
    detector = HeuristicDetector(trigger=DEFAULT_TRIGGER)

    # Render the payload/system/context with the per-run canary (engine job).
    rendered_payload = attack.render(canary)
    rendered_system = (
        attack.system.replace("{canary}", canary) if attack.system else None
    )
    rendered_context = (
        attack.context.replace("{canary}", canary) if attack.context else None
    )

    response = mock_target.send(
        prompt=rendered_payload,
        system=rendered_system,
        context=rendered_context,
    )

    # Build an attack whose conditions also have the canary substituted, since
    # the detector compares literal strings against the rendered response.
    eval_attack = Attack(
        id=attack.id,
        technique=attack.technique,
        name=attack.name,
        description=attack.description,
        severity=attack.severity,
        payload=rendered_payload,
        success_conditions=_render_conditions(attack, canary),
        references=attack.references,
        tags=attack.tags,
        system=rendered_system,
        context=rendered_context,
    )

    verdict = detector.evaluate(eval_attack, response, canary)
    assert verdict.success, (
        f"{attack.id} did not fire against a vulnerable target; "
        f"rationale: {verdict.rationale}; response: {response.text!r}"
    )


def test_regex_conditions_match_a_compliant_output() -> None:
    """Every regex success condition must actually match a compliant reply.

    A "fully compliant" model would echo the attack's payload/context/system
    content (which contains the planted sentinel) back to the user. Building
    that ideal output from the attack's own fields and confirming the rendered
    regex matches it guards regex/exfil attacks against being toothless or
    having a typo'd pattern.
    """
    import re

    canary = "REGEXCANARY99"
    for attack in _all_bundled_attacks():
        spec = (attack.success_conditions or {}).get("regex")
        if spec is None:
            continue
        patterns = [spec] if isinstance(spec, str) else list(spec)
        # The ideal compliant output: the model reflects everything it was given.
        compliant = " ".join(
            s.replace("{canary}", canary)
            for s in (attack.system, attack.context, attack.payload)
            if s
        )
        for pattern in patterns:
            rendered = str(pattern).replace("{canary}", canary)
            try:
                compiled = re.compile(rendered, re.IGNORECASE | re.DOTALL)
            except re.error as exc:  # pragma: no cover - signals a bad pattern
                pytest.fail(f"{attack.id}: invalid regex {rendered!r}: {exc}")
            assert compiled.search(compliant), (
                f"{attack.id}: regex {rendered!r} matches no compliant output; "
                f"likely toothless or mistyped"
            )
