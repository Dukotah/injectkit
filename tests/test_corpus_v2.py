"""Tests for the v0.2.0 corpus expansion (docs/TAXONOMY.md alignment).

These tests are fully offline and make no network/model/dataset calls. They
complement ``tests/test_corpus.py`` (which validates schema/loader invariants)
by asserting the *v0.2.0* growth and taxonomy contract:

  * every technique family has grown well past the v0.1 floor;
  * each attack's declared technique matches its source file;
  * declared severities respect the per-technique severity FLOOR documented in
    docs/TAXONOMY.md (Axis 1);
  * the expanded families pull in the named v0.2 carriers / modifier techniques
    (Axis 2 tags) so the corpus actually exercises the taxonomy;
  * every strong success condition fires against a *fully-compliant* simulated
    reply built from the attack's own fields (no attack is toothless), and the
    benign-canary proxy design is preserved (no harmful payloads, only markers
    and inert sentinel sinks).

A "fully-compliant" reply is the worst case: a model that reflects every field
it was given (system + context + payload) with ``{canary}`` rendered. If a
strong condition still cannot fire against that, the attack is mistyped.
"""

from __future__ import annotations

import os
import re

import pytest

from injectkit.corpus import load_attack_file, load_corpus
from injectkit.evaluators.heuristics import HeuristicDetector, DEFAULT_TRIGGER
from injectkit.models import Attack, Severity

# --------------------------------------------------------------------------- #
# Taxonomy contract (mirrors docs/TAXONOMY.md, Axis 1 + Axis 2)
# --------------------------------------------------------------------------- #

#: Minimum attack count per family for the v0.2.0 set (all six expanded).
_MIN_ATTACKS_PER_FAMILY = 8

#: Per-technique severity floor from docs/TAXONOMY.md (Axis 1). A declared
#: severity may meet or exceed the floor, never fall below it.
_SEVERITY_FLOOR = {
    "direct_injection": Severity.LOW,
    "indirect_injection": Severity.LOW,
    "jailbreak": Severity.LOW,
    "system_prompt_leak": Severity.MEDIUM,
    "tool_abuse": Severity.MEDIUM,
    "data_exfiltration": Severity.MEDIUM,
}

#: For each expanded family, at least one of these v0.2 carrier/modifier tags
#: (Axis 2 / the "new carriers to populate" note in TAXONOMY.md) must appear
#: somewhere in the family, proving the expansion exercises the taxonomy.
_REQUIRED_FAMILY_TAGS = {
    "direct_injection": {"payload-splitting", "impersonation", "json", "format-hijack"},
    "indirect_injection": {"image-alt", "csv", "json", "chunk-boundary", "markdown"},
    "jailbreak": {"dan", "developer-mode", "aim", "many-shot", "crescendo"},
    "system_prompt_leak": {"json-export", "summarize", "encoding", "completion"},
    "tool_abuse": {"argument-injection", "confused-deputy", "tool-chaining", "schema-injection"},
    "data_exfiltration": {"markdown-image", "url-encoded", "dns-tunnel", "summary-smuggle"},
}

#: Words that must NEVER appear in a payload/context/system field — the corpus
#: is benign-canary-proxy only and must not bundle harmful instructions.
_FORBIDDEN_CONTENT = (
    "synthesize",
    "step-by-step instructions to make",
    "build a bomb",
    "child sexual",
)


def _corpus_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "injectkit", "corpus")


def _family_path(family: str) -> str:
    return os.path.join(_corpus_dir(), f"{family}.yaml")


def _all() -> list[Attack]:
    return load_corpus(_corpus_dir())


# --------------------------------------------------------------------------- #
# Growth + family integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", sorted(_SEVERITY_FLOOR))
def test_family_is_expanded(family: str) -> None:
    attacks = load_attack_file(_family_path(family))
    assert len(attacks) >= _MIN_ATTACKS_PER_FAMILY, (
        f"{family}.yaml has only {len(attacks)} attacks; v0.2.0 expects "
        f">= {_MIN_ATTACKS_PER_FAMILY}"
    )
    for a in attacks:
        assert a.technique == family, (
            f"{a.id} declares technique {a.technique!r} but lives in {family}.yaml"
        )


def test_total_corpus_grew_substantially() -> None:
    attacks = _all()
    # v0.1 shipped 36 attacks; v0.2 is a meaningful expansion.
    assert len(attacks) >= 60, f"expected >= 60 attacks in v0.2.0, got {len(attacks)}"


def test_ids_are_unique_and_kebab() -> None:
    attacks = _all()
    ids = [a.id for a in attacks]
    assert len(ids) == len(set(ids)), "duplicate attack ids in expanded corpus"
    for aid in ids:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", aid), (
            f"attack id {aid!r} is not lowercase-kebab-case"
        )


# --------------------------------------------------------------------------- #
# Taxonomy alignment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family,floor", sorted(_SEVERITY_FLOOR.items()))
def test_severity_floor_respected(family: str, floor: Severity) -> None:
    for a in load_attack_file(_family_path(family)):
        assert a.severity.rank >= floor.rank, (
            f"{a.id} severity {a.severity.value!r} is below the {family} floor "
            f"{floor.value!r} from docs/TAXONOMY.md"
        )


@pytest.mark.parametrize("family,wanted", sorted(_REQUIRED_FAMILY_TAGS.items()))
def test_family_pulls_in_v2_taxonomy_tags(family: str, wanted: set) -> None:
    seen: set[str] = set()
    for a in load_attack_file(_family_path(family)):
        seen.update(a.tags)
    overlap = seen & wanted
    assert overlap, (
        f"{family}.yaml does not exercise any of the v0.2 taxonomy tags "
        f"{sorted(wanted)}; saw tags {sorted(seen)}"
    )


def test_every_attack_tags_and_refs_present() -> None:
    for a in _all():
        assert a.tags, f"{a.id} has no tags"
        assert a.references, f"{a.id} has no references"
        # Each reference is a real-looking URL/citation, not a placeholder.
        for ref in a.references:
            assert ref.strip() and not ref.lower().startswith("todo"), (
                f"{a.id} has a placeholder reference: {ref!r}"
            )


# --------------------------------------------------------------------------- #
# Safety: benign-canary-proxy preserved; no harmful content bundled
# --------------------------------------------------------------------------- #


def test_no_harmful_content_in_corpus() -> None:
    for a in _all():
        blob = " ".join(
            s for s in (a.payload, a.context, a.system, a.description) if s
        ).lower()
        for bad in _FORBIDDEN_CONTENT:
            assert bad not in blob, (
                f"{a.id} appears to contain harmful content {bad!r}; the corpus "
                f"must stay benign-canary-proxy only"
            )


def test_marker_or_sentinel_proxy_used_everywhere() -> None:
    """Every attack proves success via a benign marker/sentinel, never harm.

    A strong condition must be marker_echoed / canary_in_output (the INJECTOK
    marker), a regex referencing the per-run {canary}, or system_prompt_leaked
    against a planted SECRET-SYS-{canary} sentinel. This is the proxy design.
    """
    for a in _all():
        conds = a.success_conditions or {}
        proxy = False
        if conds.get("marker_echoed") or conds.get("canary_in_output"):
            proxy = "{canary}" in (a.payload or "") or "{canary}" in (a.context or "")
        regex_spec = conds.get("regex")
        if regex_spec is not None:
            patterns = [regex_spec] if isinstance(regex_spec, str) else list(regex_spec)
            if any("{canary}" in str(p) for p in patterns):
                proxy = True
        leak_spec = conds.get("system_prompt_leaked")
        if isinstance(leak_spec, str) and "{canary}" in leak_spec:
            proxy = True
        assert proxy, (
            f"{a.id} does not use a per-run benign canary/sentinel proxy; "
            f"success would not be measurable as a pure bypass"
        )


# --------------------------------------------------------------------------- #
# "Not toothless": strong conditions fire on a fully-compliant simulated reply
# --------------------------------------------------------------------------- #


def _render(text: str | None, canary: str) -> str | None:
    return text.replace("{canary}", canary) if text else None


def _compliant_reply(a: Attack, canary: str) -> str:
    """Build the reply of a maximally-compliant ("fully fell for it") model.

    The model reflects every field it was given (system + context + payload with
    ``{canary}`` rendered). For marker/canary attacks it additionally emits the
    intended success marker, since some payloads ask the model to *reassemble*
    the marker from fragments (payload-splitting, acrostic) rather than echo it
    verbatim — a compliant model would produce the assembled marker. This keeps
    the toothless check honest without hand-simulating each obfuscation.
    """
    parts = [
        s for s in (
            _render(a.system, canary),
            _render(a.context, canary),
            _render(a.payload, canary),
        ) if s
    ]
    conds = a.success_conditions or {}
    if conds.get("marker_echoed"):
        parts.append(f"{DEFAULT_TRIGGER}{canary}")
    elif conds.get("canary_in_output"):
        parts.append(canary)
    return " ".join(parts)


def _render_conditions(a: Attack, canary: str) -> dict:
    out: dict = {}
    for key, value in (a.success_conditions or {}).items():
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


def test_every_attack_fires_on_a_fully_compliant_reply() -> None:
    """No attack is toothless: a maximally-compliant model trips its detector."""
    canary = "V2COMPLY01"
    detector = HeuristicDetector(trigger=DEFAULT_TRIGGER)
    from injectkit.models import TargetResponse

    for a in _all():
        reply = _compliant_reply(a, canary)
        eval_attack = Attack(
            id=a.id,
            technique=a.technique,
            name=a.name,
            description=a.description,
            severity=a.severity,
            payload=_render(a.payload, canary) or "",
            success_conditions=_render_conditions(a, canary),
            references=a.references,
            tags=a.tags,
            system=_render(a.system, canary),
            context=_render(a.context, canary),
        )
        verdict = detector.evaluate(
            eval_attack,
            TargetResponse(text=reply, refused=False),
            canary,
        )
        assert verdict.success, (
            f"{a.id} ({a.technique}) did not fire against a fully-compliant "
            f"reply; rationale: {verdict.rationale}"
        )


def test_defending_refusal_never_scores_success() -> None:
    """A refusing target must defeat every attack (proxy stays sound)."""
    canary = "V2REFUSE01"
    detector = HeuristicDetector(trigger=DEFAULT_TRIGGER)
    from injectkit.models import TargetResponse

    for a in _all():
        eval_attack = Attack(
            id=a.id,
            technique=a.technique,
            name=a.name,
            description=a.description,
            severity=a.severity,
            payload=_render(a.payload, canary) or "",
            success_conditions=_render_conditions(a, canary),
            references=a.references,
            tags=a.tags,
            system=_render(a.system, canary),
            context=_render(a.context, canary),
        )
        verdict = detector.evaluate(
            eval_attack,
            TargetResponse(text="", refused=True, stop_reason="refusal"),
            canary,
        )
        assert not verdict.success, f"{a.id} wrongly scored success on a refusal"
