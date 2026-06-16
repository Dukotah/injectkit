"""Offline doc-content guards for the v0.3.0 README, landing page, and changelog.

These tests are fully offline (no network, no model, no dataset, no SDK, no
translation). They guard the user-facing docs against drift on the v0.3.0
additions, so the marketing surface stays honest and complete:

1. ``README.md`` advertises every v0.3.0 capability (the cipher transforms, the
   semantic ``translate`` transform, ``crescendo_reply``, the named attackers
   PAIR/TAP/AutoDAN/GPTFUZZER, the white-box GCG optimizer, and the 5-class
   response grade), links ``docs/RESEARCH.md``, and keeps the ethics /
   research-use posture and the honest frontier caveat prominent.
2. ``docs/index.html`` carries the same v0.3.0 story and stays self-contained
   (no JS, no remote asset imports) so GitHub Pages renders it identically.
3. ``CHANGELOG.md`` has a well-formed ``[0.3.0]`` entry that reflects the shipped
   code and keeps the boolean-success-frozen guarantee.

This module documents only — it changes no Python logic.

Cited research grounding for the techniques these docs describe lives in
``docs/RESEARCH.md`` (CipherChat 2308.06463, ArtPrompt 2402.11753, low-resource
2310.02446 / MultiJail 2310.06474, Crescendo 2404.01833, PAIR 2310.08419, TAP
2312.02119, AutoDAN 2310.04451, GPTFUZZER 2309.10253, AmpleGCG 2404.07921,
SoK Prompt Hacking 2410.13901).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: v0.3.0 cipher transform registry keys the docs must advertise.
_CIPHER_KEYS = ("caesar", "atbash", "morse", "unicode_escape", "artprompt", "selfcipher")
#: v0.3.0 semantic transform key.
_TRANSLATE_KEY = "translate"
#: v0.3.0 named attacker keys.
_ATTACKER_KEYS = ("pair", "tap", "autodan", "gptfuzzer", "gcg")
#: v0.3.0 5-class response-grade keys.
_RESPONSE_CLASSES = (
    "reject_irrelevant",
    "reject_safety",
    "too_long",
    "partial",
    "full",
)
#: Primary citations (arXiv ids) the docs should carry for the new families.
_CITATIONS = (
    "2308.06463",  # CipherChat
    "2402.11753",  # ArtPrompt
    "2310.02446",  # low-resource translation
    "2310.06474",  # MultiJail
    "2404.01833",  # Crescendo
    "2310.08419",  # PAIR
    "2312.02119",  # TAP
    "2310.04451",  # AutoDAN
    "2309.10253",  # GPTFUZZER
    "2404.07921",  # AmpleGCG
    "2410.13901",  # SoK Prompt Hacking (5-class)
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def readme() -> str:
    assert README.is_file(), f"missing README: {README}"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html() -> str:
    assert DOCS_INDEX.is_file(), f"missing landing page: {DOCS_INDEX}"
    return DOCS_INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def changelog() -> str:
    assert CHANGELOG.is_file(), f"missing CHANGELOG: {CHANGELOG}"
    return CHANGELOG.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# README — v0.3.0 capabilities advertised
# --------------------------------------------------------------------------- #
def test_readme_has_v3_whats_new_section(readme: str) -> None:
    lowered = readme.lower()
    assert "what's new in v0.3.0" in lowered
    # The v0.2.0 section must survive (history, not overwrite).
    assert "what's new in v0.2.0" in lowered
    # The v0.3.0 section appears above the v0.2.0 section.
    assert lowered.find("what's new in v0.3.0") < lowered.find("what's new in v0.2.0")


def test_readme_advertises_cipher_transforms(readme: str) -> None:
    for key in _CIPHER_KEYS:
        assert key in readme, f"README missing cipher transform: {key}"


def test_readme_advertises_translate_transform(readme: str) -> None:
    assert _TRANSLATE_KEY in readme
    lowered = readme.lower()
    assert "low-resource" in lowered
    assert "semantic" in lowered


def test_readme_advertises_crescendo_reply(readme: str) -> None:
    assert "crescendo_reply" in readme


def test_readme_advertises_named_attackers(readme: str) -> None:
    for key in _ATTACKER_KEYS:
        assert key in readme, f"README missing attacker: {key}"
    # GCG must be flagged white-box / HuggingFace-only.
    lowered = readme.lower()
    assert "white-box" in lowered
    assert "huggingface" in lowered


def test_readme_advertises_5class_scoring(readme: str) -> None:
    for cls in _RESPONSE_CLASSES:
        assert cls in readme, f"README missing response class: {cls}"
    # The boolean headline stays frozen: success only on full.
    assert "is_success" in readme or "only on `full`" in readme.lower()


def test_readme_cites_research_doc(readme: str) -> None:
    assert "docs/RESEARCH.md" in readme


def test_readme_cites_primary_sources(readme: str) -> None:
    for arxiv in _CITATIONS:
        assert arxiv in readme, f"README missing citation arXiv:{arxiv}"


def test_readme_carries_honest_frontier_caveat(readme: str) -> None:
    lowered = readme.lower()
    assert "overstated" in lowered
    assert "90%" in readme
    assert "flagship" in lowered
    assert "single dig" in lowered


def test_readme_keeps_ethics_and_benign_canary_posture(readme: str) -> None:
    import re

    lowered = readme.lower()
    assert "authorized" in lowered
    assert "defensive" in lowered
    assert "benign canary" in lowered
    # No harmful suffix artifact is bundled (GCG safety posture); collapse
    # whitespace so a soft line-wrap in the prose does not break the match.
    collapsed = re.sub(r"\s+", " ", lowered)
    assert "no harmful suffix artifact is bundled" in collapsed


def test_readme_current_release_is_v3(readme: str) -> None:
    assert "**v0.3.0**" in readme


# --------------------------------------------------------------------------- #
# Landing page — v0.3.0 story, still self-contained
# --------------------------------------------------------------------------- #
def test_landing_has_v3_section(html: str) -> None:
    assert 'id="whatsnew-v3"' in html
    assert "New in v0.3.0" in html


def test_landing_advertises_v3_transforms_and_attackers(html: str) -> None:
    # Transforms / strategy keys appear verbatim (registry keys).
    for key in (*_CIPHER_KEYS, _TRANSLATE_KEY, "crescendo_reply"):
        assert key in html, f"landing page missing: {key}"
    # Attackers may be styled in caps in headings (PAIR/TAP/...) or shown as the
    # lowercase registry key (gcg) — accept either spelling.
    lowered = html.lower()
    for key in _ATTACKER_KEYS:
        assert key in lowered, f"landing page missing attacker: {key}"


def test_landing_advertises_5class(html: str) -> None:
    for cls in _RESPONSE_CLASSES:
        assert cls in html, f"landing page missing response class: {cls}"


def test_landing_cites_primary_sources(html: str) -> None:
    for arxiv in _CITATIONS:
        assert arxiv in html, f"landing page missing citation arXiv:{arxiv}"


def test_landing_carries_honest_frontier_caveat(html: str) -> None:
    lowered = html.lower()
    assert "overstated" in lowered
    assert "90%" in html
    assert "flagship" in lowered
    assert "single dig" in lowered


def test_landing_links_research_doc(html: str) -> None:
    assert "docs/RESEARCH.md" in html


def test_landing_stays_self_contained(html: str) -> None:
    """No JS and no remote asset imports — the page must stay one static file."""
    import re

    assert "<script" not in html.lower(), "landing page must ship no JS"
    assert "@import" not in html
    assert "fonts.googleapis" not in html
    for url in re.findall(r'\bsrc\s*=\s*"([^"]+)"', html, flags=re.IGNORECASE):
        assert not url.startswith("http"), f"remote src asset not allowed: {url}"


def test_landing_v3_section_precedes_v2(html: str) -> None:
    assert html.find('id="whatsnew-v3"') < html.find('id="whatsnew"')


# --------------------------------------------------------------------------- #
# CHANGELOG — well-formed [0.3.0] entry that reflects shipped code
# --------------------------------------------------------------------------- #
def test_changelog_has_v3_entry(changelog: str) -> None:
    assert "## [0.3.0]" in changelog
    # Ordered newest-first: 0.3.0 above 0.2.0 above 0.1.0.
    assert changelog.find("## [0.3.0]") < changelog.find("## [0.2.0]")
    assert changelog.find("## [0.2.0]") < changelog.find("## [0.1.0]")


def test_changelog_v3_lists_every_module(changelog: str) -> None:
    for module in (
        "transforms/ciphers.py",
        "transforms/translate.py",
        "attacks/multiturn.py",
        "attackers/registry.py",
        "attackers/whitebox_base.py",
        "evaluators/response_class.py",
    ):
        assert module in changelog, f"CHANGELOG.md missing module: {module}"


def test_changelog_v3_keeps_success_frozen(changelog: str) -> None:
    lowered = changelog.lower()
    assert "frozen" in lowered
    assert "is_success" in changelog
    assert "only on `full`" in lowered or "succeeds only on `full`" in lowered


def test_changelog_v3_cites_sources(changelog: str) -> None:
    for arxiv in _CITATIONS:
        assert arxiv in changelog, f"CHANGELOG.md missing citation arXiv:{arxiv}"


def test_changelog_v3_has_compare_link(changelog: str) -> None:
    assert "[0.3.0]: https://github.com/Dukotah/injectkit/compare/v0.2.0...v0.3.0" in changelog
    assert "[Unreleased]: https://github.com/Dukotah/injectkit/compare/v0.3.0...HEAD" in changelog
