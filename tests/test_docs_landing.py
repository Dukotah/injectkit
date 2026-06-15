"""Unit tests for the docs/ landing page and the finalized README.

These tests are fully offline (no network, no SDK). They guard the docs against
regressions in three areas the project cares about:

1. The landing page is self-contained (no external CSS/JS/font dependencies),
   so GitHub Pages renders it identically to a local open.
2. The authorized-use / ethics posture is present and prominent in both the
   landing page and the README (a hard project requirement).
3. The README tells the intended story in order: problem -> demo -> install ->
   usage -> Action -> how it works -> contributing -> ethics.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"
README = REPO_ROOT / "README.md"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def html() -> str:
    assert DOCS_INDEX.is_file(), f"missing landing page: {DOCS_INDEX}"
    return DOCS_INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    assert README.is_file(), f"missing README: {README}"
    return README.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# A tiny well-formedness checker: every opened tag is closed (ignoring voids).
# --------------------------------------------------------------------------- #
class _TagBalancer(HTMLParser):
    VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        # self-closing like <br/> — nothing to push
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
            return
        if self.stack[-1] != tag:
            # allow implicit closing only if tag appears somewhere in the stack
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
            else:
                self.errors.append(f"mismatched </{tag}>")
        else:
            self.stack.pop()


# --------------------------------------------------------------------------- #
# Landing page: structure & self-containment
# --------------------------------------------------------------------------- #
def test_landing_page_exists_and_is_html5(html: str) -> None:
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert '<html lang="en"' in html
    assert "<title>" in html and "injectkit" in html.lower()


def test_landing_page_tags_balanced(html: str) -> None:
    parser = _TagBalancer()
    parser.feed(html)
    assert not parser.errors, f"unbalanced tags: {parser.errors}"
    # everything except the implicit <html>/<body> should be closed
    leftover = [t for t in parser.stack if t not in {"html", "body"}]
    assert not leftover, f"unclosed tags: {leftover}"


def test_landing_page_has_no_external_dependencies(html: str) -> None:
    """No external CSS/JS/fonts/images — must be a single self-contained file."""
    # No <script> tags at all (page is static).
    assert "<script" not in html.lower(), "landing page must ship no JS"

    # No stylesheet <link rel="stylesheet" ...>; only the inline data-URI favicon.
    for m in re.finditer(r"<link\b[^>]*>", html, flags=re.IGNORECASE):
        tag = m.group(0).lower()
        if 'rel="stylesheet"' in tag or "rel=stylesheet" in tag:
            pytest.fail("landing page must not link external stylesheets")

    # CSS lives inline.
    assert "<style>" in html

    # Any href/src that loads a resource must not be an http(s) asset import.
    # External anchor links to github.com are fine (they're navigation, not
    # asset loads), but there must be zero src= pulling remote assets.
    src_urls = re.findall(r'\bsrc\s*=\s*"([^"]+)"', html, flags=re.IGNORECASE)
    for url in src_urls:
        assert not url.startswith("http"), f"remote src asset not allowed: {url}"

    # No webfont imports.
    assert "fonts.googleapis" not in html
    assert "@import" not in html


def test_landing_page_covers_required_sections(html: str) -> None:
    lowered = html.lower()
    for needle in [
        "install",
        "usage",
        "how it works",
        "corpus",
        "contributing",
        "ethics",
    ]:
        assert needle in lowered, f"landing page missing section: {needle!r}"


def test_landing_page_shows_the_scan_demo(html: str) -> None:
    assert "injectkit scan" in html
    # The marker/canary mechanic is the signature detection idea.
    assert "INJECTOK-" in html
    assert "{canary}" in html


def test_landing_page_lists_all_six_techniques(html: str) -> None:
    for technique in [
        "direct_injection",
        "indirect_injection",
        "jailbreak",
        "system_prompt_leak",
        "tool_abuse",
        "data_exfiltration",
    ]:
        assert technique in html, f"technique not advertised: {technique}"


def test_landing_page_has_prominent_authorized_use_banner(html: str) -> None:
    lowered = html.lower()
    assert "authorized" in lowered
    assert "own" in lowered
    assert ("defensive" in lowered) or ("defender" in lowered)
    # Must appear before the closing of the hero region (i.e. high on the page),
    # not buried in the footer only.
    first_authorized = lowered.find("authorized")
    assert first_authorized != -1
    assert first_authorized < len(lowered) * 0.5, "authorized-use notice not prominent"


def test_landing_page_links_to_github_repo(html: str) -> None:
    assert "github.com/Dukotah/injectkit" in html


def test_landing_page_states_mit_license(html: str) -> None:
    assert "MIT" in html


# --------------------------------------------------------------------------- #
# README: story order & content
# --------------------------------------------------------------------------- #
def test_readme_has_authorized_use_notice_near_top(readme: str) -> None:
    head = readme[:900].lower()
    assert "authorized" in head
    assert "defensive" in head


def test_readme_sections_in_intended_order(readme: str) -> None:
    """problem -> demo -> install -> usage -> Action -> how it works ->
    contributing -> ethics."""
    order = [
        "## the problem",
        "## demo",
        "## install",
        "## usage",
        "## github action",
        "## how it works",
        "## contributing",
        "## ethics",
    ]
    lowered = readme.lower()
    positions = []
    for heading in order:
        idx = lowered.find(heading)
        assert idx != -1, f"README missing section heading: {heading!r}"
        positions.append(idx)
    assert positions == sorted(positions), (
        f"README sections out of order: {list(zip(order, positions))}"
    )


def test_readme_documents_the_github_action(readme: str) -> None:
    lowered = readme.lower()
    assert "uses: dukotah/injectkit@v1" in lowered
    assert "upload-sarif" in lowered
    assert "anthropic_api_key" in lowered


def test_readme_lists_report_formats(readme: str) -> None:
    for fmt in ["terminal", "json", "markdown", "sarif", "html"]:
        assert fmt in readme.lower(), f"README missing report format: {fmt}"


def test_readme_documents_install_extras(readme: str) -> None:
    for extra in ["injectkit[anthropic]", "injectkit[mcp]", "injectkit[all]"]:
        assert extra in readme, f"README missing install extra: {extra}"


def test_readme_explains_canary_marker_detection(readme: str) -> None:
    assert "{canary}" in readme
    assert "INJECTOK" in readme


def test_readme_and_docs_agree_on_techniques(html: str, readme: str) -> None:
    techniques = [
        "direct_injection",
        "indirect_injection",
        "jailbreak",
        "system_prompt_leak",
        "tool_abuse",
        "data_exfiltration",
    ]
    for technique in techniques:
        assert technique in readme, f"README missing technique: {technique}"
        assert technique in html, f"docs missing technique: {technique}"
