"""Tests for the v0.3.0 injectkit web GUI additions (:mod:`injectkit.web`).

These cover only the v0.3.0 wiring this module added, on top of the v0.2.0 GUI
tests in ``tests/test_web_gui.py``, and stay fully offline/deterministic:

* the new cipher / art-prompt / self-cipher transforms and the semantic
  ``translate`` transform appear in the *mutate* selector,
* an optional-dependency transform (``translate``) is annotated with a friendly
  "needs <dep>" note when its offline translator is absent,
* the named adaptive attackers (pair / tap / autodan / gptfuzzer / gcg) are
  listed in an informational, disabled attacker dropdown, with ``gcg`` flagged
  when torch/transformers is missing,
* the five-class response breakdown renders on the scan results page and its
  ``full`` count equals the report's vulnerable count,
* the small ``_dep_available`` / ``_register_v3_transforms`` /
  ``_response_class_counts`` helpers behave as documented.

The only target used is the built-in offline ``mock`` target; no network, no
models, no datasets, no optional heavy deps are touched.
"""

from __future__ import annotations

from unittest import mock

from injectkit import web
from injectkit.evaluators.response_class import ResponseClass


def _form_html() -> str:
    return web.form_page().decode("utf-8")


# --------------------------------------------------------------------------- #
# Mutate selector: v0.3.0 transforms surface
# --------------------------------------------------------------------------- #
def test_form_lists_v3_cipher_and_art_transforms():
    """The cipher / art / self-cipher transforms appear in the mutate selector."""
    html = _form_html()
    for name in ("caesar", "atbash", "morse", "unicode_escape", "artprompt", "selfcipher"):
        assert f"value='{name}'" in html


def test_form_lists_translate_transform():
    """The semantic low-resource translate transform appears in the selector."""
    html = _form_html()
    assert "value='translate'" in html


def test_list_transforms_registers_v3_transforms():
    """_list_transforms surfaces the v0.3.0 names alongside the v0.2.0 encoders."""
    names = web._list_transforms()
    # v0.2.0 encoder still present
    assert "base64" in names
    # v0.3.0 cipher + translate now present (registered by the helper)
    for name in ("caesar", "atbash", "morse", "artprompt", "selfcipher", "translate"):
        assert name in names
    # the no-op baseline is excluded
    assert "identity" not in names


# --------------------------------------------------------------------------- #
# Optional-dependency annotation (translate)
# --------------------------------------------------------------------------- #
def test_translate_shows_dep_note_when_argostranslate_absent():
    """translate is annotated with a 'needs argostranslate' note when dep absent."""
    with mock.patch.object(web, "_dep_available", return_value=False):
        html = _form_html()
    # the transform still lists, with a friendly note naming the offline extra
    assert "value='translate'" in html
    assert "argostranslate" in html
    assert "will not run until you install" in html


def test_translate_has_no_dep_note_when_dependency_present():
    """With the dep available the translate note is suppressed."""
    with mock.patch.object(web, "_dep_available", return_value=True):
        html = _form_html()
    assert "value='translate'" in html
    assert "needs <code>argostranslate" not in html


# --------------------------------------------------------------------------- #
# Named adaptive-attacker dropdown
# --------------------------------------------------------------------------- #
def test_form_lists_named_attackers():
    """All five named attackers are offered in the (disabled) attacker dropdown."""
    html = _form_html()
    assert "name=attacker" in html
    for name in ("pair", "tap", "autodan", "gptfuzzer", "gcg"):
        assert f"value='{name}'" in html
    # the dropdown is informational only (no attacker-model fields in the GUI)
    assert "disabled" in html
    # the benign-objective + CLI-only contract is spelled out
    assert "benign" in html.lower()
    assert "--attacker" in html


def test_gcg_flagged_when_torch_missing():
    """gcg is annotated as needing torch/transformers when they are absent."""
    # torch/transformers are not installed in CI, so gcg must be flagged.
    def _absent(module: str) -> bool:
        return module not in ("torch", "transformers")

    with mock.patch.object(web, "_dep_available", side_effect=_absent):
        html = _form_html()
    assert "needs torch/transformers" in html


def test_list_attackers_marks_gcg_runnable_by_dep():
    """_list_attackers reports gcg runnable iff torch+transformers import."""
    with mock.patch.object(
        web, "_dep_available", side_effect=lambda m: m in ("torch", "transformers")
    ):
        rows = {name: (kind, runnable) for name, kind, runnable in web._list_attackers()}
    assert rows["gcg"][1] is True  # both deps present -> runnable
    assert rows["gcg"][0] == "white_box"
    # a black-box attacker does not depend on torch
    assert rows["pair"][1] is True
    assert rows["pair"][0] == "black_box"

    with mock.patch.object(web, "_dep_available", return_value=False):
        rows2 = {name: runnable for name, _kind, runnable in web._list_attackers()}
    assert rows2["gcg"] is False  # deps absent -> not runnable
    assert rows2["pair"] is True  # black-box unaffected


# --------------------------------------------------------------------------- #
# 5-class response breakdown on the scan results page
# --------------------------------------------------------------------------- #
def test_scan_results_page_renders_5class_breakdown():
    """The scan results page surfaces the five-class response breakdown."""
    report = web.run_scan({"kind": ["mock"]})
    page = web.results_page(report).decode("utf-8")
    assert "5-class response breakdown" in page
    # every class label is present
    for label in ("off-task", "refused (safe)", "truncated", "partial", "full bypass"):
        assert label in page


def test_response_class_counts_full_equals_vulnerable_count():
    """The 'full' class count equals the report's boolean vulnerable count.

    classify_result freezes FULL <=> the engine's boolean success, so the 5-class
    breakdown's `full` count must match `report.failed` for the offline core.
    """
    report = web.run_scan({"kind": ["mock"]})
    counts = web._response_class_counts(report)
    assert counts[ResponseClass.FULL.value] == report.failed
    # the tally covers every result exactly once
    assert sum(counts.values()) == report.total


def test_response_class_counts_covers_all_five_keys():
    """Every response-class key is present in the tally (zeros included)."""
    report = web.run_scan({"kind": ["mock"], "technique": ["direct_injection"]})
    counts = web._response_class_counts(report)
    for cls in ResponseClass:
        assert cls.value in counts


def test_response_class_counts_degrades_to_empty_without_seam():
    """If the classifier import fails the tally degrades to an empty mapping."""
    report = web.run_scan({"kind": ["mock"], "technique": ["direct_injection"]})
    with mock.patch.dict(
        "sys.modules", {"injectkit.evaluators.response_class": None}
    ):
        counts = web._response_class_counts(report)
    assert counts == {}


def test_results_page_renders_without_breakdown_when_seam_missing():
    """The scan page still renders (no breakdown) if the 5-class seam is absent."""
    report = web.run_scan({"kind": ["mock"], "technique": ["direct_injection"]})
    with mock.patch.object(web, "_response_class_counts", return_value={}):
        page = web.results_page(report).decode("utf-8")
    assert b"Scan results" in page.encode("utf-8")
    assert "5-class response breakdown" not in page


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def test_dep_available_true_for_stdlib_false_for_missing():
    """_dep_available reports a real module present and a bogus one absent."""
    assert web._dep_available("json") is True
    assert web._dep_available("definitely_not_a_real_module_xyz") is False


def test_register_v3_transforms_is_idempotent():
    """Registering the v0.3.0 transforms twice does not raise."""
    web._register_v3_transforms()
    web._register_v3_transforms()  # second call must be a safe no-op
    names = web._list_transforms()
    assert "caesar" in names and "translate" in names


def test_handle_submit_scan_still_returns_report_html():
    """An end-to-end scan submit still returns a results page + standalone HTML."""
    page, report_html = web.handle_submit({"kind": ["mock"], "mode": ["scan"]})
    assert isinstance(page, bytes)
    assert report_html is not None
    # the 5-class breakdown rides along on the embedded results page
    assert b"5-class response breakdown" in page
