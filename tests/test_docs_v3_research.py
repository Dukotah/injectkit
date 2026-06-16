"""Offline doc-content guards for the v0.3.0 research docs.

These tests are fully offline (no network, no model, no dataset, no SDK). They
assert that the v0.3.0 documentation stays honest and complete:

1. ``docs/RESEARCH.md`` carries the cited 2023-2026 map AND the explicit
   frontier-robustness caveat (the "90%+ on flagship models" claim is overstated).
2. ``docs/TAXONOMY.md`` and ``docs/BENCHMARK.md`` document every new v0.3.0 family
   (ciphers / artprompt / translate, ``crescendo_reply``, the named attackers
   PAIR/TAP/AutoDAN/GPTFUZZER/GCG, and 5-class scoring), each next to a citation.
3. The keys the docs advertise actually exist in the code (no drift between the
   docs and the frozen registry/enum contracts).

This module documents only — it changes no Python logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
RESEARCH = DOCS / "RESEARCH.md"
TAXONOMY = DOCS / "TAXONOMY.md"
BENCHMARK = DOCS / "BENCHMARK.md"

#: v0.3.0 transform registry keys (ciphers + translate) the docs must advertise.
_CIPHER_KEYS = ("caesar", "atbash", "morse", "unicode_escape", "artprompt", "selfcipher")
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
#: Primary citations every doc that names a family should carry (arXiv ids).
_CITATIONS = {
    "CipherChat": "2308.06463",
    "ArtPrompt": "2402.11753",
    "low-resource translation": "2310.02446",
    "MultiJail": "2310.06474",
    "Crescendo": "2404.01833",
    "PAIR": "2310.08419",
    "TAP": "2312.02119",
    "AutoDAN": "2310.04451",
    "GPTFUZZER": "2309.10253",
    "AmpleGCG": "2404.07921",
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def research() -> str:
    assert RESEARCH.is_file(), f"missing {RESEARCH}"
    return RESEARCH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def taxonomy() -> str:
    assert TAXONOMY.is_file(), f"missing {TAXONOMY}"
    return TAXONOMY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def benchmark() -> str:
    assert BENCHMARK.is_file(), f"missing {BENCHMARK}"
    return BENCHMARK.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# RESEARCH.md — the cited map + the honest caveat
# --------------------------------------------------------------------------- #
def test_research_has_authorized_use_notice(research: str) -> None:
    head = research[:600].upper()
    assert "DEFENSIVE" in head
    assert "AUTHORIZED USE ONLY" in head


def test_research_states_benign_canary_methodology(research: str) -> None:
    lowered = research.lower()
    assert "benign canary" in lowered
    assert "bypass" in lowered


def test_research_carries_the_overstated_caveat(research: str) -> None:
    """The frontier-robustness honesty caveat is the whole point — it must stay."""
    lowered = research.lower()
    assert "overstated" in lowered
    assert "90%" in research
    assert "flagship" in lowered
    # The corrective: frontier stacks are much harder (low single-digit ASR).
    assert "single dig" in lowered


def test_research_cites_every_primary_source(research: str) -> None:
    for name, arxiv in _CITATIONS.items():
        assert arxiv in research, f"RESEARCH.md missing citation {name} ({arxiv})"


def test_research_maps_every_v3_addition_to_a_module(research: str) -> None:
    for module in (
        "transforms/ciphers.py",
        "transforms/translate.py",
        "evaluators/response_class.py",
        "attackers/registry.py",
        "attackers/whitebox_base.py",
    ):
        assert module in research, f"RESEARCH.md missing module mapping: {module}"


def test_research_documents_gcg_safety_posture(research: str) -> None:
    lowered = research.lower()
    assert "white-box" in lowered
    assert "huggingface" in lowered
    assert "benign" in lowered
    # No harmful suffix bundled; harmful artifacts are gated, not redistributed.
    assert "gated" in lowered


# --------------------------------------------------------------------------- #
# TAXONOMY.md — new families present, each with a citation nearby
# --------------------------------------------------------------------------- #
def test_taxonomy_lists_cipher_transforms_with_citations(taxonomy: str) -> None:
    for key in _CIPHER_KEYS:
        assert key in taxonomy, f"TAXONOMY.md missing cipher transform: {key}"
    assert _CITATIONS["CipherChat"] in taxonomy
    assert _CITATIONS["ArtPrompt"] in taxonomy


def test_taxonomy_lists_translate_transform_with_citation(taxonomy: str) -> None:
    assert _TRANSLATE_KEY in taxonomy
    lowered = taxonomy.lower()
    assert "semantic" in lowered
    assert "low-resource" in lowered
    assert _CITATIONS["low-resource translation"] in taxonomy


def test_taxonomy_lists_crescendo_reply_with_citation(taxonomy: str) -> None:
    assert "crescendo_reply" in taxonomy
    assert _CITATIONS["Crescendo"] in taxonomy


def test_taxonomy_lists_named_attackers_with_citations(taxonomy: str) -> None:
    for key in _ATTACKER_KEYS:
        assert key in taxonomy, f"TAXONOMY.md missing attacker: {key}"
    for name in ("PAIR", "TAP", "AutoDAN", "GPTFUZZER", "AmpleGCG"):
        assert _CITATIONS[name] in taxonomy, f"TAXONOMY.md missing {name} citation"
    # GCG must be flagged white-box / HF-only.
    lowered = taxonomy.lower()
    assert "white-box" in lowered
    assert "huggingface" in lowered


def test_taxonomy_documents_5class_scoring(taxonomy: str) -> None:
    for cls in _RESPONSE_CLASSES:
        assert cls in taxonomy, f"TAXONOMY.md missing response class: {cls}"
    # SoK Prompt Hacking is the cited source for 5-class scoring.
    assert "2410.13901" in taxonomy


# --------------------------------------------------------------------------- #
# BENCHMARK.md — new axes documented, boolean success stays frozen
# --------------------------------------------------------------------------- #
def test_benchmark_documents_new_transform_axis(benchmark: str) -> None:
    for key in (*_CIPHER_KEYS, _TRANSLATE_KEY):
        assert key in benchmark, f"BENCHMARK.md missing transform: {key}"


def test_benchmark_documents_attacker_axis(benchmark: str) -> None:
    for key in _ATTACKER_KEYS:
        assert key in benchmark, f"BENCHMARK.md missing attacker: {key}"


def test_benchmark_keeps_boolean_success_frozen(benchmark: str) -> None:
    """5-class scoring must not move the headline: success == full only."""
    lowered = benchmark.lower()
    assert "is_success" in benchmark
    assert "full" in lowered
    # The honest-baselines caveat travels into the benchmark doc too.
    assert "overstated" in lowered


# --------------------------------------------------------------------------- #
# Anti-drift: every key the docs advertise really exists in the code.
# --------------------------------------------------------------------------- #
def test_docs_match_code_contracts() -> None:
    """Guard against doc/code drift on the frozen v0.3.0 contracts."""
    from injectkit.transforms.ciphers import CIPHER_TRANSFORM_NAMES
    from injectkit.transforms.translate import TRANSLATE_TRANSFORM_NAME
    from injectkit.attackers.registry import NAMED_ATTACKERS
    from injectkit.evaluators.response_class import ResponseClass

    assert set(CIPHER_TRANSFORM_NAMES) == set(_CIPHER_KEYS)
    assert TRANSLATE_TRANSFORM_NAME == _TRANSLATE_KEY
    assert {spec.name for spec in NAMED_ATTACKERS} == set(_ATTACKER_KEYS)
    assert {c.value for c in ResponseClass} == set(_RESPONSE_CLASSES)
