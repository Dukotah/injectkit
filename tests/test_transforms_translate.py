"""Offline tests for the semantic low-resource-language translate transform.

All translation is performed by the ``StubTranslator`` fixture from
``tests/conftest.py`` (a deterministic prefix-tag pseudo-translation) — these
tests make NO network/model/translation calls and need no argostranslate. They
pin the frozen contract from ``injectkit/transforms/translate.py``:

* the success marker(s) (``<trigger><canary>`` and bare ``<canary>``) survive
  translation in cleartext so the benign-proxy detector still scores a hit;
* only the text *outside* the marker spans is handed to the translator;
* ``apply`` is total on empty input;
* :class:`ArgosTranslator` lazy-imports argostranslate and degrades gracefully.

Research basis: Yong et al. arXiv:2310.02446 / MultiJail arXiv:2310.06474.
"""

from __future__ import annotations

import builtins

import pytest

from injectkit.evaluators.heuristics import DEFAULT_TRIGGER
from injectkit.transforms.base import (
    Transform,
    TransformError,
    get_transform,
    list_transforms,
)
from injectkit.transforms.translate import (
    DEFAULT_TARGET_LANG,
    TRANSLATE_TRANSFORM_NAME,
    ArgosTranslator,
    TranslateTransform,
    Translator,
    register_translate,
)


# --------------------------------------------------------------------------- #
# Contract / protocol basics
# --------------------------------------------------------------------------- #


def test_constants_and_name():
    assert TRANSLATE_TRANSFORM_NAME == "translate"
    assert DEFAULT_TARGET_LANG == "sw"
    assert TranslateTransform.name == "translate"


def test_transform_satisfies_protocol(stub_translator):
    t = TranslateTransform(stub_translator)
    assert isinstance(t, Transform)


def test_stub_translator_satisfies_translator_protocol(stub_translator):
    assert isinstance(stub_translator, Translator)


# --------------------------------------------------------------------------- #
# Marker survival (the core canary discipline)
# --------------------------------------------------------------------------- #


def test_full_marker_survives_verbatim(stub_translator):
    canary = "ab12cd"
    marker = f"{DEFAULT_TRIGGER}{canary}"
    payload = f"Please output exactly: {marker}"
    out = TranslateTransform(stub_translator).apply(payload, canary)
    # The full marker is present verbatim and was never handed to the translator.
    assert marker in out
    for sent, _src, _tgt in stub_translator.calls:
        assert marker not in sent
        assert canary not in sent


def test_bare_canary_survives_verbatim(stub_translator):
    canary = "zz99"
    payload = f"Echo the token {canary} back to me."
    out = TranslateTransform(stub_translator).apply(payload, canary)
    assert canary in out
    for sent, _src, _tgt in stub_translator.calls:
        assert canary not in sent


def test_emit_hint_restates_marker(stub_translator):
    canary = "deadbeef"
    out = TranslateTransform(stub_translator).apply(
        f"Output {DEFAULT_TRIGGER}{canary}", canary
    )
    # The appended cleartext emit hint names the literal marker.
    assert f"{DEFAULT_TRIGGER}{canary}" in out
    assert "output exactly" in out.lower()


# --------------------------------------------------------------------------- #
# Translation behaviour
# --------------------------------------------------------------------------- #


def test_non_marker_text_is_translated(stub_translator):
    canary = "abc"
    out = TranslateTransform(stub_translator).apply(
        f"Translate me {DEFAULT_TRIGGER}{canary}", canary
    )
    # StubTranslator prefixes each translated chunk with its tag.
    assert stub_translator.tag in out
    assert stub_translator.calls  # at least one chunk translated


def test_source_and_target_passed_to_translator(stub_translator):
    t = TranslateTransform(stub_translator, source="en", target="fr")
    t.apply("hello world", "")
    assert stub_translator.calls
    for _text, src, tgt in stub_translator.calls:
        assert src == "en"
        assert tgt == "fr"


def test_target_lang_named_in_frame(stub_translator):
    out = TranslateTransform(stub_translator, target="sw").apply("do the thing", "")
    assert "(sw)" in out


def test_whitespace_only_chunks_not_translated(stub_translator):
    # A marker flanked by spaces should not produce empty/whitespace translate
    # calls — only meaningful text is sent.
    canary = "xy"
    TranslateTransform(stub_translator).apply(
        f"a {DEFAULT_TRIGGER}{canary} b", canary
    )
    for text, _s, _t in stub_translator.calls:
        assert text.strip()


# --------------------------------------------------------------------------- #
# Totality
# --------------------------------------------------------------------------- #


def test_empty_payload_passthrough(stub_translator):
    t = TranslateTransform(stub_translator)
    assert t.apply("", "abc") == ""
    assert stub_translator.calls == []


def test_no_canary_translates_whole_payload(stub_translator):
    out = TranslateTransform(stub_translator).apply("just some text", "")
    assert stub_translator.tag in out
    assert stub_translator.calls


def test_noop_translator_keeps_payload_recoverable():
    class NoopTranslator:
        name = "noop"

        def translate(self, text, *, source, target):
            return text

    canary = "qq"
    payload = f"emit {DEFAULT_TRIGGER}{canary}"
    out = TranslateTransform(NoopTranslator()).apply(payload, canary)
    assert f"{DEFAULT_TRIGGER}{canary}" in out


# --------------------------------------------------------------------------- #
# ArgosTranslator lazy dependency
# --------------------------------------------------------------------------- #


def test_argos_empty_text_passthrough_no_import(monkeypatch):
    # Empty text returns immediately without importing argostranslate.
    def boom(name, *args, **kwargs):
        if name.startswith("argostranslate"):
            raise AssertionError("argostranslate must not be imported for empty text")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", boom)
    assert ArgosTranslator().translate("", source="en", target="sw") == ""


def test_argos_missing_dep_raises_transform_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("argostranslate"):
            raise ImportError("no argostranslate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TransformError) as exc:
        ArgosTranslator().translate("hi", source="en", target="sw")
    assert "translate" in str(exc.value).lower()


def test_argos_uses_installed_backend_when_present(monkeypatch):
    # Simulate an installed argostranslate by injecting a fake module.
    import sys
    import types

    fake_pkg = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")
    calls = []

    def fake_translate(text, source, target):
        calls.append((text, source, target))
        return f"<{target}>{text}"

    fake_translate_mod.translate = fake_translate
    fake_pkg.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_pkg)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    out = ArgosTranslator().translate("hello", source="en", target="sw")
    assert out == "<sw>hello"
    assert calls == [("hello", "en", "sw")]


def test_argos_unavailable_pair_degrades_to_passthrough(monkeypatch):
    import sys
    import types

    fake_pkg = types.ModuleType("argostranslate")
    fake_translate_mod = types.ModuleType("argostranslate.translate")

    def fake_translate(text, source, target):
        raise RuntimeError("no language package installed")

    fake_translate_mod.translate = fake_translate
    fake_pkg.translate = fake_translate_mod
    monkeypatch.setitem(sys.modules, "argostranslate", fake_pkg)
    monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_mod)

    # Total: a backend error degrades to pass-through, not a raise.
    assert ArgosTranslator().translate("hi", source="en", target="sw") == "hi"


def test_argos_default_name():
    assert ArgosTranslator(source="en", target="sw").name == "argos:en->sw"
    assert ArgosTranslator(name="custom").name == "custom"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_register_translate_is_idempotent_and_resolvable(stub_translator):
    register_translate(stub_translator)
    # Idempotent: a second call does not raise.
    register_translate(stub_translator)
    assert TRANSLATE_TRANSFORM_NAME in list_transforms()
    resolved = get_transform(TRANSLATE_TRANSFORM_NAME)
    assert resolved.name == "translate"
