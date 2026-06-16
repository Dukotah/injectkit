"""Tests for the v0.3.0 scaffold: cipher/art/translate transforms, the five-class
response framework, the named-attacker registry, the white-box GCG base, and the
crescendo reply-referencing strategy.

All offline and deterministic — no network, no torch/transformers, no
argostranslate, no real models. These tests pin the FROZEN CONTRACTS the v0.3.0
builders implement against; the not-yet-implemented bodies raise
``NotImplementedError`` and that is asserted here so the contract stays stable.
"""

from __future__ import annotations

import pytest

from injectkit.attackers.base import AdaptiveAttacker, AttackerError
from injectkit.attackers.registry import (
    NAMED_ATTACKERS,
    AttackerRegistry,
    AttackerSpec,
    get_attacker,
    list_attackers,
    register_attacker,
)
from injectkit.attackers.whitebox_base import (
    GCGConfig,
    GCGStep,
    WhiteBoxGCGAttacker,
    WhiteBoxModel,
)
from injectkit.attacks.multiturn import (
    MULTI_TURN_STRATEGIES,
    CrescendoReplyReferencingStrategy,
    CrescendoStrategy,
)
from injectkit.evaluators.response_class import (
    LENGTH_STOP_REASONS,
    ResponseClass,
    classify_response,
)
from injectkit.transforms.base import TransformError
from injectkit.transforms.ciphers import (
    CIPHER_TRANSFORM_NAMES,
    ArtPromptTransform,
    AtbashCipher,
    CaesarCipher,
    MorseCipher,
    SelfCipherTransform,
    UnicodeEscapeCipher,
    register_builtin_ciphers,
)
from injectkit.transforms.translate import (
    DEFAULT_TARGET_LANG,
    TRANSLATE_TRANSFORM_NAME,
    ArgosTranslator,
    TranslateTransform,
    Translator,
)


# --------------------------------------------------------------------------- #
# Cipher / art / self-cipher transform contracts
# --------------------------------------------------------------------------- #


def test_cipher_transform_names_and_stable_keys():
    assert CIPHER_TRANSFORM_NAMES == (
        "caesar",
        "atbash",
        "morse",
        "unicode_escape",
        "artprompt",
        "selfcipher",
    )
    assert CaesarCipher.name == "caesar"
    assert AtbashCipher.name == "atbash"
    assert MorseCipher.name == "morse"
    assert UnicodeEscapeCipher.name == "unicode_escape"
    assert ArtPromptTransform.name == "artprompt"
    assert SelfCipherTransform.name == "selfcipher"


def test_cipher_transforms_construct_without_heavy_deps():
    # Construction must import nothing heavy and accept the documented args.
    assert CaesarCipher(shift=5)._shift == 5
    assert AtbashCipher().name == "atbash"
    assert UnicodeEscapeCipher(style="ncr")._style == "ncr"
    assert ArtPromptTransform(mask_word="output")._mask_word == "output"


def test_cipher_apply_implemented_and_marker_recoverable():
    # The scaffold's frozen-TODO assertion has been implemented in v0.3.0: every
    # cipher/art/self-cipher transform now keeps the success marker recoverable
    # (in cleartext) so the benign-canary detection still fires. ArtPrompt masks
    # the benign word "output", so use a payload that contains it.
    payload = "emit output INJECTOK-abc"
    for t in (
        CaesarCipher(),
        AtbashCipher(),
        MorseCipher(),
        UnicodeEscapeCipher(),
        ArtPromptTransform(mask_word="output"),
        SelfCipherTransform(),
    ):
        out = t.apply(payload, "abc")
        assert isinstance(out, str) and out
        assert "INJECTOK-abc" in out


def test_register_builtin_ciphers_is_idempotent():
    from injectkit.transforms.base import registry as _registry

    register_builtin_ciphers()
    names = _registry.names()
    for key in CIPHER_TRANSFORM_NAMES:
        assert key in names
    # Idempotent: a second call does not raise on duplicates.
    register_builtin_ciphers()


# --------------------------------------------------------------------------- #
# Semantic translate transform contract
# --------------------------------------------------------------------------- #


def test_translate_keys_and_protocol():
    assert TRANSLATE_TRANSFORM_NAME == "translate"
    assert DEFAULT_TARGET_LANG == "sw"
    assert TranslateTransform.name == "translate"


def test_stub_translator_satisfies_protocol(stub_translator):
    assert isinstance(stub_translator, Translator)
    out = stub_translator.translate("hello", source="en", target="sw")
    assert out == "[xx] hello"
    assert stub_translator.calls == [("hello", "en", "sw")]


def test_translate_transform_accepts_injected_translator(stub_translator):
    t = TranslateTransform(stub_translator, source="en", target="sw")
    assert t.translator is stub_translator
    # Implemented: the marker survives and the body was sent to the translator.
    out = t.apply("emit INJECTOK-abc", "abc")
    assert "INJECTOK-abc" in out
    assert stub_translator.calls  # the non-marker text was translated


def test_argos_translator_lazy_dep_friendly_error(monkeypatch):
    # With argostranslate absent, translate() raises a friendly TransformError
    # (not ImportError) BEFORE any NotImplementedError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("argostranslate"):
            raise ImportError("no argostranslate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TransformError):
        ArgosTranslator().translate("hi", source="en", target="sw")


# --------------------------------------------------------------------------- #
# Five-class response framework
# --------------------------------------------------------------------------- #


def test_response_class_values_and_success_bridge():
    assert ResponseClass.FULL.value == "full"
    assert ResponseClass.PARTIAL.value == "partial"
    assert ResponseClass.REJECT_SAFETY.value == "reject_safety"
    assert ResponseClass.REJECT_IRRELEVANT.value == "reject_irrelevant"
    assert ResponseClass.TOO_LONG.value == "too_long"
    # FROZEN INVARIANT: only FULL is a success, so the boolean stays derivable.
    assert ResponseClass.FULL.is_success is True
    for c in (
        ResponseClass.PARTIAL,
        ResponseClass.REJECT_SAFETY,
        ResponseClass.REJECT_IRRELEVANT,
        ResponseClass.TOO_LONG,
    ):
        assert c.is_success is False
    assert ResponseClass.REJECT_SAFETY.is_reject is True
    assert ResponseClass.REJECT_IRRELEVANT.is_reject is True
    assert ResponseClass.FULL.is_reject is False


def test_length_stop_reasons_cover_adapter_conventions():
    assert "max_tokens" in LENGTH_STOP_REASONS  # Anthropic
    assert "length" in LENGTH_STOP_REASONS  # OpenAI
    assert "max_new_tokens" in LENGTH_STOP_REASONS  # HF


def test_classify_response_marker_is_full(sample_attack):
    # The scaffold's frozen-TODO assertion has been implemented in v0.3.0: the
    # success-marker case now classifies as FULL (the only success class).
    from injectkit.models import TargetResponse

    cls = classify_response(
        sample_attack,
        TargetResponse(text="INJECTOK-abc"),
        "abc",
        [],
    )
    assert cls is ResponseClass.FULL
    assert cls.is_success is True


# --------------------------------------------------------------------------- #
# Named-attacker registry
# --------------------------------------------------------------------------- #


def test_named_attackers_declared_with_kinds_and_docs():
    names = {s.name for s in NAMED_ATTACKERS}
    assert names == {"pair", "tap", "autodan", "gptfuzzer", "gcg"}
    kinds = {s.name: s.kind for s in NAMED_ATTACKERS}
    assert kinds["pair"] == "black_box"
    assert kinds["gcg"] == "white_box"
    for s in NAMED_ATTACKERS:
        assert s.doc  # every spec cites a one-line doc/source


def test_default_registry_lists_all_five_but_none_available():
    listed = list_attackers()
    for key in ("pair", "tap", "autodan", "gptfuzzer", "gcg"):
        assert key in listed
    # Declared-but-not-yet-implemented: resolving raises a friendly error.
    with pytest.raises(AttackerError):
        get_attacker("pair")


def test_registry_register_marks_available_and_builds():
    reg = AttackerRegistry()
    reg.declare(AttackerSpec(name="demo", kind="black_box", doc="demo doc"))
    assert reg.available_names() == []

    class DemoAttacker:
        name = "demo"
        max_rounds = 1

        def run(self, seed_attack, target, detectors):  # pragma: no cover - trivial
            raise NotImplementedError

    reg.register("demo", lambda **opts: DemoAttacker())
    assert reg.available_names() == ["demo"]
    built = reg.get("demo")
    assert isinstance(built, AdaptiveAttacker)
    assert reg.spec("demo").available is True


def test_register_unknown_attacker_raises():
    reg = AttackerRegistry()
    with pytest.raises(KeyError):
        reg.register("nope", lambda **opts: None)
    with pytest.raises(KeyError):
        reg.get("nope")


def test_register_attacker_on_default_registry_then_resolve():
    from injectkit.attackers.registry import registry as _default_registry

    class PairLike:
        name = "pair"
        max_rounds = 3

        def run(self, seed_attack, target, detectors):  # pragma: no cover
            raise NotImplementedError

    # Snapshot the shared default-registry spec so this test does not leak a stub
    # factory into other modules' tests (e.g. tests/test_pair_attacker.py, which
    # resolves the *real* PAIRAttacker off the same process-wide registry).
    _saved = _default_registry.spec("pair")
    try:
        register_attacker("pair", lambda **opts: PairLike())
        built = get_attacker("pair")
        assert isinstance(built, AdaptiveAttacker)
        assert built.name == "pair"
    finally:
        # Restore the original spec (placeholder or builder-wired factory).
        _default_registry.declare(_saved)


# --------------------------------------------------------------------------- #
# White-box GCG base (HF-only, lazy torch, benign target, stub-testable)
# --------------------------------------------------------------------------- #


def test_stub_whitebox_model_satisfies_protocol(stub_whitebox_model):
    assert isinstance(stub_whitebox_model, WhiteBoxModel)
    ids = stub_whitebox_model.token_ids("hello")
    assert isinstance(ids, list) and ids
    loss = stub_whitebox_model.target_loss(ids, [1, 2])
    assert isinstance(loss, float)
    grads = stub_whitebox_model.token_gradients(ids, [1, 2], slice(0, 3))
    assert isinstance(grads, list) and isinstance(grads[0], list)


def test_gcg_config_defaults_are_benign_and_small():
    cfg = GCGConfig()
    # Default target is the benign marker (None until rendered with the canary).
    assert cfg.target_string is None
    assert cfg.max_steps >= 1
    assert cfg.trigger == "INJECTOK-"


def test_gcg_attacker_satisfies_protocol_and_budget(stub_whitebox_model):
    attacker = WhiteBoxGCGAttacker(stub_whitebox_model, GCGConfig(max_steps=1))
    assert isinstance(attacker, AdaptiveAttacker)
    assert attacker.name == "gcg"
    # max_rounds mirrors the step budget (protocol budget is meaningful).
    assert attacker.max_rounds == 1


def test_gcg_attacker_rejects_zero_steps(stub_whitebox_model):
    with pytest.raises(AttackerError):
        WhiteBoxGCGAttacker(stub_whitebox_model, GCGConfig(max_steps=0))


def test_gcg_run_and_optimize_are_frozen_todo(stub_whitebox_model, sample_attack):
    attacker = WhiteBoxGCGAttacker(stub_whitebox_model, GCGConfig(max_steps=1))
    with pytest.raises(NotImplementedError):
        attacker.run(sample_attack, stub_whitebox_model, [])
    with pytest.raises(NotImplementedError):
        attacker._optimize_suffix([1, 2, 3], [4, 5])


def test_import_torch_transformers_friendly_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from injectkit.attackers.whitebox_base import import_torch_transformers

    with pytest.raises(AttackerError):
        import_torch_transformers()


def test_gcg_step_record_shape():
    step = GCGStep(step=0, suffix="! ! !", loss=1.5, succeeded=False)
    assert step.step == 0
    assert step.suffix == "! ! !"
    assert step.loss == 1.5
    assert step.succeeded is False


# --------------------------------------------------------------------------- #
# Crescendo reply-referencing strategy
# --------------------------------------------------------------------------- #


def test_crescendo_reply_strategy_registered_and_extends_base():
    assert "crescendo_reply" in MULTI_TURN_STRATEGIES
    assert issubclass(CrescendoReplyReferencingStrategy, CrescendoStrategy)
    strat = CrescendoReplyReferencingStrategy(steps=2)
    assert strat.name == "crescendo_reply"
    assert strat.steps == 2


def test_crescendo_reply_static_build_still_works(sample_attack):
    # Inherited static build() (the base contract) must still produce steps so a
    # reply-referencing strategy degrades gracefully where the engine has no
    # reply hook yet.
    strat = CrescendoReplyReferencingStrategy(steps=2)
    steps = strat.build(sample_attack, "cana42")
    assert steps[-1].scored is True
    assert "INJECTOK-cana42" in steps[-1].message.content


def test_crescendo_reply_hooks_are_implemented(sample_attack):
    # v0.3.0: the reply-aware hooks are now implemented (no longer a TODO).
    # next_turn quotes the model's own prior reply as a benign, canary-free
    # lead-in; final_step is the single scored ask carrying the live marker.
    strat = CrescendoReplyReferencingStrategy(steps=1)
    lead = strat.next_turn(
        sample_attack, "cana42", [("assistant", "I said something")]
    )
    assert lead.scored is False
    assert lead.expect_response is True
    assert "I said something" in lead.message.content
    assert "cana42" not in lead.message.content

    final = strat.final_step(sample_attack, "cana42", [])
    assert final.scored is True
    assert "INJECTOK-cana42" in final.message.content
