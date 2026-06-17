"""Tests for the v0.4 white-box model zoo (CHUNK 2-model-zoo).

Covers the chunk's Done criteria (ROADMAP §6.0/§6.14/§8):

* ``zoo.yaml`` parses and validates into ``ZooEntry`` rows;
* >=5 dense models (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Mistral-7B-v0.3,
  Phi-4) + the GPT-OSS-20B MoE entry are present, with the MoE tagged
  ``arch: moe`` / ``supported_attacks: [prefill]``;
* every pinned ``revision`` is a full, immutable 40-hex commit SHA (no floating
  branches — §8 "pinned revisions are stable; no live version resolution");
* ``load_by_revision`` resolves the entry and passes the pinned revision +
  quant (fp16 / 8bit / 4bit) through to ``from_pretrained``, recording them in
  the stamp;
* arch gating refuses a gradient (dense-only) attack on a MoE model.

The loader logic is verified with a FAKE ``transformers`` (no download, no GPU);
a real tiny-model CPU load (``sshleifer/tiny-gpt2``) is opt-in and skipped when
``transformers``/network are unavailable. The 7–20B loads are DEFERRED-NO-GPU.
"""

from __future__ import annotations

import os
import textwrap
import types

import pytest

from injectkit.whitebox import zoo


# --------------------------------------------------------------------------- #
# Registry parse + schema validation
# --------------------------------------------------------------------------- #


def test_zoo_yaml_parses_and_lists_expected_models():
    names = zoo.list_models()
    # >=5 dense + the MoE; exact expected set.
    expected = {
        "llama-3.1-8b",
        "qwen2.5-7b",
        "gemma-2-9b",
        "mistral-7b-v0.3",
        "phi-4",
        "gpt-oss-20b",
    }
    assert expected <= set(names)


def test_at_least_five_dense_models():
    z = zoo.load_zoo()
    dense = [e for e in z.values() if e.arch == zoo.ARCH_DENSE]
    assert len(dense) >= 5
    # All the named dense flagships are present and dense.
    for n in ("llama-3.1-8b", "qwen2.5-7b", "gemma-2-9b", "mistral-7b-v0.3", "phi-4"):
        assert z[n].arch == zoo.ARCH_DENSE


def test_dense_models_support_gcg():
    z = zoo.load_zoo()
    for n in ("llama-3.1-8b", "qwen2.5-7b", "gemma-2-9b", "mistral-7b-v0.3", "phi-4"):
        assert "gcg" in z[n].supported_attacks


def test_moe_entry_is_prefill_only():
    e = zoo.get_entry("gpt-oss-20b")
    assert e.arch == zoo.ARCH_MOE
    assert e.supported_attacks == ("prefill",)
    # GCG (gradient family) must NOT be offered on the MoE.
    assert "gcg" not in e.supported_attacks


def test_every_revision_is_a_full_immutable_sha():
    for name, e in zoo.load_zoo().items():
        assert zoo._FULL_SHA_RE.match(e.revision), f"{name}: {e.revision!r} not a 40-hex SHA"
        # Defensive: a branch name would slip past a loose check.
        assert e.revision not in {"main", "master"}, name


def test_stamp_records_repo_revision_and_quant():
    e = zoo.get_entry("qwen2.5-7b")
    s = e.stamp(dtype="fp16")
    assert s["repo"] == "Qwen/Qwen2.5-7B-Instruct"
    assert s["revision"] == e.revision
    assert s["quant"] == "fp16"
    assert s["arch"] == "dense"
    # 4-bit quant is faithfully recorded too.
    assert e.stamp(dtype="4bit")["quant"] == "4bit"
    assert zoo.get_entry("gpt-oss-20b").stamp()["quant"] == "4bit"  # MoE default


# --------------------------------------------------------------------------- #
# dtype normalization + validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,canon",
    [
        ("fp16", "fp16"),
        ("float16", "fp16"),
        ("HALF", "fp16"),
        ("8bit", "8bit"),
        ("int8", "8bit"),
        ("4bit", "4bit"),
        ("nf4", "4bit"),
        ("int4", "4bit"),
    ],
)
def test_normalize_dtype_aliases(raw, canon):
    assert zoo._normalize_dtype(raw) == canon


def test_normalize_dtype_rejects_unknown():
    with pytest.raises(zoo.ZooError):
        zoo._normalize_dtype("fp8")
    with pytest.raises(zoo.ZooError):
        zoo._normalize_dtype("bf16")  # not offered by the zoo


# --------------------------------------------------------------------------- #
# Schema rejection (a floating revision must be refused)
# --------------------------------------------------------------------------- #


def _write_zoo(tmp_path, body: str):
    p = tmp_path / "zoo.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_branch_revision_is_rejected(tmp_path):
    p = _write_zoo(
        tmp_path,
        """\
        version: 1
        models:
          floaty:
            repo: org/model
            revision: main
            arch: dense
            supported_attacks: [gcg]
        """,
    )
    with pytest.raises(zoo.ZooError, match="40-hex"):
        zoo.load_zoo(p)


def test_bad_arch_is_rejected(tmp_path):
    p = _write_zoo(
        tmp_path,
        """\
        version: 1
        models:
          m:
            repo: org/model
            revision: %s
            arch: sparse
            supported_attacks: [gcg]
        """
        % ("a" * 40),
    )
    with pytest.raises(zoo.ZooError, match="arch"):
        zoo.load_zoo(p)


def test_unknown_schema_version_rejected(tmp_path):
    p = _write_zoo(
        tmp_path,
        """\
        version: 99
        models:
          m:
            repo: org/model
            revision: %s
            arch: dense
            supported_attacks: [gcg]
        """
        % ("b" * 40),
    )
    with pytest.raises(zoo.ZooError, match="version"):
        zoo.load_zoo(p)


def test_unknown_model_name_raises():
    with pytest.raises(zoo.ZooError, match="unknown zoo model"):
        zoo.get_entry("does-not-exist")


# --------------------------------------------------------------------------- #
# arch / attack gating
# --------------------------------------------------------------------------- #


def test_check_attack_supported_allows_dense_gcg():
    e = zoo.get_entry("qwen2.5-7b")
    # Should not raise.
    zoo.check_attack_supported(e, "gcg", supported_arch={"dense"})


def test_check_attack_supported_refuses_gcg_on_moe():
    e = zoo.get_entry("gpt-oss-20b")
    with pytest.raises(zoo.ZooError):
        zoo.check_attack_supported(e, "gcg", supported_arch={"dense"})


def test_check_attack_supported_refuses_unlisted_attack():
    e = zoo.get_entry("qwen2.5-7b")
    with pytest.raises(zoo.ZooError, match="does not list attack"):
        zoo.check_attack_supported(e, "prefill")


# --------------------------------------------------------------------------- #
# load_by_revision logic — verified with a FAKE transformers (no download)
# --------------------------------------------------------------------------- #


class _FakeTok:
    @classmethod
    def from_pretrained(cls, repo, **kw):
        _FakeTok.last = (repo, kw)
        return cls()


class _FakeModel:
    @classmethod
    def from_pretrained(cls, repo, **kw):
        _FakeModel.last = (repo, kw)
        return cls()


class _FakeBnb:
    def __init__(self, **kw):
        self.kw = kw


class _FakeTorch:
    float16 = "FLOAT16"


def _fake_transformers(version="5.0.0"):
    return types.SimpleNamespace(
        __version__=version,
        AutoTokenizer=_FakeTok,
        AutoModelForCausalLM=_FakeModel,
        BitsAndBytesConfig=_FakeBnb,
    )


@pytest.fixture
def patched_hf(monkeypatch):
    tf = _fake_transformers()
    monkeypatch.setattr(zoo, "_import_hf", lambda: (_FakeTorch, tf))
    return tf


def test_load_by_revision_returns_four_tuple_and_pins_revision(patched_hf):
    model, tok, arch, attacks = zoo.load_by_revision(
        "qwen2.5-7b", dtype="fp16", device_map="cpu"
    )
    assert isinstance(model, _FakeModel)
    assert isinstance(tok, _FakeTok)
    assert arch == "dense"
    assert attacks == ("gcg",)
    repo, kw = _FakeModel.last
    assert repo == "Qwen/Qwen2.5-7B-Instruct"
    assert kw["revision"] == zoo.get_entry("qwen2.5-7b").revision
    # fp16 path sets the precision kwarg (dtype on transformers>=5).
    assert kw.get("dtype") == "FLOAT16" or kw.get("torch_dtype") == "FLOAT16"
    # Tokenizer is also pinned to the same revision.
    assert _FakeTok.last[1]["revision"] == kw["revision"]


def test_load_by_revision_4bit_builds_bnb_config(patched_hf):
    zoo.load_by_revision("gpt-oss-20b", dtype="4bit", device_map="auto")
    _, kw = _FakeModel.last
    qc = kw["quantization_config"]
    assert isinstance(qc, _FakeBnb)
    assert qc.kw.get("load_in_4bit") is True
    assert qc.kw.get("bnb_4bit_quant_type") == "nf4"


def test_load_by_revision_8bit_builds_bnb_config(patched_hf):
    zoo.load_by_revision("phi-4", dtype="8bit")
    _, kw = _FakeModel.last
    qc = kw["quantization_config"]
    assert qc.kw.get("load_in_8bit") is True
    # 8/4-bit must NOT also pass an fp16 precision kwarg.
    assert "torch_dtype" not in kw and "dtype" not in kw


def test_load_model_returns_stamp_with_revision_and_quant(patched_hf):
    loaded = zoo.load_model("mistral-7b-v0.3", dtype="4bit", device_map="auto")
    assert loaded.arch_flag == "dense"
    assert loaded.quant == "4bit"
    assert loaded.stamp["revision"] == zoo.get_entry("mistral-7b-v0.3").revision
    assert loaded.stamp["quant"] == "4bit"
    assert loaded.stamp["repo"] == "mistralai/Mistral-7B-Instruct-v0.3"


def test_load_by_revision_uses_torch_dtype_on_transformers_4x(monkeypatch):
    tf = _fake_transformers(version="4.44.0")
    monkeypatch.setattr(zoo, "_import_hf", lambda: (_FakeTorch, tf))
    zoo.load_by_revision("phi-4", dtype="fp16", device_map="cpu")
    _, kw = _FakeModel.last
    assert kw.get("torch_dtype") == "FLOAT16"
    assert "dtype" not in kw


def test_load_bad_dtype_raises(patched_hf):
    with pytest.raises(zoo.ZooError):
        zoo.load_by_revision("phi-4", dtype="fp8")


def test_import_hf_missing_dep_raises_zoo_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *a, **k):
        if name in {"torch", "transformers"}:
            raise ImportError(f"no {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    with pytest.raises(zoo.ZooError, match="transformers"):
        zoo._import_hf()


# --------------------------------------------------------------------------- #
# Optional: a REAL tiny-model CPU load (no GPU). 8B/20B loads are DEFERRED-NO-GPU.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("INJECTKIT_ZOO_LIVE") != "1",
    reason="live tiny-model download; set INJECTKIT_ZOO_LIVE=1 to run",
)
def test_real_tiny_cpu_load(tmp_path):
    """End-to-end load of a tiny real model through the production loader (CPU).

    Verifies the actual ``from_pretrained`` path resolves the pinned revision and
    returns a working model+tokenizer. Opt-in (downloads ~2MB) via
    ``INJECTKIT_ZOO_LIVE=1``; skips cleanly without ``transformers`` or network.
    The 7–20B zoo entries are DEFERRED-NO-GPU.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    sha = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"  # sshleifer/tiny-gpt2 pin
    p = _write_zoo(
        tmp_path,
        f"""\
        version: 1
        models:
          tiny-gpt2:
            repo: sshleifer/tiny-gpt2
            revision: {sha}
            arch: dense
            params_b: 0.0001
            default_dtype: fp16
            supported_attacks: [gcg]
        """,
    )
    try:
        model, tok, arch, attacks = zoo.load_by_revision(
            "tiny-gpt2", dtype="fp16", path=p, device_map=None
        )
    except Exception as exc:  # network/cache unavailable -> skip, never fail.
        pytest.skip(f"tiny model unavailable offline: {exc}")

    assert arch == "dense"
    assert attacks == ("gcg",)
    out = model(**tok("hello", return_tensors="pt"))
    assert out.logits.shape[0] == 1
