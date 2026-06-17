"""The v0.4 model zoo — pinned-revision loader for the white-box attack lane.

CHUNK 2-model-zoo (ROADMAP §6.0, §6.14, §8). This is the model seam the white-box
:class:`~injectkit.whitebox.base.Attack` ABC loads through. It resolves a friendly
zoo name (``"qwen2.5-7b"``) to a concrete, *pinned* HuggingFace checkpoint and
returns a ready-to-attack ``(model, tokenizer, arch_flag, supported_attacks)``
tuple plus a reproducibility stamp recording the exact revision + quantisation.

Why this lives in ``whitebox/`` (not ``injectkit/models/zoo.py``)
-----------------------------------------------------------------
The shipped package is **flat-layout** and already owns a top-level
``injectkit/models.py`` module (the core dataclasses, imported package-wide as
``from .models import ...``). Creating a ``models/`` package would shadow that
module and break every one of those imports. Per the ROADMAP's own
"if flat-layout, rewrite every path to match — extend, don't rebuild" rule (the
same reasoning that put the typed configs in ``whitebox/config.py`` rather than
``config/base.py``), the zoo lives beside the white-box interface it serves.

What it guarantees
------------------
* **Stable pins, no live resolution** (ROADMAP §8): every entry pins a *full
  40-hex commit SHA*, and the loader refuses anything that is not one (a branch
  name like ``main`` is mutable and would make a run unreproducible). The pinned
  SHA is passed straight to ``from_pretrained(..., revision=...)`` and recorded in
  the stamp, so two runs of the same name are byte-identical inputs.
* **arch gating** (ROADMAP §6.14): each model is tagged ``dense`` or ``moe``.
  Gradient-family attacks (GCG and relatives) are dense-only because MoE routing
  is non-differentiable; the zoo carries the per-model ``supported_attacks`` list
  and :func:`check_attack_supported` enforces it up front.
* **quant in the stamp**: ``fp16 | 8bit | 4bit``. 8/4-bit loads go through
  ``accelerate`` + ``bitsandbytes`` (a ``BitsAndBytesConfig`` quantisation config);
  the chosen quant is recorded in the stamp regardless of backend.

Offline-first: ``torch`` / ``transformers`` / ``accelerate`` are **lazy-imported**
only when a model is actually loaded, so importing this module (to read/validate
the registry, resolve metadata, or build a stamp) never needs the heavy ``[hf]``
extra. The 7–20B checkpoints additionally need a GPU + a multi-GB download to
instantiate; on a CPU/no-GPU host the loader logic is verified against a tiny
model (GPT-2 / Pythia-160M) and the big loads are DEFERRED-NO-GPU.

DEFENSIVE / AUTHORIZED USE ONLY. The zoo loads models you run or are explicitly
authorized to test, for the benign-canary robustness methodology only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch  # noqa: F401
    import transformers  # noqa: F401

__all__ = [
    "ZooError",
    "ARCH_DENSE",
    "ARCH_MOE",
    "VALID_ARCHS",
    "VALID_DTYPES",
    "ZooEntry",
    "ZOO_PATH",
    "load_zoo",
    "get_entry",
    "list_models",
    "check_attack_supported",
    "load_by_revision",
]

#: Path to the pinned-revision registry shipped beside this module.
ZOO_PATH = Path(__file__).with_name("zoo.yaml")

ARCH_DENSE = "dense"
ARCH_MOE = "moe"
#: Architectures the zoo understands (ROADMAP §6.14).
VALID_ARCHS = frozenset({ARCH_DENSE, ARCH_MOE})
#: Load precisions recorded in the stamp. 8/4-bit go via accelerate+bitsandbytes.
VALID_DTYPES = frozenset({"fp16", "8bit", "4bit"})

#: A full HuggingFace commit revision is a 40-char hex SHA-1. We refuse anything
#: else (a branch/tag/short-sha is mutable and would float — ROADMAP §8).
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ZooError(RuntimeError):
    """Raised on a malformed zoo registry, unknown name, or unloadable entry."""


@dataclass(frozen=True)
class ZooEntry:
    """One resolved, validated row of ``zoo.yaml`` (immutable).

    The metadata half of the loader contract: everything resolvable WITHOUT a
    download, so the registry can be parsed, validated, listed and stamped on any
    host (no GPU, no ``[hf]`` extra). :meth:`load` turns it into a live model.
    """

    #: Friendly zoo key (the name callers pass to :func:`load_by_revision`).
    name: str
    #: HuggingFace ``org/repo`` id the checkpoint is pulled from.
    repo: str
    #: Pinned, immutable 40-hex commit SHA (NEVER a branch — ROADMAP §8).
    revision: str
    #: ``"dense"`` | ``"moe"`` (ROADMAP §6.14 arch gate).
    arch: str
    #: v0.4 attack-registry keys this model may be driven with (dense-only for
    #: the GCG family). Sorted, deduplicated, immutable.
    supported_attacks: tuple[str, ...]
    #: Load precision used when the caller passes no ``dtype``.
    default_dtype: str = "fp16"
    #: Approximate parameter count in billions (informational / budget sizing).
    params_b: float = 0.0
    #: True if the HF repo is access-gated and/or non-permissively licensed; the
    #: zoo references such models for benchmarking only and never bundles them.
    gated: bool = False
    #: Free-form note (licence posture, etc.).
    notes: str = ""

    def supports(self, attack_name: str) -> bool:
        """Whether ``attack_name`` is listed in :attr:`supported_attacks`."""
        return attack_name in self.supported_attacks

    def stamp(self, *, dtype: Optional[str] = None) -> dict[str, Any]:
        """Build the reproducibility stamp fragment for this model + quant.

        Records the exact ``repo@revision``, the architecture, and the chosen
        quantisation so an :class:`~injectkit.whitebox.base.AttackResult.stamp`
        pins precisely which checkpoint at which precision produced a run
        (ROADMAP §8 "records revision + quant in the stamp").

        Args:
            dtype: The quant actually used (``fp16|8bit|4bit``). ``None`` uses the
                entry's :attr:`default_dtype`.

        Returns:
            A JSON-serialisable dict: ``model``, ``repo``, ``revision``, ``arch``,
            ``quant``, ``params_b``, ``supported_attacks``.
        """
        quant = _normalize_dtype(dtype if dtype is not None else self.default_dtype)
        return {
            "model": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "arch": self.arch,
            "quant": quant,
            "params_b": self.params_b,
            "supported_attacks": list(self.supported_attacks),
        }

    def load(
        self,
        dtype: str = "fp16",
        *,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = False,
        **from_pretrained_kwargs: Any,
    ) -> "tuple[Any, Any]":
        """Instantiate ``(model, tokenizer)`` at this pinned revision and dtype.

        fp16 loads in half precision; ``8bit`` / ``4bit`` build a
        ``transformers.BitsAndBytesConfig`` and load through accelerate +
        bitsandbytes. The pinned :attr:`revision` is always passed to
        ``from_pretrained`` so the bytes are immutable.

        Heavy deps are lazy-imported here (never at module import). The 7–20B zoo
        checkpoints additionally require a GPU + a multi-GB download to actually
        instantiate (DEFERRED-NO-GPU on a CPU host); the load *path* below is the
        real production code and is exercised against a tiny model in the tests.

        Args:
            dtype: ``"fp16" | "8bit" | "4bit"``.
            device_map: passed through to ``from_pretrained`` (accelerate device
                placement). ``"auto"`` shards onto available GPUs; pass ``None``
                or ``"cpu"`` for a CPU load.
            trust_remote_code: forwarded to ``from_pretrained`` (default False —
                a security-relevant default; only set True for repos you trust).
            **from_pretrained_kwargs: extra kwargs forwarded to
                ``AutoModelForCausalLM.from_pretrained``.

        Returns:
            ``(model, tokenizer)``.
        """
        quant = _normalize_dtype(dtype)
        torch, transformers = _import_hf()

        model_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": trust_remote_code,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        if quant == "fp16":
            # torch.float16 weights. Half precision is the white-box default.
            # transformers >=5 renamed `torch_dtype` -> `dtype`; pick whichever
            # the installed version accepts so neither warns nor breaks.
            model_kwargs[_dtype_kwarg(transformers)] = torch.float16
        else:
            # 8/4-bit: accelerate + bitsandbytes. Building BitsAndBytesConfig
            # requires the `bitsandbytes` backend at load time (GPU-only);
            # constructing the config itself is what we validate here.
            bnb = _make_bnb_config(transformers, torch, quant)
            model_kwargs["quantization_config"] = bnb

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.repo,
            revision=self.revision,
            trust_remote_code=trust_remote_code,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.repo,
            **model_kwargs,
            **from_pretrained_kwargs,
        )
        return model, tokenizer


def _normalize_dtype(dtype: str) -> str:
    """Validate + canonicalise a dtype/quant string to one of :data:`VALID_DTYPES`.

    Accepts a few friendly aliases (``"fp16"``/``"float16"``/``"half"``,
    ``"int8"``, ``"int4"``/``"nf4"``) and lower-cases. Raises :class:`ZooError`
    on anything outside the supported set so a typo never silently picks fp16.
    """
    if not isinstance(dtype, str):
        raise ZooError(f"dtype must be a string, got {type(dtype).__name__}.")
    key = dtype.strip().lower()
    aliases = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "f16": "fp16",
        "8bit": "8bit",
        "int8": "8bit",
        "8-bit": "8bit",
        "4bit": "4bit",
        "int4": "4bit",
        "nf4": "4bit",
        "4-bit": "4bit",
    }
    canon = aliases.get(key)
    if canon is None:
        raise ZooError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(VALID_DTYPES)} "
            "(fp16 | 8bit | 4bit)."
        )
    return canon


def _dtype_kwarg(transformers: Any) -> str:
    """Name of the ``from_pretrained`` precision kwarg for the installed version.

    ``transformers`` >= 5 renamed ``torch_dtype`` -> ``dtype`` (the old name is
    deprecated and warns). We pick by version so the loader is quiet and correct
    on both 4.x and 5.x.
    """
    ver = str(getattr(transformers, "__version__", "0") or "0")
    try:
        major = int(ver.split(".", 1)[0])
    except (ValueError, IndexError):
        major = 0
    return "dtype" if major >= 5 else "torch_dtype"


def _make_bnb_config(transformers: Any, torch: Any, quant: str) -> Any:
    """Build a ``BitsAndBytesConfig`` for an 8/4-bit accelerate load.

    Separated out so the quant-config construction is unit-testable without
    touching a real checkpoint. 4-bit uses NF4 + fp16 compute (the standard
    QLoRA-style config); 8-bit uses LLM.int8().
    """
    BitsAndBytesConfig = transformers.BitsAndBytesConfig
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    # 4bit
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _import_hf() -> "tuple[Any, Any]":
    """Lazy-import ``torch`` + ``transformers`` with a friendly error if missing.

    Mirrors :func:`injectkit.targets.hf._import_hf`: the heavy ``[hf]`` extra is
    only needed to *load* a model, never to read the registry.
    """
    try:
        import torch  # noqa: PLC0415 (intentional lazy import)
        import transformers  # noqa: PLC0415 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ZooError(
            "Loading a zoo model requires 'transformers' and 'torch' (and "
            "'accelerate'/'bitsandbytes' for 8/4-bit). Install them with "
            "`pip install 'injectkit[hf]' accelerate` (bitsandbytes is GPU-only "
            "and needed for 8/4-bit quant)."
        ) from exc
    return torch, transformers


def _coerce_entry(name: str, raw: Any) -> ZooEntry:
    """Validate one raw YAML row into a frozen :class:`ZooEntry` (or raise)."""
    if not isinstance(raw, dict):
        raise ZooError(f"zoo entry {name!r} must be a mapping, got {type(raw).__name__}.")

    repo = raw.get("repo")
    if not isinstance(repo, str) or "/" not in repo:
        raise ZooError(f"zoo entry {name!r}: 'repo' must be an 'org/name' string.")

    revision = raw.get("revision")
    if not isinstance(revision, str) or not _FULL_SHA_RE.match(revision):
        raise ZooError(
            f"zoo entry {name!r}: 'revision' must be a full 40-hex commit SHA "
            f"(got {revision!r}). A branch/tag/short-sha is mutable and would "
            "float; pin an immutable commit (ROADMAP §8)."
        )

    arch = raw.get("arch")
    if arch not in VALID_ARCHS:
        raise ZooError(
            f"zoo entry {name!r}: 'arch' must be one of {sorted(VALID_ARCHS)} "
            f"(got {arch!r})."
        )

    default_dtype = _normalize_dtype(raw.get("default_dtype", "fp16"))

    raw_attacks = raw.get("supported_attacks", [])
    if not isinstance(raw_attacks, (list, tuple)):
        raise ZooError(
            f"zoo entry {name!r}: 'supported_attacks' must be a list (got "
            f"{type(raw_attacks).__name__})."
        )
    if not all(isinstance(a, str) and a for a in raw_attacks):
        raise ZooError(
            f"zoo entry {name!r}: every 'supported_attacks' entry must be a "
            "non-empty string."
        )
    supported_attacks = tuple(sorted(set(raw_attacks)))

    params_b = raw.get("params_b", 0.0)
    try:
        params_b = float(params_b)
    except (TypeError, ValueError):
        raise ZooError(f"zoo entry {name!r}: 'params_b' must be a number.") from None

    return ZooEntry(
        name=name,
        repo=repo,
        revision=revision,
        arch=arch,
        supported_attacks=supported_attacks,
        default_dtype=default_dtype,
        params_b=params_b,
        gated=bool(raw.get("gated", False)),
        notes=str(raw.get("notes", "") or ""),
    )


@lru_cache(maxsize=8)
def _load_zoo_cached(path_str: str) -> dict[str, ZooEntry]:
    """Parse + validate the registry at ``path_str`` (cached per path)."""
    path = Path(path_str)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ZooError(f"cannot read zoo file {path}: {exc}") from exc

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ZooError(f"zoo file {path} is not valid YAML: {exc}") from exc

    if not isinstance(doc, dict):
        raise ZooError(f"zoo file {path}: top level must be a mapping.")

    version = doc.get("version")
    if version != 1:
        raise ZooError(
            f"zoo file {path}: unsupported schema version {version!r} (expected 1)."
        )

    models = doc.get("models")
    if not isinstance(models, dict) or not models:
        raise ZooError(f"zoo file {path}: 'models' must be a non-empty mapping.")

    entries: dict[str, ZooEntry] = {}
    for name, raw in models.items():
        if not isinstance(name, str) or not name:
            raise ZooError(f"zoo file {path}: model keys must be non-empty strings.")
        entries[name] = _coerce_entry(name, raw)
    return entries


def load_zoo(path: Optional[Path | str] = None) -> dict[str, ZooEntry]:
    """Parse and validate the zoo registry into ``{name: ZooEntry}``.

    Args:
        path: registry file to read. ``None`` uses the bundled :data:`ZOO_PATH`.

    Returns:
        A fresh dict (a copy of the cached parse) mapping name -> :class:`ZooEntry`.

    Raises:
        ZooError: on any malformed/unparseable registry.
    """
    p = Path(path) if path is not None else ZOO_PATH
    # Return a shallow copy so callers can mutate the dict without poisoning cache.
    return dict(_load_zoo_cached(str(p)))


def list_models(path: Optional[Path | str] = None) -> list[str]:
    """All zoo model names, sorted."""
    return sorted(load_zoo(path))


def get_entry(name: str, path: Optional[Path | str] = None) -> ZooEntry:
    """Resolve ``name`` to its :class:`ZooEntry` (metadata only, no download).

    Raises:
        ZooError: if ``name`` is not in the registry.
    """
    zoo = load_zoo(path)
    try:
        return zoo[name]
    except KeyError:
        raise ZooError(
            f"unknown zoo model {name!r}; known: {sorted(zoo)}."
        ) from None


def check_attack_supported(
    entry: ZooEntry,
    attack_name: str,
    *,
    supported_arch: Optional[set[str]] = None,
) -> None:
    """Verify ``attack_name`` may run against ``entry`` (ROADMAP §6.0/§8/§6.14).

    Two complementary checks:

    * the zoo's own ``supported_attacks`` list must contain ``attack_name`` — this
      is the per-model allow-list (e.g. the MoE entry lists only ``prefill``); and
    * if ``supported_arch`` is given (an attack's
      :attr:`~injectkit.whitebox.base.Attack.supported_arch`), the model's
      :attr:`~ZooEntry.arch` must be in it — so a GCG attack (dense-only) refuses a
      MoE checkpoint up front rather than producing silently-wrong gradients.

    Args:
        entry: the resolved zoo entry.
        attack_name: the v0.4 attack-registry key being requested.
        supported_arch: optional set of arch flags the attack supports.

    Raises:
        ZooError: if the model does not list the attack, or its arch is unsupported.
    """
    if not entry.supports(attack_name):
        raise ZooError(
            f"model {entry.name!r} (arch {entry.arch!r}) does not list attack "
            f"{attack_name!r}; supported_attacks = {list(entry.supported_attacks)}. "
            "Gradient-family attacks (GCG) are dense-only for v0.4 (ROADMAP §6.14)."
        )
    if supported_arch is not None and entry.arch not in supported_arch:
        raise ZooError(
            f"attack {attack_name!r} supports arch {sorted(supported_arch)} but "
            f"model {entry.name!r} is {entry.arch!r}. MoE routing is "
            "non-differentiable (ROADMAP §6.14)."
        )


@dataclass
class LoadedModel:
    """The full result of a :func:`load_by_revision` call (richer than the tuple).

    :func:`load_by_revision` returns the 4-tuple the chunk contract specifies, but
    also attaches this structured view (revision, quant, stamp) on the tuple's
    underlying objects is awkward — so callers wanting the stamp use
    :func:`load_by_revision`'s tuple plus :meth:`ZooEntry.stamp`, or this dataclass
    via :func:`load_model`.
    """

    model: Any
    tokenizer: Any
    arch_flag: str
    supported_attacks: tuple[str, ...]
    entry: ZooEntry = field(repr=False)
    quant: str = "fp16"
    stamp: dict[str, Any] = field(default_factory=dict)


def load_model(
    name: str,
    dtype: str = "fp16",
    *,
    path: Optional[Path | str] = None,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> LoadedModel:
    """Resolve + load ``name`` and return a structured :class:`LoadedModel`.

    The structured sibling of :func:`load_by_revision` (which returns the bare
    4-tuple the chunk contract specifies). Use this when you also want the stamp
    and the resolved :class:`ZooEntry` in one object.
    """
    entry = get_entry(name, path)
    quant = _normalize_dtype(dtype)
    model, tokenizer = entry.load(
        dtype=quant,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        arch_flag=entry.arch,
        supported_attacks=entry.supported_attacks,
        entry=entry,
        quant=quant,
        stamp=entry.stamp(dtype=quant),
    )


def load_by_revision(
    name: str,
    dtype: str = "fp16",
    *,
    path: Optional[Path | str] = None,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = False,
    **from_pretrained_kwargs: Any,
) -> "tuple[Any, Any, str, tuple[str, ...]]":
    """Load a zoo model at its PINNED revision and dtype (the chunk contract).

    The headline loader (ROADMAP §6.0 / CHUNK 2):

        model, tokenizer, arch_flag, supported_attacks = load_by_revision("qwen2.5-7b")

    Resolves ``name`` -> ``ZooEntry``, loads ``(model, tokenizer)`` at the entry's
    immutable pinned commit SHA (never a branch — ROADMAP §8) in the requested
    ``dtype`` (``fp16`` / ``8bit`` / ``4bit``; 8/4-bit go via accelerate +
    bitsandbytes), and returns the 4-tuple plus the arch flag and the attack
    allow-list. The exact ``revision`` + ``quant`` are available for the run stamp
    via :meth:`ZooEntry.stamp` (use :func:`load_model` for a one-call structured
    result that includes the stamp).

    The 7–20B checkpoints need a GPU + a multi-GB download to instantiate
    (DEFERRED-NO-GPU on a CPU host); the loader path itself is real production code
    and is verified against a tiny model in the test suite.

    Args:
        name: a zoo key (see :func:`list_models`).
        dtype: ``"fp16" | "8bit" | "4bit"`` (default fp16).
        path: optional alternative registry file (default: the bundled one).
        device_map: accelerate device placement, forwarded to ``from_pretrained``.
        trust_remote_code: forwarded to ``from_pretrained`` (default False).
        **from_pretrained_kwargs: extra kwargs forwarded to ``from_pretrained``.

    Returns:
        ``(model, tokenizer, arch_flag, supported_attacks)``.

    Raises:
        ZooError: unknown name, bad dtype, malformed registry, or load failure
            (missing deps).
    """
    loaded = load_model(
        name,
        dtype=dtype,
        path=path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    return (
        loaded.model,
        loaded.tokenizer,
        loaded.arch_flag,
        loaded.supported_attacks,
    )
