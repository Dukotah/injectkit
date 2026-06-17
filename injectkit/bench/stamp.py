"""The reproducibility STAMP — the 8 mandatory provenance fields (ROADMAP §8).

CHUNK 7-bench-harness. Every leaderboard number injectkit publishes must carry a
stamp that names *exactly* how it was produced, so a third party can replay the
run byte-for-byte and so two runs of the same cell are comparable. The stamp is
the audit record the ROADMAP makes load-bearing: a number with a drifting or
incomplete stamp is not publishable.

The 8 mandatory fields (ALL required — :class:`ReproStamp` refuses to build
without every one of them, and ``quant`` is mandatory, never defaulted):

1. ``version``        — the injectkit package version that produced the run.
2. ``corpus_hash``    — SHA-256 of the behavior set (the exact prompts graded).
3. ``model_revision`` — the pinned HF commit SHA (``repo@revision``) of the model.
4. ``seed``           — the RNG seed threaded through generation + sampling.
5. ``quant``          — ``fp16 | 8bit | 4bit`` (MANDATORY — the quant column).
6. ``judge_id``       — the EVAL judge that graded the responses.
7. ``attack_id``      — the attack family the cell ran.
8. ``backend``        — ``hf | vllm`` (the generation backend; §3.2 lock).

``corpus_hash`` is the SHA-256 of the *canonical* serialisation of the behaviors
(sorted, JSON, stable separators) so reordering the behavior list does not change
the hash but adding/removing/editing a behavior does — exactly the tamper-evidence
the gate wants.

Two stamps with the same 8 fields denote the same experiment; the harness asserts
that two seeded runs of the same cell carry identical *non-seed* stamp fields, so a
reproducibility check is a stamp-equality check plus an ASR-within-CI check.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import __version__

__all__ = [
    "VALID_QUANTS",
    "VALID_BACKENDS",
    "STAMP_FIELDS",
    "StampError",
    "ReproStamp",
    "corpus_hash",
    "build_stamp",
    "stamps_reproduce",
]

#: The quantisations a stamp may record (mirrors ``whitebox.zoo.VALID_DTYPES``).
#: ``quant`` is MANDATORY — there is no default; a missing quant is a build error.
VALID_QUANTS = frozenset({"fp16", "8bit", "4bit"})

#: The generation backends a stamp may record (mirrors ``generate.runner``).
VALID_BACKENDS = frozenset({"hf", "vllm"})

#: The ordered names of the 8 mandatory stamp fields (the audit contract).
STAMP_FIELDS = (
    "version",
    "corpus_hash",
    "model_revision",
    "seed",
    "quant",
    "judge_id",
    "attack_id",
    "backend",
)

#: Stamp fields that must be identical across two seeded runs of the SAME cell
#: (everything except the seed, which is what is varied between the two runs).
INVARIANT_FIELDS = tuple(f for f in STAMP_FIELDS if f != "seed")


class StampError(ValueError):
    """Raised when a stamp is built with a missing/invalid mandatory field."""


def corpus_hash(behaviors: Iterable[Any]) -> str:
    """SHA-256 of a behavior set's canonical serialisation (the corpus hash).

    The behaviors are normalised to comparable strings (a behavior may be a plain
    prompt string or a mapping carrying ``id``/``prompt``/... — both are supported)
    and serialised as a *sorted* JSON array with stable separators. Sorting makes
    the hash order-independent (reordering the list is the same experiment) while
    any edit/add/removal flips it (the tamper-evidence the gate wants).

    Args:
        behaviors: an iterable of behaviors — strings, or mappings with the
            fields that define a behavior.

    Returns:
        The lowercase hex SHA-256 digest of the canonical serialisation.
    """
    canon = sorted(_canonical_behavior(b) for b in behaviors)
    blob = json.dumps(canon, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical_behavior(behavior: Any) -> str:
    """Normalise one behavior to a stable string for hashing.

    A string behavior hashes as itself; a mapping hashes as its canonical JSON
    (sorted keys) so two equal behaviors with differently-ordered keys agree.
    Anything else falls back to ``str()`` so the hash is always total.
    """
    if isinstance(behavior, str):
        return behavior
    if isinstance(behavior, Mapping):
        return json.dumps(
            dict(behavior), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    # A behavior dataclass / object: prefer an ``id``+``prompt`` view if present.
    bid = getattr(behavior, "id", None)
    prompt = getattr(behavior, "prompt", None) or getattr(behavior, "payload", None)
    if bid is not None or prompt is not None:
        return json.dumps(
            {"id": bid, "prompt": prompt},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return str(behavior)


@dataclass(frozen=True)
class ReproStamp:
    """The 8-field reproducibility stamp (ROADMAP §8) — immutable + total.

    Construct via :meth:`build` (or :func:`build_stamp`) which validates every
    field; direct construction is allowed but the validators run in
    ``__post_init__`` regardless, so a bad stamp can never exist. ``extra`` carries
    non-mandatory provenance (avg-queries, wall-clock, ...) that the leaderboard
    surfaces but that is NOT part of the 8-field identity.
    """

    #: The injectkit version that produced the run.
    version: str
    #: SHA-256 of the behavior set (the exact prompts graded).
    corpus_hash: str
    #: Pinned model revision, ``repo@sha`` or a bare 40-hex SHA.
    model_revision: str
    #: RNG seed threaded through generation + candidate sampling.
    seed: int
    #: ``fp16 | 8bit | 4bit`` — MANDATORY (the quant column).
    quant: str
    #: The EVAL judge id that graded the responses.
    judge_id: str
    #: The attack family id the cell ran.
    attack_id: str
    #: ``hf | vllm`` — the generation backend (§3.2 lock).
    backend: str
    #: Non-identity provenance extras (avg_queries, wall_clock_s, gpu_hours, ...).
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [
            name
            for name in STAMP_FIELDS
            if _is_blank(getattr(self, name))
        ]
        if missing:
            raise StampError(
                "reproducibility stamp is missing mandatory field(s) "
                f"{missing}; all 8 of {list(STAMP_FIELDS)} are required "
                "(ROADMAP §8). quant is MANDATORY — pass an explicit fp16|8bit|4bit."
            )
        quant = str(self.quant).strip().lower()
        if quant not in VALID_QUANTS:
            raise StampError(
                f"stamp.quant must be one of {sorted(VALID_QUANTS)} (got "
                f"{self.quant!r}); the quant column is mandatory and not defaulted."
            )
        backend = str(self.backend).strip().lower()
        if backend not in VALID_BACKENDS:
            raise StampError(
                f"stamp.backend must be one of {sorted(VALID_BACKENDS)} (got "
                f"{self.backend!r})."
            )
        # Normalise the validated scalars in-place (frozen ⇒ object.__setattr__).
        object.__setattr__(self, "quant", quant)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def build(
        cls,
        *,
        corpus_hash: str,
        model_revision: str,
        seed: int,
        quant: str,
        judge_id: str,
        attack_id: str,
        backend: str,
        version: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> "ReproStamp":
        """Build + validate a stamp; ``version`` defaults to the package version.

        ``quant`` is keyword-only and has NO default so a caller cannot forget it —
        an omitted quant is a ``TypeError`` at the call site, the strongest possible
        "quant mandatory" guarantee.
        """
        return cls(
            version=version if version is not None else __version__,
            corpus_hash=corpus_hash,
            model_revision=model_revision,
            seed=seed,
            quant=quant,
            judge_id=judge_id,
            attack_id=attack_id,
            backend=backend,
            extra=dict(extra or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of all 8 fields plus ``extra`` (the audit record)."""
        d: dict[str, Any] = {name: getattr(self, name) for name in STAMP_FIELDS}
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    def identity(self) -> dict[str, Any]:
        """The 8 identity fields only (no ``extra``) — what equality compares."""
        return {name: getattr(self, name) for name in STAMP_FIELDS}

    def invariant_identity(self) -> dict[str, Any]:
        """The identity fields EXCEPT ``seed`` — the cross-seed reproduce key.

        Two seeded runs of the same cell must agree on these; the harness uses
        :func:`stamps_reproduce` to assert it.
        """
        return {name: getattr(self, name) for name in INVARIANT_FIELDS}


def _is_blank(value: Any) -> bool:
    """Whether a mandatory field is effectively unset (None or empty string).

    ``seed`` is an int and ``0`` is a legitimate seed, so only ``None`` and empty
    strings count as blank — an integer (even ``0``) is always present.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def build_stamp(
    *,
    behaviors: Iterable[Any],
    model_revision: str,
    seed: int,
    quant: str,
    judge_id: str,
    attack_id: str,
    backend: str,
    version: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> ReproStamp:
    """Build a :class:`ReproStamp` deriving ``corpus_hash`` from ``behaviors``.

    The convenience constructor the harness uses: it hashes the behavior set for
    you (so the corpus hash and the graded behaviors can never drift) and forwards
    the remaining mandatory fields. ``quant`` is keyword-only with no default.
    """
    return ReproStamp.build(
        corpus_hash=corpus_hash(behaviors),
        model_revision=model_revision,
        seed=seed,
        quant=quant,
        judge_id=judge_id,
        attack_id=attack_id,
        backend=backend,
        version=version,
        extra=extra,
    )


def stamps_reproduce(a: ReproStamp, b: ReproStamp) -> bool:
    """Whether two stamps describe the same cell up to the seed (reproduce key).

    True iff every identity field except ``seed`` matches — the precondition for a
    two-seed reproducibility check (the ASR must then agree within the CI).
    """
    return a.invariant_identity() == b.invariant_identity()
