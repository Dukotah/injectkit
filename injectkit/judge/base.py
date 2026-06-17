"""The v0.4 offline judge layer — the :class:`Judge` ABC + the judge registry.

ROADMAP §6.10 / §11 chunk 4. The judge layer scores a target's *response* into
three signals that are **reported separately and never collapsed into one number**
(ROADMAP §6.10, §3 "substring-ASR + judge-ASR + SR-mean reported SEPARATELY"):

* ``label_5class`` — the StrongREJECT/SoK 5-class grade
  (:class:`~injectkit.evaluators.response_class.ResponseClass`).
* ``success_bool`` — the binarised judge-ASR success label.
* ``sr_score`` — the continuous StrongREJECT 0..1 score (the primary leaderboard
  column; binarisation at θ is a *reporting* choice, not the metric).

Every concrete judge implements one method, :meth:`Judge.judge`, and declares
three class attributes the licensing/gating machinery enforces:

* :attr:`judge_id` — the registry key, recorded in the reproducibility stamp so a
  leaderboard number always names the judge that produced it (and so the
  circularity firewall can assert ``opt_judge_id != eval_judge_id``, §6.10.1).
* :attr:`license` — the SPDX-ish licence string of the judge's *weights*. The
  bundleable default is MIT; the loader-gated judges carry their non-permissive
  upstream licence (Llama-2 / Llama-3.1 Community) so :mod:`docs/JUDGES.md` and a
  test can assert "only the MIT judge is bundleable" (§6.10, §7 Risk 6).
* :attr:`is_bundled` — True only for judges whose weights/logic injectkit ships
  in the wheel. The gated loaders set this False; a test asserts every
  ``is_bundled`` judge is permissively licensed.

A judge also exposes :attr:`prompt_hash` (the SHA-256 of its frozen judge-prompt
or rubric string). The calibration gate refuses to publish if this hash drifts
(ROADMAP §8 "fails the build … if the frozen judge-prompt hash drifts").

ETHICS — the judges grade the BENIGN-canary robustness proxy injectkit uses by
default; "success" means the target emitted the benign marker it was told to
withhold, never harmful content. DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from ..evaluators.response_class import ResponseClass
from ..models import TargetResponse

__all__ = [
    "JudgeVerdict",
    "Judge",
    "JudgeRegistry",
    "registry",
    "register",
    "get_judge",
    "get_judge_class",
    "list_judges",
    "prompt_hash",
]


def prompt_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a frozen judge-prompt / rubric string.

    The hash is the tamper-evidence the calibration gate checks (ROADMAP §8):
    a judge records the hash of the exact prompt/rubric it was calibrated with,
    and the gate refuses to publish a number if the live hash differs from the
    one in :mod:`docs/JUDGES.md`. Normalises only trailing whitespace so a stray
    final newline does not flip the hash, but is otherwise byte-exact.
    """
    return hashlib.sha256(text.rstrip("\n").encode("utf-8")).hexdigest()


@dataclass
class JudgeVerdict:
    """The structured outcome of one :meth:`Judge.judge` call.

    Carries the three never-collapsed signals (ROADMAP §6.10) plus the judge id
    that produced them (for the stamp + circularity firewall). ``success_bool``
    is the binarised judge-ASR label; ``sr_score`` is the continuous 0..1
    StrongREJECT score; ``label_5class`` is the 5-class grade.
    """

    #: The 5-class grade (REJECT_IRRELEVANT … FULL).
    label_5class: ResponseClass
    #: Binarised success (judge-ASR contribution). For the StrongREJECT judge this
    #: is ``sr_score >= threshold``; for classifier judges it is the class call.
    success_bool: bool
    #: Continuous StrongREJECT-style score in [0, 1]. A non-rubric judge reports a
    #: degenerate 0/1 (its own binary confidence) so the field is always present.
    sr_score: float
    #: The :attr:`Judge.judge_id` of the judge that produced this verdict.
    judge_id: str = ""
    #: Human-readable explanation, for transcripts/reports.
    rationale: str = ""
    #: Optional per-signal extras (sub-scores, matched markers, …).
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive clamp so a misbehaving judge can never emit an out-of-range
        # score that corrupts a downstream mean.
        if self.sr_score < 0.0:
            self.sr_score = 0.0
        elif self.sr_score > 1.0:
            self.sr_score = 1.0


class Judge(ABC):
    """Abstract base for every offline judge (ROADMAP §6.10).

    A judge maps a target :class:`~injectkit.models.TargetResponse` (optionally
    with the attack/canary context) to a :class:`JudgeVerdict`. Concrete judges
    set three class attributes the gating machinery enforces and implement
    :meth:`judge`.

    The licence/bundling attributes are the legal firewall: only judges whose
    weights are permissively licensed may set ``is_bundled = True`` (the MIT
    from-scratch ``clean_cls`` is the sole bundleable model judge); the
    loader-gated Llama-Guard-3 / HarmBench-cls judges keep ``is_bundled = False``
    and carry their non-permissive upstream licence so a test (and JUDGES.md) can
    assert injectkit ships nothing it may not redistribute.
    """

    #: Registry key + stamp id (e.g. "clean_cls"). Set by every concrete judge.
    judge_id: str = ""
    #: SPDX-ish licence of the judge's WEIGHTS/logic. MIT for the bundleable
    #: default; the upstream non-permissive licence for the gated loaders.
    license: str = "MIT"
    #: True only when injectkit ships this judge's weights/logic in the wheel.
    is_bundled: bool = False
    #: Whether using this judge obliges "Built with Llama" attribution on
    #: redistribution (ROADMAP §6.10). True for the Llama-derived gated loaders.
    requires_llama_attribution: bool = False

    #: The frozen judge-prompt / rubric string the judge was calibrated against.
    #: Subclasses with a textual prompt/rubric override this; the hash is checked
    #: by the calibration gate. Empty for purely-lexical judges (substring/canary)
    #: whose "prompt" is their marker list — those report a stable hash of it.
    PROMPT: str = ""

    @property
    def prompt_hash(self) -> str:
        """SHA-256 of this judge's frozen prompt/rubric (ROADMAP §8 drift guard)."""
        return prompt_hash(self.PROMPT)

    @abstractmethod
    def judge(
        self,
        response: TargetResponse,
        *,
        canary: str = "",
        target: str = "",
        trigger: str = "",
    ) -> JudgeVerdict:
        """Grade ``response`` into the three never-collapsed signals.

        Args:
            response: The normalised target response to grade.
            canary: The per-run benign marker the attack tried to elicit (used by
                the canary/substring judges to recognise the benign success
                proxy). Empty when not applicable.
            target: The benign target string the attack optimised toward, if the
                caller wants to pass it explicitly (defaults to ``canary``-built).
            trigger: The benign-marker prefix (e.g. ``INJECTOK-``).

        Returns:
            A :class:`JudgeVerdict` carrying ``label_5class``, ``success_bool``,
            and ``sr_score``, stamped with this judge's :attr:`judge_id`.
        """
        ...


# --------------------------------------------------------------------------- #
# Registry — name -> Judge class, mirroring the white-box attack registry shape.
# --------------------------------------------------------------------------- #

_J = TypeVar("_J", bound=Judge)


class JudgeRegistry:
    """Name → :class:`Judge` *class* registry (mirrors the attack registry).

    Stores judge classes keyed by :attr:`Judge.judge_id`. Resolution constructs a
    fresh instance so judges are never shared mutable singletons. Re-registering a
    name raises unless ``override=True`` — a double-import that would silently
    shadow a judge is a bug, not a no-op.
    """

    def __init__(self) -> None:
        self._classes: dict[str, Type[Judge]] = {}

    def register(
        self, judge_id: str, cls: Type[_J], *, override: bool = False
    ) -> Type[_J]:
        """Register judge class ``cls`` under ``judge_id`` and return it unchanged."""
        if not judge_id:
            raise ValueError("judge registry id must be a non-empty string.")
        if not (isinstance(cls, type) and issubclass(cls, Judge)):
            raise ValueError(
                f"cannot register {cls!r} as judge {judge_id!r}: not a Judge subclass."
            )
        if judge_id in self._classes and not override:
            raise ValueError(
                f"judge {judge_id!r} is already registered "
                f"({self._classes[judge_id].__name__}); pass override=True to replace."
            )
        self._classes[judge_id] = cls
        return cls

    def get_class(self, judge_id: str) -> Type[Judge]:
        """Return the registered class for ``judge_id`` (KeyError if absent)."""
        try:
            return self._classes[judge_id]
        except KeyError:
            raise KeyError(
                f"unknown judge {judge_id!r}; registered: {sorted(self._classes)}"
            ) from None

    def get(self, judge_id: str, **kwargs: Any) -> Judge:
        """Resolve ``judge_id`` and return a fresh judge instance."""
        return self.get_class(judge_id)(**kwargs)

    def ids(self) -> list[str]:
        """All registered judge ids, sorted."""
        return sorted(self._classes)


#: The process-wide default judge registry.
registry = JudgeRegistry()


def register(
    judge_id: str, *, override: bool = False
) -> Callable[[Type[_J]], Type[_J]]:
    """Class decorator registering a :class:`Judge` under ``judge_id``.

    Sets ``cls.judge_id = judge_id`` so the registry key and the judge's
    self-reported id never drift, then wires it onto the default :data:`registry`.
    """

    def _decorator(cls: Type[_J]) -> Type[_J]:
        cls.judge_id = judge_id
        return registry.register(judge_id, cls, override=override)

    return _decorator


def get_judge(judge_id: str, **kwargs: Any) -> Judge:
    """Resolve ``judge_id`` to a fresh judge instance from the default registry."""
    return registry.get(judge_id, **kwargs)


def get_judge_class(judge_id: str) -> Type[Judge]:
    """Resolve ``judge_id`` to the registered judge class."""
    return registry.get_class(judge_id)


def list_judges() -> list[str]:
    """All registered judge ids on the default registry (sorted)."""
    return registry.ids()
