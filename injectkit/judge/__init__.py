"""injectkit v0.4 offline judge layer (ROADMAP §6.10 / §11 chunk 4).

The license-clean, fully-offline scoring layer that replaces the benign-canary-only
success signal with a calibrated judge before any leaderboard number is published.
Three signals are reported **separately and never collapsed** (ROADMAP §3/§6.10):

* **substring-ASR** — :class:`SubstringJudge` (refusal-prefix matcher; report-only
  baseline, deliberately the loosest signal).
* **judge-ASR** — the binarised success label from the DEFAULT EVAL judge,
  :class:`CleanClassifierJudge` (``clean_cls``): an MIT, from-scratch classifier
  trained on the bundled calibration set, the ONLY bundleable model judge.
* **StrongREJECT-mean** — :class:`StrongRejectJudge`, the continuous 0..1 rubric
  (the primary leaderboard column; binarisation at θ is a reporting choice).

Plus :class:`CanaryJudge` (the benign-echo default for tests/examples) and two
OPTIONAL, gated, NOT-bundled loaders — :class:`HarmBenchClassifierJudge` and
:class:`LlamaGuardJudge` — whose Llama-derived weights are never shipped and which
error gracefully without HF auth.

Importing this package registers every built-in judge, so
``injectkit.judge.get_judge("clean_cls")`` resolves after ``import injectkit.judge``.
At least two distinct judge ids are always resolvable (ROADMAP §6.10.1: the
optimisation judge must differ from the evaluation judge).

The calibration GATE (:func:`assert_calibrated`) fails the build if the default
judge falls below κ ≥ 0.6 / ≥ 85% agreement, or if its frozen prompt/feature hash
drifts (ROADMAP §8). See ``docs/JUDGES.md`` for licences, gating, and hashes.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from .base import (
    Judge,
    JudgeRegistry,
    JudgeVerdict,
    get_judge,
    get_judge_class,
    list_judges,
    prompt_hash,
    register,
    registry,
)

# Import the concrete judges for their @register side effect.
from .substring import SubstringJudge
from .canary import CanaryJudge
from .clean_cls import (
    CALIBRATION_PATH,
    FEATURE_NAMES,
    CleanClassifierJudge,
    extract_features,
)
from .strongreject import DEFAULT_BINARIZE_THETA, StrongRejectJudge
from .harmbench_cls import GatedJudgeUnavailable, HarmBenchClassifierJudge
from .llama_guard import LlamaGuardJudge
from .calibration import (
    AGREEMENT_FLOOR,
    KAPPA_FLOOR,
    CalibrationResult,
    assert_calibrated,
    calibrate_judge,
    cohen_kappa,
)

#: The default EVAL judge id (ROADMAP §6.10): the bundleable MIT classifier.
DEFAULT_EVAL_JUDGE = "clean_cls"
#: A sensible small/cheap OPTIMISATION-loop judge id, distinct from the eval judge
#: (ROADMAP §6.10.1 circularity firewall). The substring matcher is bundled, fast,
#: and never the eval judge, so it is a safe in-loop default for tests.
DEFAULT_OPT_JUDGE = "substring"

__all__ = [
    # ABC + registry.
    "Judge",
    "JudgeVerdict",
    "JudgeRegistry",
    "registry",
    "register",
    "get_judge",
    "get_judge_class",
    "list_judges",
    "prompt_hash",
    # Judges.
    "SubstringJudge",
    "CanaryJudge",
    "CleanClassifierJudge",
    "StrongRejectJudge",
    "HarmBenchClassifierJudge",
    "LlamaGuardJudge",
    "GatedJudgeUnavailable",
    # clean_cls internals.
    "CALIBRATION_PATH",
    "FEATURE_NAMES",
    "extract_features",
    "DEFAULT_BINARIZE_THETA",
    # Calibration gate.
    "KAPPA_FLOOR",
    "AGREEMENT_FLOOR",
    "CalibrationResult",
    "cohen_kappa",
    "calibrate_judge",
    "assert_calibrated",
    # Defaults.
    "DEFAULT_EVAL_JUDGE",
    "DEFAULT_OPT_JUDGE",
]
