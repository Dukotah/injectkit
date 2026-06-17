"""injectkit v0.4 bench harness — generalized ASR + leaderboard + repro stamp.

CHUNK 7-bench-harness (ROADMAP §3, §6.10, §8). This package generalises the ASR
benchmark into the leaderboard machinery the rest of v0.4 feeds:

* :mod:`injectkit.bench.harness` — the generalized cell runner. One call,
  :func:`run_cell`, turns ``attack_name × model × behavior_set × num_seeds ×
  judge_id`` into an aggregated ASR ± Wilson CI (per-behavior/seed run via the
  white-box attack registry + the model zoo + the judge layer + the greedy
  generation seam). The three never-collapsed signals — substring-ASR, judge-ASR,
  StrongREJECT-mean — are aggregated separately.
* :mod:`injectkit.bench.leaderboard` — the model × attack matrix. Primary columns
  are the three signals (each with its CI); metadata columns are avg-queries /
  GPU-hours / wall-clock / quant. Exports to CSV + JSON + Markdown.
* :mod:`injectkit.bench.stamp` — the reproducibility stamp with ALL 8 mandatory
  fields (version, corpus-hash, model-revision, seed, quant, judge-id, attack-id,
  backend); ``quant`` is mandatory and never defaulted.

Two seeded runs of the same cell reproduce within the CI (:func:`runs_reproduce`),
and a single-cell ASR with a full stamp runs on a tiny CPU model offline. The 8B /
fp16-vs-4bit anchor cells need a GPU + multi-GB download and are DEFERRED-NO-GPU;
their code paths exist (the zoo loader + the harness ``ModelSpec.loader`` seam) and
are exercised on a tiny model / offline seam.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from .harness import (
    ASRStat,
    Behavior,
    BehaviorRun,
    CellResult,
    ModelSpec,
    run_cell,
    runs_reproduce,
    wilson_interval,
)
from .leaderboard import (
    METADATA_COLUMNS,
    PRIMARY_COLUMNS,
    Leaderboard,
)
from .stamp import (
    STAMP_FIELDS,
    VALID_BACKENDS,
    VALID_QUANTS,
    ReproStamp,
    StampError,
    build_stamp,
    corpus_hash,
    stamps_reproduce,
)

__all__ = [
    # harness
    "ASRStat",
    "Behavior",
    "BehaviorRun",
    "CellResult",
    "ModelSpec",
    "run_cell",
    "runs_reproduce",
    "wilson_interval",
    # leaderboard
    "Leaderboard",
    "PRIMARY_COLUMNS",
    "METADATA_COLUMNS",
    # stamp
    "ReproStamp",
    "StampError",
    "STAMP_FIELDS",
    "VALID_QUANTS",
    "VALID_BACKENDS",
    "build_stamp",
    "corpus_hash",
    "stamps_reproduce",
]
