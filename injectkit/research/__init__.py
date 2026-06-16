"""Research dataset loaders — OPT-IN, GATED references to official academic data.

This package NEVER ships harmful prompts or harm-behavior data. It provides only
the *interface* for loading official, public academic prompt-injection /
jailbreak datasets, plus a reference list of their names and canonical URLs (see
``docs/RESEARCH-USE.md``). Loading any dataset requires an EXPLICIT, per-run
acknowledgment (a flag/env var) and downloads the data on demand from its
official source — injectkit does not redistribute it.

See :mod:`injectkit.research.base` for the gating rules and the loader contract.
"""

from __future__ import annotations

from .base import (
    RESEARCH_ACK_ENV,
    RESEARCH_DISCLAIMER,
    DatasetReference,
    ResearchAcknowledgmentError,
    ResearchDatasetLoader,
    require_acknowledgment,
)
from .datasets import (
    LOADERS,
    RefusalComplianceDetector,
    ResearchDownloadError,
    available_datasets,
    get_loader,
)
from .registry import KNOWN_DATASETS

__all__ = [
    "ResearchDatasetLoader",
    "DatasetReference",
    "ResearchAcknowledgmentError",
    "require_acknowledgment",
    "RESEARCH_ACK_ENV",
    "RESEARCH_DISCLAIMER",
    "KNOWN_DATASETS",
    "ResearchDownloadError",
    "RefusalComplianceDetector",
    "LOADERS",
    "available_datasets",
    "get_loader",
]
