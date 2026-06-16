"""The ResearchDatasetLoader interface — OPT-IN, GATED, never bundles data.

injectkit's bundled corpus uses benign canary proxies and ships no harmful
content. For users doing *authorized academic research* who want to benchmark a
target they own against the official public datasets from the prompt-injection /
jailbreak literature, this module defines a loader interface that:

1. **Bundles no data.** Only dataset *names* and canonical *URLs* are referenced
   (see :mod:`injectkit.research.registry` and ``docs/RESEARCH-USE.md``). The
   data itself is downloaded on demand from its official source; injectkit never
   redistributes it.
2. **Is opt-in and gated.** Every load requires an EXPLICIT acknowledgment for
   that run: either ``acknowledge=True`` passed to the loader, the
   ``--i-am-authorized`` CLI flag (paired with ``--research-benchmark
   <dataset>``), or the ``INJECTKIT_RESEARCH_ACK=1`` environment variable.
   Without it, :meth:`load` raises :class:`ResearchAcknowledgmentError` carrying
   the research-use disclaimer.
3. **Stays defensive.** Loaded prompts are used to measure a target's robustness
   (does my system resist these documented attacks?), under the same
   authorized-use posture as the rest of the toolkit.

This module is interface-only: the concrete download/parse logic for each
dataset is a future builder's job, lazy-importing any optional dependency
(``datasets``/``requests``) and honouring the gate below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..models import Attack

__all__ = [
    "RESEARCH_ACK_ENV",
    "RESEARCH_DISCLAIMER",
    "ResearchAcknowledgmentError",
    "DatasetReference",
    "ResearchDatasetLoader",
    "require_acknowledgment",
]

#: Environment variable that, when set to a truthy value ("1"/"true"/"yes"),
#: acknowledges the research-use terms for the whole process.
RESEARCH_ACK_ENV = "INJECTKIT_RESEARCH_ACK"

#: The disclaimer surfaced whenever a research dataset load is attempted without
#: acknowledgment. Kept in sync with docs/RESEARCH-USE.md.
RESEARCH_DISCLAIMER = (
    "Research datasets are loaded ONLY for authorized defensive research on "
    "targets you own or are explicitly permitted to test. These datasets are "
    "maintained by third parties, are downloaded from their official sources "
    "(injectkit does not redistribute them), and may contain sensitive or "
    "offensive material. By acknowledging, you confirm authorized, ethical, "
    "research-only use and acceptance of each dataset's own licence/terms. See "
    "docs/RESEARCH-USE.md."
)


class ResearchAcknowledgmentError(RuntimeError):
    """Raised when a research dataset load is attempted without acknowledgment.

    Its message always includes :data:`RESEARCH_DISCLAIMER` and how to opt in
    (the ``acknowledge=True`` argument, the ``--i-am-authorized`` flag, or the
    ``INJECTKIT_RESEARCH_ACK`` env var).
    """


def require_acknowledgment(acknowledge: bool = False) -> None:
    """Enforce the research-use gate; raise if the user has not opted in.

    The gate passes if either ``acknowledge`` is True (explicit per-call opt-in,
    e.g. from the ``--i-am-authorized`` flag) OR the
    :data:`RESEARCH_ACK_ENV` environment variable is set truthy. Otherwise it
    raises :class:`ResearchAcknowledgmentError` with the disclaimer.

    Args:
        acknowledge: Explicit per-call acknowledgment (the CLI passes the value
            of ``--i-am-authorized`` here).

    Raises:
        ResearchAcknowledgmentError: if neither acknowledgment path is satisfied.
    """
    env = os.environ.get(RESEARCH_ACK_ENV, "").strip().lower()
    env_ack = env in {"1", "true", "yes", "on"}
    if acknowledge or env_ack:
        return
    raise ResearchAcknowledgmentError(
        "Research dataset access is opt-in and gated.\n\n"
        f"{RESEARCH_DISCLAIMER}\n\n"
        "To proceed, pass --research-benchmark <dataset> --i-am-authorized on "
        f"the CLI, construct the loader with acknowledge=True, or set "
        f"{RESEARCH_ACK_ENV}=1 in your environment."
    )


@dataclass
class DatasetReference:
    """A reference to an official public research dataset — names/URLs, no data.

    Describes where a dataset lives and how to cite it. injectkit ships these
    references (see :mod:`injectkit.research.registry`) but NEVER the underlying
    prompts; a loader downloads from ``url`` on explicit opt-in.
    """

    #: Stable key used on the CLI (e.g. "advbench", "harmbench", "jailbreakbench").
    key: str
    #: Human-readable dataset name.
    name: str
    #: Canonical homepage / repository / dataset-card URL (official source).
    url: str
    #: One-line description of what the dataset contains and its intended use.
    description: str
    #: Citation / paper reference (e.g. arXiv id or BibTeX key), if any.
    citation: str = ""
    #: SPDX-ish licence note or "see source" — the user must honour the source's
    #: actual licence/terms.
    license_note: str = "see source"
    #: Optional tags (e.g. ["jailbreak", "harmful-behaviors"]).
    tags: list[str] = field(default_factory=list)


@runtime_checkable
class ResearchDatasetLoader(Protocol):
    """Loads an official public research dataset into injectkit :class:`Attack`s.

    Implementations MUST call :func:`require_acknowledgment` before any download,
    lazy-import their optional download dependency, and download from the
    dataset's official source rather than bundling it. The returned attacks carry
    the benign-proxy scoring conventions where possible (a per-run canary success
    condition), so the existing engine/detector path scores them.
    """

    #: The dataset this loader handles (its :class:`DatasetReference`).
    reference: DatasetReference

    def load(
        self,
        *,
        acknowledge: bool = False,
        limit: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> list[Attack]:
        """Download (on opt-in) and return the dataset as :class:`Attack` objects.

        Args:
            acknowledge: Explicit per-call research-use acknowledgment. The
                loader passes this to :func:`require_acknowledgment`, which also
                honours :data:`RESEARCH_ACK_ENV`.
            limit: Optional cap on the number of attacks returned (for quick
                runs); ``None`` loads the full set.
            cache_dir: Optional directory to cache the downloaded data in; the
                loader downloads once and reuses it on subsequent runs.

        Returns:
            A list of :class:`Attack` objects.

        Raises:
            ResearchAcknowledgmentError: if the research-use gate is not
                satisfied (raised before any download).
        """
        ...
