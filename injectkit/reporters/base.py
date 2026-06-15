"""The Reporter protocol — the contract every report renderer implements.

A Reporter turns a :class:`~injectkit.models.ScanReport` into a string in some
format (terminal text, JSON, Markdown, SARIF, HTML). The CLI selects a reporter
by ``--format`` and either prints the string or writes it to ``--out``.

Every rendered report MUST carry the authorized-use notice (injectkit is a
defensive tool); reporters can pull that line from
:data:`AUTHORIZED_USE_NOTICE`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import ScanReport

__all__ = ["Reporter", "AUTHORIZED_USE_NOTICE"]

#: Standard notice every report embeds. Keep wording in sync with SECURITY.md.
AUTHORIZED_USE_NOTICE = (
    "injectkit is a defensive tool. Scan only endpoints you own or are "
    "explicitly authorized to test."
)


@runtime_checkable
class Reporter(Protocol):
    """Renders a :class:`ScanReport` to a string for display or a file.

    ``render`` must be pure and side-effect free (writing files is the CLI's
    job). The returned string is the complete report in the reporter's format.
    """

    #: Format identifier matching the CLI ``--format`` value (e.g. "json").
    name: str
    #: Suggested file extension when writing to disk (e.g. ".json").
    extension: str

    def render(self, report: ScanReport) -> str:
        """Render ``report`` to a string in this reporter's format."""
        ...
