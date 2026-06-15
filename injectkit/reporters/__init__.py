"""Reporters: render a ScanReport to terminal, JSON, Markdown, SARIF, or HTML.

Every reporter implements the :class:`~injectkit.reporters.base.Reporter`
protocol: ``render(report) -> str``. The CLI picks a reporter by ``--format``.
"""

from __future__ import annotations

from .base import Reporter

__all__ = ["Reporter"]
