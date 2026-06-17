"""The leaderboard — a model × attack matrix exported to CSV / JSON / Markdown.

CHUNK 7-bench-harness (ROADMAP §3, §6.10, §8). A :class:`Leaderboard` collects the
:class:`~injectkit.bench.harness.CellResult`s produced by the harness and renders
them as the model × attack robustness matrix. The PRIMARY columns are the three
never-collapsed signals (ROADMAP §3/§6.10 — "reported SEPARATELY, never one
number"):

* ``substring-ASR``      — the loosest baseline (refusal-prefix matcher).
* ``judge-ASR``          — the calibrated EVAL judge's success rate.
* ``StrongREJECT-mean``  — the continuous 0..1 rubric mean (the headline metric).

Each is shown with its confidence interval. The METADATA columns carry the
provenance/budget the ROADMAP requires beside every number: avg-queries,
GPU-hours, wall-clock, and **quant** (the mandatory quant column from the stamp).

Exports: :meth:`Leaderboard.to_json`, :meth:`to_csv`, :meth:`to_markdown` (and the
``export`` convenience that writes all three). The Markdown table is the
human-facing leaderboard; the CSV/JSON are the machine-readable artifacts a CI gate
or a paper's results table consumes. Every row also serialises its full 8-field
stamp so a published row can always be audited back to the exact run.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .harness import ASRStat, CellResult

__all__ = [
    "PRIMARY_COLUMNS",
    "METADATA_COLUMNS",
    "Leaderboard",
]

#: The primary leaderboard columns (the three never-collapsed signals).
PRIMARY_COLUMNS = ("substring_asr", "judge_asr", "strongreject_mean")

#: The metadata columns surfaced beside the primary signals (quant is mandatory).
METADATA_COLUMNS = ("avg_queries", "gpu_hours", "wall_clock_s", "quant")

#: The flat CSV header — identity + primaries (+CI) + metadata + the 8 stamp fields.
_CSV_HEADER = (
    "model",
    "attack",
    "judge_id",
    "backend",
    "quant",
    "n_behaviors",
    "seeds",
    "substring_asr",
    "substring_ci_low",
    "substring_ci_high",
    "judge_asr",
    "judge_ci_low",
    "judge_ci_high",
    "strongreject_mean",
    "sr_ci_low",
    "sr_ci_high",
    "avg_queries",
    "gpu_hours",
    "wall_clock_s",
    # The 8 mandatory stamp fields (audit trail on every row).
    "stamp_version",
    "stamp_corpus_hash",
    "stamp_model_revision",
    "stamp_seed",
    "stamp_quant",
    "stamp_judge_id",
    "stamp_attack_id",
    "stamp_backend",
)


@dataclass
class Leaderboard:
    """A model × attack robustness matrix built from harness cell results.

    Append :class:`~injectkit.bench.harness.CellResult`s (one per cell) with
    :meth:`add`, then export. The matrix axes are the union of the models and
    attacks seen across the added cells; a missing (model, attack) cell renders
    blank. ``title`` is shown above the Markdown table.
    """

    title: str = "injectkit robustness leaderboard"
    cells: list[CellResult] = field(default_factory=list)

    def add(self, cell: CellResult) -> "Leaderboard":
        """Append one cell result and return self (chainable)."""
        self.cells.append(cell)
        return self

    def extend(self, cells: Iterable[CellResult]) -> "Leaderboard":
        """Append many cell results and return self."""
        for cell in cells:
            self.add(cell)
        return self

    # -- axes ------------------------------------------------------------- #

    def models(self) -> list[str]:
        """The distinct models in the matrix, in first-seen order."""
        return _ordered_unique(c.model for c in self.cells)

    def attacks(self) -> list[str]:
        """The distinct attacks in the matrix, in first-seen order."""
        return _ordered_unique(c.attack_id for c in self.cells)

    def cell(self, model: str, attack: str) -> Optional[CellResult]:
        """The cell for ``(model, attack)`` or None if absent."""
        for c in self.cells:
            if c.model == model and c.attack_id == attack:
                return c
        return None

    # -- JSON ------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """The full leaderboard as a JSON-serialisable dict (rows + axes)."""
        return {
            "title": self.title,
            "models": self.models(),
            "attacks": self.attacks(),
            "primary_columns": list(PRIMARY_COLUMNS),
            "metadata_columns": list(METADATA_COLUMNS),
            "cells": [c.as_dict() for c in self.cells],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the leaderboard to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # -- CSV -------------------------------------------------------------- #

    def to_csv(self) -> str:
        """Serialise the leaderboard to a flat CSV (one row per cell).

        Every row carries the primary signals with their CIs, the metadata columns
        (incl. the mandatory quant), and the full 8-field stamp so the CSV is a
        complete, auditable artifact on its own.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(_CSV_HEADER)
        for c in self.cells:
            s = c.stamp.to_dict()
            writer.writerow(
                (
                    c.model,
                    c.attack_id,
                    c.judge_id,
                    c.backend,
                    c.quant,
                    c.n_behaviors,
                    " ".join(str(x) for x in c.seeds),
                    _r(c.substring_asr.rate),
                    _r(c.substring_asr.lo),
                    _r(c.substring_asr.hi),
                    _r(c.judge_asr.rate),
                    _r(c.judge_asr.lo),
                    _r(c.judge_asr.hi),
                    _r(c.strongreject_mean.rate),
                    _r(c.strongreject_mean.lo),
                    _r(c.strongreject_mean.hi),
                    _r(c.avg_queries),
                    _r(c.gpu_hours),
                    _r(c.wall_clock_s),
                    s["version"],
                    s["corpus_hash"],
                    s["model_revision"],
                    s["seed"],
                    s["quant"],
                    s["judge_id"],
                    s["attack_id"],
                    s["backend"],
                )
            )
        return buf.getvalue()

    # -- Markdown --------------------------------------------------------- #

    def to_markdown(self) -> str:
        """Render the leaderboard as a Markdown document (the human-facing table).

        Emits two tables: a primary model × attack matrix showing
        ``judge-ASR [CI]`` per cell (the headline view), and a per-cell detail table
        carrying all three signals + the metadata columns (incl. quant) + the corpus
        hash, so the Markdown is both a glanceable matrix and a full record.
        """
        models = self.models()
        attacks = self.attacks()
        lines: list[str] = [f"# {self.title}", ""]

        lines.append(
            "Three signals reported **separately** (substring-ASR, judge-ASR, "
            "StrongREJECT-mean) — never collapsed into one number. Cells below "
            "show **judge-ASR** with its 95% CI; full per-signal detail follows."
        )
        lines.append("")

        # Primary matrix: judge-ASR per (model, attack) cell.
        header = "| model \\ attack | " + " | ".join(attacks) + " |"
        sep = "| --- | " + " | ".join("---" for _ in attacks) + " |"
        lines.append(header)
        lines.append(sep)
        for model in models:
            row = [model]
            for attack in attacks:
                c = self.cell(model, attack)
                row.append(_fmt_asr(c.judge_asr) if c is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Detail table: all three signals + metadata + provenance per cell.
        lines.append("## Per-cell detail")
        lines.append("")
        detail_header = (
            "| model | attack | judge | quant | substring-ASR | judge-ASR | "
            "StrongREJECT-mean | avg-queries | GPU-hours | wall-clock(s) | "
            "corpus-hash |"
        )
        detail_sep = "| " + " | ".join("---" for _ in range(11)) + " |"
        lines.append(detail_header)
        lines.append(detail_sep)
        for c in self.cells:
            lines.append(
                "| "
                + " | ".join(
                    (
                        c.model,
                        c.attack_id,
                        c.judge_id,
                        c.quant,
                        _fmt_asr(c.substring_asr),
                        _fmt_asr(c.judge_asr),
                        _fmt_mean(c.strongreject_mean),
                        f"{c.avg_queries:.1f}",
                        f"{c.gpu_hours:.3f}",
                        f"{c.wall_clock_s:.2f}",
                        c.stamp.corpus_hash[:12],
                    )
                )
                + " |"
            )
        lines.append("")
        lines.append(
            "_Every row is reproducible from its 8-field stamp "
            "(version, corpus-hash, model-revision, seed, quant, judge-id, "
            "attack-id, backend); see the CSV/JSON export for the full stamps._"
        )
        lines.append("")
        return "\n".join(lines)

    # -- combined export -------------------------------------------------- #

    def export(self, out_dir: "str | Path", *, stem: str = "leaderboard") -> dict[str, Path]:
        """Write CSV + JSON + Markdown into ``out_dir`` and return the paths.

        Creates ``out_dir`` if needed and writes ``{stem}.csv``, ``{stem}.json``,
        ``{stem}.md``. Returns ``{"csv": ..., "json": ..., "markdown": ...}``.
        """
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "csv": d / f"{stem}.csv",
            "json": d / f"{stem}.json",
            "markdown": d / f"{stem}.md",
        }
        paths["csv"].write_text(self.to_csv(), encoding="utf-8")
        paths["json"].write_text(self.to_json(), encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(), encoding="utf-8")
        return paths


# --------------------------------------------------------------------------- #
# Formatting helpers.
# --------------------------------------------------------------------------- #


def _ordered_unique(items: Iterable[str]) -> list[str]:
    """First-seen-order de-duplication (stable matrix axes)."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def _fmt_asr(stat: ASRStat) -> str:
    """``42.0% [30.1, 55.2]`` — a rate with its CI, as percentages."""
    return (
        f"{stat.rate * 100:.1f}% "
        f"[{stat.lo * 100:.1f}, {stat.hi * 100:.1f}]"
    )


def _fmt_mean(stat: ASRStat) -> str:
    """``0.62 [0.50, 0.74]`` — the continuous SR mean with its CI (0..1)."""
    return f"{stat.rate:.3f} [{stat.lo:.3f}, {stat.hi:.3f}]"


def _r(value: float) -> str:
    """Round a float to 6 dp for the CSV (stable, compact)."""
    return f"{float(value):.6f}"
