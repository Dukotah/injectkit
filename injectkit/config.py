"""Configuration: load .injectkit.yaml and merge CLI args into a typed config.

The precedence is: built-in defaults < ``.injectkit.yaml`` file < CLI args.
The result is a :class:`Config` dataclass holding a fully-resolved
:class:`~injectkit.models.TargetConfig` plus scan-level options the engine and
CLI read (corpus path, judge on/off, fail-on threshold, report format/output).

This is a stub foundation: it implements the documented load+merge contract so
the CLI and engine builders can code against :class:`Config` and
``load_config`` today. Adapter-specific validation lives in the adapters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .models import Severity, TargetConfig

__all__ = ["Config", "load_config", "DEFAULT_CONFIG_FILENAME"]

DEFAULT_CONFIG_FILENAME = ".injectkit.yaml"

# Default models, authoritative per the Anthropic SDK facts:
#   - target uses claude-opus-4-8
#   - judge uses claude-haiku-4-5 (cheap/fast, runs per attack)
DEFAULT_TARGET_MODEL = "claude-opus-4-8"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"


@dataclass
class Config:
    """Fully-resolved scan configuration."""

    target: TargetConfig = field(default_factory=TargetConfig)
    # Path to corpus dir or file; None => the bundled corpus shipped in-package.
    corpus_path: Optional[str] = None
    # LLM judge: off unless explicitly enabled (lazy-imports anthropic).
    use_judge: bool = False
    judge_model: str = DEFAULT_JUDGE_MODEL
    # Minimum finding severity that makes the scan "fail" (non-zero exit).
    fail_on: Severity = Severity.HIGH
    # Report format and optional output file.
    report_format: str = "terminal"
    out_path: Optional[str] = None
    # Optional list of techniques/tags to include (None => all).
    techniques: Optional[list[str]] = None

    def bundled_corpus_dir(self) -> str:
        """Absolute path to the corpus directory bundled inside the package."""
        return os.path.join(os.path.dirname(__file__), "corpus")

    def resolved_corpus_path(self) -> str:
        """The corpus path to load: explicit ``corpus_path`` or the bundle."""
        return self.corpus_path or self.bundled_corpus_dir()


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Read a .injectkit.yaml file into a dict (empty dict if missing)."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config root must be a mapping")
    return data


def _build_target_config(file_target: dict[str, Any], overrides: dict[str, Any]) -> TargetConfig:
    """Merge file 'target' block with CLI overrides into a TargetConfig."""
    merged: dict[str, Any] = dict(file_target or {})
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    tc = TargetConfig()
    for attr in (
        "kind", "name", "model", "system", "max_tokens", "url", "method",
        "headers", "request_template", "response_path", "timeout_s",
        "mcp_command", "mcp_args", "mcp_url", "extra",
    ):
        if attr in merged and merged[attr] is not None:
            setattr(tc, attr, merged[attr])

    # Default the anthropic model if the kind is anthropic and none was set.
    if tc.kind == "anthropic" and not tc.model:
        tc.model = DEFAULT_TARGET_MODEL
    return tc


def load_config(
    config_path: Optional[str] = None,
    *,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> Config:
    """Load and merge configuration into a :class:`Config`.

    Args:
        config_path: Path to a ``.injectkit.yaml``. If ``None``, look for
            ``.injectkit.yaml`` in the current working directory.
        cli_overrides: Flat dict of CLI-supplied values. Recognized keys mirror
            :class:`Config` and :class:`TargetConfig` fields, plus a nested
            ``target`` dict of target overrides. ``None`` values are ignored so
            absent CLI flags don't clobber file/default values.

    Returns:
        A resolved :class:`Config`.
    """
    cli = dict(cli_overrides or {})
    path = config_path or os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
    file_cfg = _load_yaml_config(path)

    target_overrides: dict[str, Any] = dict(cli.pop("target", {}) or {})
    target = _build_target_config(file_cfg.get("target", {}), target_overrides)

    cfg = Config(target=target)

    # Scan-level options: file first, then CLI override (if not None).
    def pick(key: str, default: Any) -> Any:
        if key in cli and cli[key] is not None:
            return cli[key]
        if key in file_cfg and file_cfg[key] is not None:
            return file_cfg[key]
        return default

    cfg.corpus_path = pick("corpus_path", None)
    cfg.use_judge = bool(pick("use_judge", False))
    cfg.judge_model = str(pick("judge_model", DEFAULT_JUDGE_MODEL))
    cfg.report_format = str(pick("report_format", "terminal"))
    cfg.out_path = pick("out_path", None)

    fail_on = pick("fail_on", Severity.HIGH)
    cfg.fail_on = Severity.coerce(fail_on) if fail_on is not None else Severity.HIGH

    techniques = pick("techniques", None)
    if isinstance(techniques, str):
        techniques = [t.strip() for t in techniques.split(",") if t.strip()]
    cfg.techniques = techniques

    return cfg
