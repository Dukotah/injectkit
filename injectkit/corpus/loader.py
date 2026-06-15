"""Load and validate YAML attack corpus files into Attack objects.

A corpus file is a YAML document with a top-level ``attacks:`` list (a bare
list at the top level is also accepted). Each entry is validated against the
attack schema:

    id                 (str, required, unique within a load)
    technique          (str, required)
    name               (str, required)
    description        (str, required)
    severity           (info|low|medium|high|critical, required)
    payload            (str, required; may contain a {canary} placeholder)
    success_conditions (mapping, required; detector rules, see below)
    references         (list[str], optional)
    tags               (list[str], optional)
    system             (str, optional)
    context            (str, optional)

success_conditions rule keys (all optional, at least one recommended):
    marker_echoed       (bool)        - the literal success marker appears
    canary_in_output    (bool)        - the per-run canary appears anywhere
    regex               (str | list)  - any pattern matches the response text
    refusal_absent      (bool)        - success requires the model NOT refuse
    system_prompt_leaked(bool|str|list) - system-prompt-leak marker(s) appear

Validation errors raise :class:`CorpusError` with a message naming the file and
the offending attack id/index.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import yaml

from ..models import Attack, Severity

__all__ = ["load_corpus", "load_attack_file", "CorpusError"]

# Keys allowed in a success_conditions block. The heuristics detector defines
# their semantics; the loader only validates shape/presence.
_VALID_CONDITION_KEYS = {
    "marker_echoed",
    "canary_in_output",
    "regex",
    "refusal_absent",
    "system_prompt_leaked",
}

_REQUIRED_FIELDS = ("id", "technique", "name", "description", "severity", "payload")


class CorpusError(ValueError):
    """Raised when an attack corpus file is malformed or fails validation."""


def _validate_attack(raw: Any, source_file: str, index: int) -> Attack:
    """Validate one raw mapping and build an :class:`Attack`."""
    where = f"{source_file} (attack #{index})"
    if not isinstance(raw, dict):
        raise CorpusError(f"{where}: each attack must be a mapping, got {type(raw).__name__}")

    for fieldname in _REQUIRED_FIELDS:
        if fieldname not in raw or raw[fieldname] in (None, ""):
            raise CorpusError(f"{where}: missing required field '{fieldname}'")

    attack_id = str(raw["id"])
    where = f"{source_file} (attack '{attack_id}')"

    try:
        severity = Severity.coerce(raw["severity"])
    except (ValueError, KeyError):
        raise CorpusError(
            f"{where}: invalid severity {raw['severity']!r}; "
            f"must be one of {[s.value for s in Severity]}"
        )

    conditions = raw.get("success_conditions") or {}
    if not isinstance(conditions, dict):
        raise CorpusError(f"{where}: success_conditions must be a mapping")
    unknown = set(conditions) - _VALID_CONDITION_KEYS
    if unknown:
        raise CorpusError(
            f"{where}: unknown success_conditions keys {sorted(unknown)}; "
            f"valid keys: {sorted(_VALID_CONDITION_KEYS)}"
        )

    def _as_str_list(value: Any, field_label: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        raise CorpusError(f"{where}: '{field_label}' must be a string or list of strings")

    return Attack(
        id=attack_id,
        technique=str(raw["technique"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        severity=severity,
        payload=str(raw["payload"]),
        success_conditions=dict(conditions),
        references=_as_str_list(raw.get("references"), "references"),
        tags=_as_str_list(raw.get("tags"), "tags"),
        system=str(raw["system"]) if raw.get("system") not in (None, "") else None,
        context=str(raw["context"]) if raw.get("context") not in (None, "") else None,
        source_file=source_file,
    )


def load_attack_file(path: str) -> list[Attack]:
    """Load and validate a single corpus YAML file into a list of Attacks."""
    if not os.path.isfile(path):
        raise CorpusError(f"corpus file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise CorpusError(f"{path}: YAML parse error: {exc}") from exc

    if doc is None:
        return []
    if isinstance(doc, dict):
        items = doc.get("attacks", [])
    elif isinstance(doc, list):
        items = doc
    else:
        raise CorpusError(f"{path}: top level must be a mapping with 'attacks:' or a list")

    if not isinstance(items, list):
        raise CorpusError(f"{path}: 'attacks' must be a list")

    return [_validate_attack(raw, path, i) for i, raw in enumerate(items)]


def load_corpus(path: str) -> list[Attack]:
    """Load every attack from a corpus path.

    ``path`` may be a single ``.yaml``/``.yml`` file or a directory. When a
    directory is given, all ``*.yaml`` and ``*.yml`` files in it are loaded
    (sorted by filename for determinism). Duplicate attack ids across the whole
    load raise :class:`CorpusError`.

    Returns:
        A list of validated :class:`Attack` objects.
    """
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "*.yaml")) + glob.glob(os.path.join(path, "*.yml"))
        )
    elif os.path.isfile(path):
        files = [path]
    else:
        raise CorpusError(f"corpus path not found: {path}")

    attacks: list[Attack] = []
    seen: dict[str, str] = {}
    for f in files:
        for attack in load_attack_file(f):
            if attack.id in seen:
                raise CorpusError(
                    f"duplicate attack id '{attack.id}' in {f} "
                    f"(already defined in {seen[attack.id]})"
                )
            seen[attack.id] = f
            attacks.append(attack)
    return attacks
