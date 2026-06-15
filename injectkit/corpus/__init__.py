"""Attack corpus: data-driven prompt-injection test cases as YAML files.

One YAML file per technique (direct_injection, indirect_injection, jailbreak,
system_prompt_leak, tool_abuse, data_exfiltration). The community adds attacks
by PRing new YAML entries. :func:`~injectkit.corpus.loader.load_corpus` parses
and validates them into :class:`~injectkit.models.Attack` objects.
"""

from __future__ import annotations

from .loader import CorpusError, load_attack_file, load_corpus

__all__ = ["load_corpus", "load_attack_file", "CorpusError"]
