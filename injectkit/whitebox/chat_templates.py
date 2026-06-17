"""Bundled chat-template fixtures for the 5 dense zoo families (OFFLINE, no weights).

CHUNK 3-gcg-advprefix. The hardened GCG optimiser must locate the optimisable
*span* inside a model's rendered chat prompt **without hard-coded offsets**
(ROADMAP correctness trap: "tokenizer-agnostic chat-template slice location for
Llama-3/Qwen/Gemma/Mistral/Phi — never hard-coded offsets"). The real slice
logic lives in :mod:`injectkit.whitebox.gcg` and runs against whatever tokenizer
the zoo loads.

This module ships **structurally faithful Jinja2 chat-template strings** for the
five dense families so the slice-location logic is testable **fully offline** —
with just *a* tokenizer (e.g. ``gpt2``) wearing each family's template via
``tokenizer.apply_chat_template(..., chat_template=TEMPLATE)``. No gated
tokenizer download (Llama/Gemma are gated), no weights, no network.

These are *fixtures for testing the slicing algorithm*, NOT a redistribution of
any vendor tokenizer. Each carries the family's real role-delimiter special-token
strings (``<|start_header_id|>`` etc.) so that the generation-prompt boundary the
GCG slicer keys off is reproduced exactly. The production path always uses the
model's own ``tokenizer.chat_template`` (pinned via the zoo); these are only the
fallback the offline round-trip test exercises.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

__all__ = [
    "CHAT_TEMPLATES",
    "GENERATION_PROMPT_MARKERS",
    "template_for",
]


# Each template is a minimal-but-structurally-faithful rendering of the family's
# real chat format: the same role-delimiter special tokens and the same
# add_generation_prompt suffix the real template emits. The GCG slicer never
# parses these — it inserts the optim/target between rendered segments — so only
# the *boundary structure* (role headers + generation prompt) needs to be real.

#: Llama 3 / 3.1 — header-id delimited turns, <|eot_id|> turn terminator.
_LLAMA3 = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

#: Qwen2.5 — ChatML (<|im_start|>/<|im_end|>).
_QWEN = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

#: Gemma 2 — <start_of_turn>/<end_of_turn>; 'assistant' role maps to 'model'.
_GEMMA = (
    "{% for message in messages %}"
    "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)

#: Mistral v0.3 — [INST] ... [/INST] instruction wrapping.
_MISTRAL = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% else %}{{ message['content'] + '</s>' }}{% endif %}"
    "{% endfor %}"
)

#: Phi-4 — <|im_start|>/<|im_end|> with an explicit <|im_sep|> role separator.
_PHI = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '<|im_sep|>' + message['content'] + '<|im_end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
)


#: Family key -> faithful Jinja2 chat-template string. Keys mirror the zoo names'
#: family stems so a test can iterate the five dense families by name.
CHAT_TEMPLATES: dict[str, str] = {
    "llama-3": _LLAMA3,
    "qwen": _QWEN,
    "gemma": _GEMMA,
    "mistral": _MISTRAL,
    "phi": _PHI,
}


#: The literal generation-prompt header each family emits when
#: ``add_generation_prompt=True``. Used only to *assert* in tests that the slice
#: the algorithm found sits before this boundary; the slicer itself never reads
#: it (no hard-coded offsets). Mistral has no generation-prompt header (the
#: [/INST] close *is* the boundary), hence the empty marker.
GENERATION_PROMPT_MARKERS: dict[str, str] = {
    "llama-3": "<|start_header_id|>assistant<|end_header_id|>",
    "qwen": "<|im_start|>assistant",
    "gemma": "<start_of_turn>model",
    "mistral": "[/INST]",
    "phi": "<|im_start|>assistant<|im_sep|>",
}


def template_for(family: str) -> str:
    """Return the bundled chat template for ``family`` (KeyError if unknown)."""
    return CHAT_TEMPLATES[family]
