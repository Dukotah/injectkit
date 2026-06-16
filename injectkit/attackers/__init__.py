"""Adaptive attackers — propose/refine loops that optimise attack STRUCTURE.

An :class:`~injectkit.attackers.base.AdaptiveAttacker` iteratively rewrites an
attack to bypass a target's instructions, scored each round by a judge/detector
(standard automated red-teaming / ASR methodology). It is **local-model-first**:
the default attacker model is a local model (Ollama / an HF model / a stub),
needs no API key, and is lazy-imported.

ETHICS — NON-NEGOTIABLE: the attacker optimises attack STRUCTURE to defeat
instructions, measured by a benign canary proxy. It is NOT a harmful-output
generator. It only ever tries to make the target emit the benign success marker
it was told to withhold. Tests use a scripted stub model and make no network or
model calls.
"""

from __future__ import annotations

from .base import (
    AdaptiveAttacker,
    AttackerError,
    AttackerModel,
    AttackerResult,
    AttackerTranscriptStep,
)
from .adaptive import (
    OllamaAttackerModel,
    RefineAttacker,
    RefinePromptBuilder,
    ensure_canary,
    extract_payload,
)

# v0.3.0: the named-attacker registry (pair/tap/autodan/gptfuzzer/gcg) and the
# white-box GCG base (HF-only, lazy torch, benign target, stub-testable). The
# registry is pre-seeded with the five declared specs; the concrete attacker
# modules imported below register a real factory under each name at import time.
from .registry import (  # noqa: F401
    NAMED_ATTACKERS,
    AttackerRegistry,
    AttackerSpec,
    get_attacker,
    list_attackers,
    register_attacker,
)
from .whitebox_base import (  # noqa: F401
    GCGConfig,
    GCGStep,
    WhiteBoxGCGAttacker,
    WhiteBoxModel,
    import_torch_transformers,
)

# v0.3.0 concrete black-box attacker: PAIR (arXiv:2310.08419). Importing this
# module registers the `pair` factory onto the named-attacker registry above.
from .pair import (  # noqa: F401
    PAIRAttacker,
    PAIRJudge,
    PAIRPromptBuilder,
    extract_pair_prompt,
    make_pair_attacker,
)

# v0.3.0 concrete black-box attacker: TAP (arXiv:2312.02119). Importing this
# module registers the `tap` factory onto the named-attacker registry above.
from .tap import (  # noqa: F401
    TAPAttacker,
    TAPNode,
    is_on_topic,
    make_tap_attacker,
)

# v0.3.0 concrete black-box attacker: GPTFUZZER (arXiv:2309.10253). Importing this
# module registers the `gptfuzzer` factory onto the named-attacker registry above.
from .gptfuzzer import (  # noqa: F401
    GPTFuzzAttacker,
    GPTFuzzPromptBuilder,
    MutatorBank,
    SeedTemplate,
    UCBSeedScheduler,
    make_gptfuzz_attacker,
)

# v0.3.0 concrete black-box attacker: AutoDAN (arXiv:2310.04451). Importing this
# module registers the `autodan` factory onto the named-attacker registry above.
from .autodan import (  # noqa: F401
    AutoDANAttacker,
    Individual,
    ModelMutator,
    MutationOperator,
    OfflineMutator,
    register_autodan,
)

# v0.3.0 concrete WHITE-BOX attacker: GCG / AmpleGCG (arXiv:2404.07921). HF-only,
# lazy torch, benign-canary target. Importing this module registers the `gcg`
# factory onto the named-attacker registry above.
from .gcg import (  # noqa: F401
    DEFAULT_INIT_SUFFIX,
    GCGSuffixAttacker,
    load_amplegcg_suffixes,
    make_gcg_attacker,
)

__all__ = [
    "AttackerModel",
    "AdaptiveAttacker",
    "AttackerResult",
    "AttackerTranscriptStep",
    "AttackerError",
    "RefineAttacker",
    "RefinePromptBuilder",
    "OllamaAttackerModel",
    "ensure_canary",
    "extract_payload",
    # v0.3.0 named-attacker registry
    "AttackerSpec",
    "AttackerRegistry",
    "NAMED_ATTACKERS",
    "register_attacker",
    "get_attacker",
    "list_attackers",
    # v0.3.0 white-box GCG base
    "WhiteBoxModel",
    "WhiteBoxGCGAttacker",
    "GCGConfig",
    "GCGStep",
    "import_torch_transformers",
    # v0.3.0 PAIR attacker
    "PAIRAttacker",
    "PAIRJudge",
    "PAIRPromptBuilder",
    "extract_pair_prompt",
    "make_pair_attacker",
    # v0.3.0 TAP attacker
    "TAPAttacker",
    "TAPNode",
    "is_on_topic",
    "make_tap_attacker",
    # v0.3.0 GPTFUZZER attacker
    "GPTFuzzAttacker",
    "GPTFuzzPromptBuilder",
    "MutatorBank",
    "SeedTemplate",
    "UCBSeedScheduler",
    "make_gptfuzz_attacker",
    # v0.3.0 AutoDAN genetic attacker
    "AutoDANAttacker",
    "Individual",
    "MutationOperator",
    "OfflineMutator",
    "ModelMutator",
    "register_autodan",
    # v0.3.0 GCG / AmpleGCG white-box attacker
    "GCGSuffixAttacker",
    "DEFAULT_INIT_SUFFIX",
    "load_amplegcg_suffixes",
    "make_gcg_attacker",
]
