"""Reference list of official public research datasets — names and URLs ONLY.

injectkit ships ZERO dataset content. This module lists, by name and canonical
URL, the official public prompt-injection / jailbreak datasets a research user
may opt into loading (via a gated :class:`~injectkit.research.base.ResearchDatasetLoader`).
The data itself lives at each ``url`` and is downloaded from there on explicit
acknowledgment; injectkit never redistributes it. See ``docs/RESEARCH-USE.md``.

These entries are pointers and citations, not payloads. They are safe to ship.
"""

from __future__ import annotations

from .base import DatasetReference

__all__ = ["KNOWN_DATASETS"]

#: Official, public research datasets a gated loader may download on opt-in.
#: URLs are the canonical sources; injectkit does not host or bundle the data.
KNOWN_DATASETS: dict[str, DatasetReference] = {
    "advbench": DatasetReference(
        key="advbench",
        name="AdvBench (harmful behaviors / strings)",
        url="https://github.com/llm-attacks/llm-attacks",
        description=(
            "Benchmark of adversarial behaviors/strings from the GCG paper; "
            "used to measure attack-success-rate of jailbreaks. Download from the "
            "official repository."
        ),
        citation="arXiv:2307.15043 (Zou et al., 2023)",
        license_note="MIT (see source)",
        tags=["jailbreak", "harmful-behaviors", "asr"],
    ),
    "harmbench": DatasetReference(
        key="harmbench",
        name="HarmBench",
        url="https://www.harmbench.org/",
        description=(
            "Standardized evaluation framework and behavior set for automated "
            "red-teaming. Download from the official site/repository."
        ),
        citation="arXiv:2402.04249 (Mazeika et al., 2024)",
        license_note="MIT (see source)",
        tags=["red-team", "harmful-behaviors", "evaluation"],
    ),
    "jailbreakbench": DatasetReference(
        key="jailbreakbench",
        name="JailbreakBench (JBB-Behaviors)",
        url="https://jailbreakbench.github.io/",
        description=(
            "Open robustness benchmark with a behaviors dataset and leaderboard "
            "for jailbreaking LLMs. Download from the official source."
        ),
        citation="arXiv:2404.01318 (Chao et al., 2024)",
        license_note="MIT (see source)",
        tags=["jailbreak", "benchmark", "leaderboard"],
    ),
    "in_the_wild_jailbreaks": DatasetReference(
        key="in_the_wild_jailbreaks",
        name="In-The-Wild Jailbreak Prompts",
        url="https://github.com/verazuo/jailbreak_llms",
        description=(
            "Collection of real-world jailbreak prompts gathered from online "
            "communities. Download from the official repository."
        ),
        citation="arXiv:2308.03825 (Shen et al., 2023)",
        license_note="see source",
        tags=["jailbreak", "in-the-wild", "prompts"],
    ),
    "tensor_trust": DatasetReference(
        key="tensor_trust",
        name="Tensor Trust (prompt injection)",
        url="https://tensortrust.ai/paper/",
        description=(
            "Human-generated prompt-injection attacks/defenses from an online "
            "game, for studying instruction-following robustness. Download from "
            "the official source."
        ),
        citation="arXiv:2311.01011 (Toyer et al., 2023)",
        license_note="see source",
        tags=["prompt-injection", "human-generated", "defenses"],
    ),
}
