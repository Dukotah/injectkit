"""Typed Pydantic configs for the v0.4 white-box attack interface.

This is the ``config/base.py`` of the v0.4 reposition (ROADMAP §4, §5, §6.0),
realised as ``injectkit/whitebox/config.py`` because the shipped v0.3 package is
**flat-layout** and already has a top-level ``injectkit/config.py`` module (the
run config), which a ``config/`` package would shadow. Per the ROADMAP's own
"if flat-layout, rewrite every path to match — extend, don't rebuild" rule, the
typed-config *intent* lives here, alongside the attack interface it parameterises.

Decision of record (ROADMAP §4 tech-stack table — "was open"): typed configs are
**Pydantic v2 settings models over plain YAML registries**. Rationale captured in
the ROADMAP: validation + env-override + JSON-schema export with one dependency
injectkit already assumes (``pydantic>=2`` is a hard dependency); Hydra's
compositional sweeps are not needed and would add a heavy dep.

Every white-box attack takes an :class:`AttackConfig` subclass. The base carries
the cross-attack knobs (seed, benign trigger, step budget); each family adds its
own typed fields (e.g. :class:`GCGConfig` adds ``suffix_len``/``top_k``/...).

ETHICS: the optimisation objective is **always** the per-run BENIGN canary marker
``<trigger><canary>`` — a robustness probe — never a harmful string. ``target``
defaults to ``None`` so the attack builds the benign marker itself; a non-default
``target`` is an explicit, documented baseline override only.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..evaluators.heuristics import DEFAULT_TRIGGER

__all__ = [
    "AttackConfig",
    "GCGConfig",
    "PrefillConfig",
    "ReinforceGCGConfig",
    "UJAConfig",
]


class AttackConfig(BaseModel):
    """Base typed config shared by every white-box :class:`~injectkit.whitebox.base.Attack`.

    A frozen, validated Pydantic v2 model. The defaults are SMALL and SAFE so the
    offline test path runs fast; real runs raise the budgets (and want a GPU).

    Cross-attack knobs only — attack-specific knobs (suffix length, top-k, ...)
    live on the per-family subclass. ``env_prefix`` is not set here so plain
    instantiation never reads the environment; registries that want env-override
    can subclass with ``pydantic_settings`` later (ROADMAP §4) without changing
    this contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hard cap on optimisation steps so every run terminates. Tests pass 1.
    max_steps: int = Field(default=50, ge=1)
    #: The string the attack optimises the model to emit. ``None`` (default) ⇒ the
    #: attack builds the BENIGN per-run marker ``<trigger><canary>``. A non-None
    #: value is an explicit, documented baseline override only — NEVER harmful.
    target: str | None = None
    #: Benign success-marker prefix used to build the benign target string.
    trigger: str = DEFAULT_TRIGGER
    #: RNG seed for reproducible candidate sampling (threaded everywhere).
    seed: int = 0


class GCGConfig(AttackConfig):
    """Typed config for the GCG family (ROADMAP §6.1; Zou et al. arXiv:2307.15043).

    Adds the greedy-coordinate-gradient knobs on top of :class:`AttackConfig`.
    Field names mirror the v0.3 dataclass
    :class:`injectkit.attackers.whitebox_base.GCGConfig` so the re-wrap maps one
    field to one field; :meth:`to_legacy` projects this typed model onto that
    dataclass for the existing optimiser.
    """

    #: Length (in tokens) of the adversarial suffix being optimised.
    suffix_len: int = Field(default=20, ge=1)
    #: Per-step number of candidate token swaps evaluated (the GCG batch).
    batch_size: int = Field(default=64, ge=1)
    #: Top-k most-promising replacement tokens drawn from the gradient per slot
    #: (nanoGCG default 256).
    top_k: int = Field(default=256, ge=1)
    #: Number of candidate single-token swaps scored per step in one batched
    #: forward pass (nanoGCG's ``search_width``; default 512). Distinct from
    #: ``batch_size``, which is the v0.3 per-slot sampler's budget.
    search_width: int = Field(default=512, ge=1)
    #: Best-suffix attack-buffer size (nanoGCG ``buffer_size``). 0 ⇒ keep only the
    #: single best candidate; >0 ⇒ optimise from a pool to escape plateaus.
    buffer_size: int = Field(default=0, ge=0)
    #: Drop candidate suffixes that fail the tokenizer round-trip
    #: ``encode(decode(ids)) == ids`` (the filter_ids correctness trap). Keep True
    #: for correctness; exposed only so the trap itself can be tested both ways.
    filter_nonascii: bool = True
    #: nanoGCG ``probe_sampling`` toggle (draft-model candidate pre-filtering).
    #: Off by default; the draft-model loop is a GPU deliverable (DEFERRED-NO-GPU).
    probe_sampling: bool = False
    #: When True (default), GCG's target prefix is the model-specific AdvPrefix
    #: (arXiv:2412.10321); when False, the documented fixed "Sure, here is"
    #: baseline. See :func:`injectkit.whitebox.targets.advprefix_target`.
    use_advprefix: bool = True
    #: Initial benign filler suffix the optimiser starts from. ``None`` ⇒ the
    #: attacker's default benign filler. May be seeded from a research-gated
    #: AmpleGCG artifact, but the objective is re-pointed at the benign marker.
    init_suffix: str | None = None

    def to_legacy(self) -> Any:
        """Project onto the v0.3 :class:`GCGConfig` dataclass the optimiser reads.

        Returns an :class:`injectkit.attackers.whitebox_base.GCGConfig` (lazily
        imported to keep this module free of attacker-side imports at load). The
        v0.4 typed model is the public surface; the existing greedy-coordinate-
        gradient optimiser still consumes the v0.3 dataclass, so the re-wrap
        converts here rather than rewriting the proven inner loop.
        """
        from ..attackers.whitebox_base import GCGConfig as LegacyGCGConfig

        return LegacyGCGConfig(
            max_steps=self.max_steps,
            suffix_len=self.suffix_len,
            batch_size=self.batch_size,
            top_k=self.top_k,
            target_string=self.target,
            init_suffix=self.init_suffix,
            trigger=self.trigger,
            seed=self.seed,
        )


#: The default in-loop OPTIMISATION judge id (ROADMAP §6.10.1 circularity
#: firewall). It MUST differ from the EVAL judge the leaderboard scores with, both
#: to avoid the optimiser gaming its own grader (arXiv:2502.11910) and to fit the
#: 24 GB VRAM budget. The substring matcher is bundled, fast, never the eval judge,
#: and a safe in-loop default for the offline test path. Mirrors
#: ``injectkit.judge.DEFAULT_OPT_JUDGE`` (re-declared here so the config module
#: stays free of a judge-layer import at load time).
DEFAULT_OPT_JUDGE_ID = "substring"


class ReinforceGCGConfig(GCGConfig):
    """Typed config for REINFORCE-GCG (judge-in-the-loop GCG; arXiv:2502.17924).

    REINFORCE-GCG (Geisler et al., *"REINFORCE Adversarial Attacks on Large
    Language Models: An Adaptive, Distributional, and Semantic Objective"*,
    arXiv:2502.17924) replaces GCG's single teacher-forced target-NLL objective
    with a **REINFORCE reward** estimated by sampling the model's own
    continuations and scoring them with an in-loop judge. The gradient signal that
    selects candidate token swaps is then ``loss = nll - reward_weight * reward``,
    so candidates that actually *make the model emit the benign marker* (as judged)
    are preferred over candidates that merely lower the teacher-forced NLL of a
    fixed prefix — the distributional/semantic objective the paper introduces.

    Inherits every GCG knob (``suffix_len``/``top_k``/``search_width``/...) and
    adds the REINFORCE-specific ones. The in-loop ``opt_judge_id`` defaults to a
    judge DISTINCT from the leaderboard EVAL judge (``clean_cls``), preserving the
    §6.10.1 circularity firewall (a later test asserts ``opt_judge_id !=
    eval_judge_id``).

    ETHICS: the reward judges the BENIGN-canary proxy — "reward" means the model
    emitted the per-run marker it was told to withhold, never harmful content.
    """

    #: Registry id of the in-loop OPTIMISATION judge whose reward steers candidate
    #: selection. MUST differ from the EVAL judge (§6.10.1). Bundled + offline.
    opt_judge_id: str = DEFAULT_OPT_JUDGE_ID
    #: Weight λ on the judge reward in the combined objective
    #: ``nll - reward_weight * reward``. 0 ⇒ plain GCG (reward ignored); larger ⇒
    #: the semantic judge signal dominates the teacher-forced NLL.
    reward_weight: float = Field(default=1.0, ge=0.0)
    #: Number of continuations sampled per candidate to estimate the REINFORCE
    #: reward (the paper's distributional objective averages over samples). 1 keeps
    #: the offline path cheap; real runs raise it.
    num_samples: int = Field(default=4, ge=1)
    #: Greedy/sampled continuation length (tokens) the in-loop judge scores.
    judge_n_tokens: int = Field(default=64, ge=1)


class UJAConfig(GCGConfig):
    """Typed config for UJA — Universal Jailbreak Adversarial (judge-in-the-loop).

    UJA optimises ONE *universal* adversarial suffix that transfers across a SET of
    behaviors at once (the universal/transferable GCG objective; Zou et al.,
    arXiv:2307.15043 §"Universal", extended with an in-loop judge reward as in the
    universal-jailbreak-adversarial line of work). Each optimisation step scores
    candidate swaps by the **mean** in-loop judge reward across the behavior batch,
    so the surviving suffix is the one that drives the benign marker on the *most*
    behaviors — a single suffix, many prompts.

    Inherits the GCG knobs and adds the universal-batch + in-loop-judge knobs. The
    behaviors themselves are passed at ``run`` time (in ``messages`` / a behavior
    set), not on the config. The in-loop ``opt_judge_id`` again defaults DISTINCT
    from the EVAL judge (§6.10.1).

    ETHICS: every behavior is a BENIGN-canary robustness probe; the universal suffix
    is optimised to elicit the per-run marker across them, never harmful content.
    """

    #: Registry id of the in-loop OPTIMISATION judge whose mean reward across the
    #: behavior batch steers candidate selection. MUST differ from the EVAL judge.
    opt_judge_id: str = DEFAULT_OPT_JUDGE_ID
    #: Weight λ on the judge reward in the combined universal objective.
    reward_weight: float = Field(default=1.0, ge=0.0)
    #: Max number of behaviors averaged per step (the universal batch). Caps the
    #: per-step cost; the suffix is still scored on all behaviors for the result.
    behaviors_per_step: int = Field(default=4, ge=1)
    #: Greedy continuation length (tokens) the in-loop judge scores per behavior.
    judge_n_tokens: int = Field(default=64, ge=1)


class PrefillConfig(AttackConfig):
    """Typed config for the prefill attack family (arXiv:2602.14689).

    Assistant-turn prefilling is a **one-shot** attack, not an optimisation loop:
    the attacker writes the opening of the assistant's reply (a benign-seeming
    prefix that pre-commits the model past its refusal point), the model greedily
    completes it, and a judge grades the continuation. There is no gradient and no
    per-step trajectory — so the GCG knobs (``suffix_len``/``top_k``/...) do not
    apply and ``max_steps`` is effectively the number of candidate prefixes tried.

    Adds the prefill-specific knobs on top of :class:`AttackConfig`:

    * :attr:`candidate_prefixes` — the explicit prefix inventory to try. ``None``
      (default) ⇒ the attack picks the model-specific bundled inventory for the
      target's family (Llama-3/Qwen/Mistral/Gemma/Phi, or the GPT-OSS harmony
      path), always including the generic benign prefix.
    * :attr:`n_tokens` — greedy continuation length to generate and judge (the
      paper's N=512 default).
    * :attr:`use_target` — when False (default), success is judged on the model's
      *generated continuation* alone (the realistic prefill condition: the prefix
      is the assistant's own words). When True, the benign target marker is also
      appended to the prefix as an explicit baseline override.
    """

    #: Explicit prefix inventory to try. ``None`` ⇒ model-specific bundled
    #: inventory chosen by family at run time (always incl. the generic prefix).
    candidate_prefixes: tuple[str, ...] | None = None
    #: Greedy continuation length generated and judged per candidate (paper N=512).
    n_tokens: int = Field(default=512, ge=1)
    #: When True, also append the benign target marker to the prefix (explicit
    #: baseline override). Default False — judge the model's own continuation.
    use_target: bool = False
