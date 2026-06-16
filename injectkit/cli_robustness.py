"""CLI wiring for the v0.2.0 robustness axes and the research benchmark.

This module owns the *glue* between the parsed CLI flags and the v0.2.0
machinery so :mod:`injectkit.cli` stays a thin dispatcher. It resolves the
``--mutate`` / ``--defense`` / ``--multiturn`` / ``--adaptive`` flags into the
concrete :class:`~injectkit.transforms.base.Transform`,
:class:`~injectkit.defenses.base.Defense`,
:class:`~injectkit.attacks.base.AttackStrategy` and
:class:`~injectkit.attackers.base.AdaptiveAttacker` objects, wraps the target for
multi-turn delivery, runs the ``bench`` ASR scorecard, and enforces the GATED
``--research-benchmark`` opt-in.

Every helper:

* lazy-imports its optional dependency (transforms/defenses/attacker backends are
  imported only when the corresponding flag is used), and
* raises :class:`~injectkit.engine.ScanError` with a friendly, actionable message
  when a name is unknown or an optional dependency is missing — so the CLI can
  print one consistent error and exit ``2`` without a traceback.

DEFENSIVE / AUTHORIZED USE ONLY. The robustness sweep and the research benchmark
both measure the benign-canary proxy — "did the injection make the target emit
the benign marker it was told to withhold?" — never whether harmful content was
produced. The research loaders ship NO data and download from the dataset's own
source only after an explicit acknowledgment.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .engine import ScanError
from .models import Attack, TargetResponse

__all__ = [
    "BuiltRobustness",
    "build_transforms",
    "build_defenses",
    "build_strategy_for",
    "build_attacker",
    "wrap_target_for_multiturn",
    "resolve_robustness",
    "load_research_attacks",
    "run_bench",
]


# --------------------------------------------------------------------------- #
# Transforms (--mutate)
# --------------------------------------------------------------------------- #
def build_transforms(spec: Optional[str], *, seed: Optional[int] = None) -> list:
    """Resolve a ``--mutate`` spec into a list of Transform instances.

    Args:
        spec: The raw ``--mutate`` value: a comma-separated list of transform
            names (applied in order via a single composed transform), the special
            value ``"all"`` (every registered transform as its own variant), or
            ``None``/empty (no transforms — identity baseline only).
        seed: Optional RNG seed forwarded to seeded transform factories (ignored
            by deterministic transforms).

    Returns:
        A list of transform instances to sweep. Empty when ``spec`` is falsy
        (the runner always adds the identity baseline itself).

    Raises:
        ScanError: if a named transform is unknown (the message lists the
            available names).
    """
    if not spec or not spec.strip():
        return []

    # Importing the encoders module registers every built-in transform factory on
    # the process-wide registry; do it lazily so `list`/`init` stay light.
    from .transforms import base as tbase
    from .transforms import encoders as _encoders  # noqa: F401 - registers factories

    available = tbase.list_transforms()
    parts = [p.strip() for p in spec.split(",") if p.strip()]

    if len(parts) == 1 and parts[0].lower() == "all":
        # Every registered transform as a separate variant (skip identity — the
        # runner adds it as the baseline).
        names = [n for n in available if n != "identity"]
        return [_get_transform(n, available) for n in names]

    # A comma-separated list is applied in ORDER as one composed transform, so
    # `--mutate rot13,zero_width` measures the stacked obfuscation.
    transforms = [_get_transform(n, available) for n in parts]
    if len(transforms) == 1:
        return transforms
    return [tbase.Compose(*transforms)]


def _get_transform(name: str, available: Sequence[str]) -> object:
    """Resolve one transform name, raising a friendly ScanError if unknown."""
    from .transforms import base as tbase

    try:
        return tbase.get_transform(name)
    except KeyError as exc:  # pragma: no cover - exercised via build_transforms
        raise ScanError(
            f"unknown transform {name!r}. Available: {', '.join(available)}."
        ) from exc


# --------------------------------------------------------------------------- #
# Defenses (--defense)
# --------------------------------------------------------------------------- #
def build_defenses(spec: Optional[str]) -> list:
    """Resolve a ``--defense`` spec into a list of Defense instances.

    Args:
        spec: The raw ``--defense`` value: a single defense name, a
            comma-separated list of defenses to sweep, the special value
            ``"all"`` (every registered defense), or ``None``/empty (no defense —
            the ``none`` baseline only).

    Returns:
        A list of defense instances to sweep. Empty when ``spec`` is falsy (the
        runner always adds the ``none`` baseline itself).

    Raises:
        ScanError: if a named defense is unknown (the message lists the
            available names).
    """
    if not spec or not spec.strip():
        return []

    from .defenses import base as dbase
    from .defenses import mitigations as _mit  # noqa: F401 - registers factories

    available = dbase.list_defenses()
    parts = [p.strip() for p in spec.split(",") if p.strip()]

    if len(parts) == 1 and parts[0].lower() == "all":
        return [_get_defense(n, available) for n in available if n != "none"]

    return [_get_defense(n, available) for n in parts]


def _get_defense(name: str, available: Sequence[str]) -> object:
    """Resolve one defense name, raising a friendly ScanError if unknown."""
    from .defenses import base as dbase

    try:
        return dbase.get_defense(name)
    except KeyError as exc:  # pragma: no cover - exercised via build_defenses
        raise ScanError(
            f"unknown defense {name!r}. Available: {', '.join(available)}."
        ) from exc


# --------------------------------------------------------------------------- #
# Multi-turn strategy (--multiturn)
# --------------------------------------------------------------------------- #
def build_strategy_for(name: Optional[str]):
    """Resolve a ``--multiturn`` strategy name into an AttackStrategy.

    Args:
        name: A multi-turn strategy name (``crescendo`` | ``many_shot`` |
            ``context_overflow`` | ``persona_priming``), or ``None`` for the
            default single-shot delivery.

    Returns:
        An :class:`~injectkit.attacks.base.AttackStrategy`, or ``None`` when no
        multi-turn strategy was requested.

    Raises:
        ScanError: if ``name`` is not a known multi-turn strategy.
    """
    if not name:
        return None
    from .attacks.multiturn import StrategyError, build_strategy

    try:
        return build_strategy(name)
    except StrategyError as exc:
        raise ScanError(f"multi-turn: {exc}") from exc


def wrap_target_for_multiturn(target: object, strategy: object, *, trigger: str = "INJECTOK-"):
    """Wrap a single-shot target so each attack is delivered via ``strategy``.

    The engine still drives one canary-rendered prompt per attack and scores the
    response with the existing detectors. This wrapper intercepts that prompt,
    recovers the per-run canary the engine planted (by the trigger marker), asks
    the strategy to build the multi-turn :class:`AttackStep` list for that
    canary, delivers those turns through a
    :class:`~injectkit.targets.conversational.ConversationalTarget` view of the
    target, and returns the *scored* step's response. Earlier turns are delivered
    as conversation context but their responses are discarded.

    Because the canary is recovered from the engine-rendered prompt, the
    benign-canary proxy is preserved end to end: the detector hunts for exactly
    the marker the engine planted, and a multi-turn success is a real marker echo.

    Args:
        target: The single-shot (or already-conversational) target to wrap.
        strategy: The :class:`AttackStrategy` whose turns are delivered.
        trigger: The success-marker prefix used to recover the canary.

    Returns:
        A target-shaped object exposing ``send`` and ``name``.
    """
    from .attacks.base import AttackStep
    from .targets.conversational import ChatMessage, as_conversational

    conv = as_conversational(target)
    marker_re = re.compile(re.escape(trigger) + r"([A-Za-z0-9_-]+)")

    class _MultiTurnTarget:
        name = getattr(target, "name", "target")

        def _canary(self, prompt: str, context: Optional[str]) -> str:
            for text in (prompt, context or ""):
                m = marker_re.search(text)
                if m:
                    return m.group(1)
            return ""

        def send(
            self,
            prompt: str,
            system: Optional[str] = None,
            context: Optional[str] = None,
        ) -> TargetResponse:
            canary = self._canary(prompt, context)
            # Build a single-shot Attack carrying the already-rendered prompt so a
            # multi-turn strategy that reads attack.payload sees the engine's
            # canary; strategies render {canary} into their own turns with the
            # recovered canary below.
            seed = Attack(
                id="multiturn",
                technique="multiturn",
                name="multiturn",
                description="",
                severity=_default_severity(),
                payload=prompt,
                system=system,
            )
            try:
                steps = strategy.build(seed, canary)
            except Exception as exc:  # noqa: BLE001 - a strategy fault degrades to single-shot
                return target.send(prompt, system=system, context=context)

            messages: list[ChatMessage] = []
            scored_response: Optional[TargetResponse] = None
            last_response: Optional[TargetResponse] = None
            for step in steps:
                if not isinstance(step, AttackStep):
                    continue
                messages.append(step.message)
                if not step.expect_response:
                    # A scripted assistant/history turn — do not call the target.
                    continue
                resp = conv.chat(messages, system=system)
                last_response = resp
                if isinstance(resp, TargetResponse) and resp.error is None:
                    # Append the assistant reply so later turns see the history.
                    messages.append(ChatMessage(role="assistant", content=resp.text))
                if step.scored:
                    scored_response = resp
            result = scored_response if scored_response is not None else last_response
            if isinstance(result, TargetResponse):
                return result
            # No turn produced a response (degenerate strategy); fall back.
            return target.send(prompt, system=system, context=context)

    return _MultiTurnTarget()


def _default_severity():
    """The severity stamped on the internal multi-turn seed attack."""
    from .models import Severity

    return Severity.HIGH


# --------------------------------------------------------------------------- #
# Adaptive attacker (--adaptive)
# --------------------------------------------------------------------------- #
def build_attacker(
    *,
    backend: str = "ollama",
    model: Optional[str] = None,
    max_rounds: int = 5,
    seed: Optional[int] = None,
    detectors: Optional[Sequence[object]] = None,
    use_judge: bool = False,
):
    """Construct the local-model adaptive attacker for ``--adaptive``.

    Local-model-first: the only built-in backend is a local Ollama server (no API
    key). The attacker optimizes attack STRUCTURE against the benign-canary
    objective scored by the supplied detectors — it is not a harmful-content
    generator.

    Args:
        backend: The attacker model backend (currently only ``"ollama"``).
        model: The attacker model id (defaults to the backend default).
        max_rounds: Hard round budget (must be >= 1).
        seed: Reproducibility seed (recorded; the refine loop is deterministic).
        detectors: Detectors used to score each round (defaults to heuristics).
        use_judge: Whether the judge participates in round scoring.

    Returns:
        An :class:`~injectkit.attackers.base.AdaptiveAttacker`.

    Raises:
        ScanError: on an unknown backend or an invalid round budget.
    """
    if backend != "ollama":
        raise ScanError(
            f"unknown adaptive attacker backend {backend!r}; supported: ollama."
        )
    if max_rounds < 1:
        raise ScanError("--max-rounds must be >= 1.")

    from .attackers.adaptive import OllamaAttackerModel, RefineAttacker

    attacker_model = OllamaAttackerModel(model=model or "llama3.1")
    return RefineAttacker(
        attacker_model,
        max_rounds=max_rounds,
        detectors=list(detectors) if detectors else None,
        use_judge=use_judge,
    )


# --------------------------------------------------------------------------- #
# Aggregate resolver used by both scan and bench
# --------------------------------------------------------------------------- #
class BuiltRobustness:
    """The resolved robustness objects for one scan/bench invocation.

    Holds the transform variants, defense variants, optional multi-turn strategy,
    and optional adaptive attacker that the CLI flags produced, so the scan and
    bench handlers can apply them without re-resolving names.
    """

    def __init__(
        self,
        *,
        transforms: list,
        defenses: list,
        strategy: object,
        attacker: object,
        seed: Optional[int],
    ) -> None:
        self.transforms = transforms
        self.defenses = defenses
        self.strategy = strategy
        self.attacker = attacker
        self.seed = seed


def resolve_robustness(args, *, detectors: Optional[Sequence[object]] = None) -> BuiltRobustness:
    """Resolve every robustness flag on ``args`` into a :class:`BuiltRobustness`.

    Args:
        args: The parsed argparse namespace (carries ``mutate``, ``defense``,
            ``multiturn``, ``adaptive``, ``attacker_target``, ``attacker_model``,
            ``max_rounds`` and ``seed``).
        detectors: Detectors to hand the adaptive attacker (so its rounds score
            the same way the scan does).

    Returns:
        A populated :class:`BuiltRobustness`.

    Raises:
        ScanError: on any unknown transform/defense/strategy or attacker setup
            problem (one consistent, friendly error type).
    """
    seed = getattr(args, "seed", None)
    transforms = build_transforms(getattr(args, "mutate", None), seed=seed)
    defenses = build_defenses(getattr(args, "defense", None))
    strategy = build_strategy_for(getattr(args, "multiturn", None))

    attacker = None
    if getattr(args, "adaptive", False):
        attacker = build_attacker(
            backend=getattr(args, "attacker_target", "ollama"),
            model=getattr(args, "attacker_model", None),
            max_rounds=getattr(args, "max_rounds", 5),
            seed=seed,
            detectors=detectors,
            use_judge=bool(getattr(args, "judge", False)),
        )

    return BuiltRobustness(
        transforms=transforms,
        defenses=defenses,
        strategy=strategy,
        attacker=attacker,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Research benchmark (--research-benchmark, GATED)
# --------------------------------------------------------------------------- #
def load_research_attacks(
    dataset: str,
    *,
    acknowledge: bool,
    limit: Optional[int] = None,
) -> list[Attack]:
    """Load a GATED research dataset as benign-canary-proxied attacks.

    Routes through the loader's opt-in gate (which also honours the
    ``INJECTKIT_RESEARCH_ACK`` env var), so without ``acknowledge=True`` nothing
    is downloaded — a :class:`ResearchAcknowledgmentError` carrying the
    research-use disclaimer is raised instead. Behaviors are wrapped in the
    benign-canary proxy so success is a benign marker echo, never harmful output.

    Args:
        dataset: A known dataset key (advbench | harmbench | jailbreakbench |
            in_the_wild_jailbreaks | tensor_trust).
        acknowledge: The value of ``--i-am-authorized`` — the explicit opt-in.
        limit: Cap on the number of behaviors loaded.

    Returns:
        A list of benign-canary-proxied :class:`Attack` objects.

    Raises:
        ScanError: on an unknown dataset key.
        ResearchAcknowledgmentError: if the opt-in gate is not satisfied.
        ResearchDownloadError: on a download/parse failure after the gate.
    """
    from .research.datasets import available_datasets, get_loader

    try:
        loader = get_loader(dataset)
    except KeyError as exc:  # pragma: no cover - message exercised in tests
        raise ScanError(
            f"unknown research dataset {dataset!r}. Available: "
            f"{', '.join(available_datasets())}."
        ) from exc

    return loader.load(acknowledge=acknowledge, limit=limit, proxy="canary")


def build_bench_reporter(fmt: str):
    """Construct the scorecard reporter for the ``bench`` ``--format`` value."""
    fmt = (fmt or "terminal").lower()
    from .reporters import scorecard as sc

    if fmt == "terminal":
        return sc.ScorecardTerminalReporter()
    if fmt == "json":
        return sc.ScorecardJSONReporter()
    if fmt == "markdown":
        return sc.ScorecardMarkdownReporter()
    if fmt == "html":
        return sc.ScorecardHtmlReporter()
    raise ScanError(
        f"unknown bench format {fmt!r}. Choose one of: terminal, json, markdown, html."
    )


def run_bench(
    *,
    target: object,
    detectors: Sequence[object],
    attacks: Sequence[Attack],
    robustness: BuiltRobustness,
    use_judge: bool,
    tool_version: str,
):
    """Run the ASR benchmark sweep and return a BenchmarkResult.

    Builds a :class:`~injectkit.benchmark_runner.BenchmarkRunner` from the
    resolved robustness axes (transforms, defenses, optional adaptive attacker)
    and runs the corpus into the per-technique/per-defense scorecard. A
    multi-turn strategy, when configured, wraps the target so every attack is
    delivered as a conversation before the sweep.

    Args:
        target: The target under test.
        detectors: Detectors to score each attack.
        attacks: The corpus (or research dataset) to benchmark.
        robustness: The resolved transforms/defenses/strategy/attacker.
        use_judge: Whether the judge participates in scoring.
        tool_version: Version stamped on the run metadata.

    Returns:
        A populated :class:`~injectkit.benchmark.BenchmarkResult`.
    """
    from .benchmark_runner import BenchmarkRunner

    effective_target = target
    if robustness.strategy is not None:
        effective_target = wrap_target_for_multiturn(target, robustness.strategy)

    runner = BenchmarkRunner(
        effective_target,
        detectors,
        transforms=robustness.transforms or None,
        defenses=robustness.defenses or None,
        attacker=robustness.attacker,
        use_judge=use_judge,
        seed=robustness.seed,
        tool_version=tool_version,
    )
    return runner.run(attacks)
