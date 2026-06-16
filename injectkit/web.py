"""injectkit local web GUI.

A tiny, dependency-free browser front end for injectkit so you can interact with
a scan without the command line. It reuses the exact same engine, corpus, target
adapters, detectors, reporters, transforms, defenses, and benchmark runner the
CLI uses — this is a thin web layer, not a second implementation.

Run it with::

    python -m injectkit.web            # opens http://127.0.0.1:8765 in your browser
    python -m injectkit.web --port 9000 --no-open

Then pick a target (the offline ``mock`` target needs no API key and no network,
and the new ``ollama`` / ``openai`` / ``hf`` targets drive a model on your own
machine), choose which attack techniques to run, optionally turn on obfuscation
transforms (``--mutate``), a mitigation ``defense``, multi-turn delivery, or the
adaptive attacker, set the CI fail-on threshold, and click *Run scan*. The full
HTML report renders right in the page.

Switch the *Mode* to **Benchmark** to sweep the corpus across the selected
transforms and defenses and render the ASR robustness scorecard instead of a
single scan report.

The optional **research benchmark** stays GATED behind an explicit acknowledgment
checkbox plus the research-use disclaimer, exactly like the CLI's
``--research-benchmark`` / ``--i-am-authorized`` pairing — it never downloads
anything unless you tick the box.

v0.3.0 additions this layer surfaces (each cited in ``docs/RESEARCH.md``):

* the cipher / art-prompt / self-cipher transforms (CipherChat 2308.06463,
  ArtPrompt 2402.11753) and the semantic low-resource-language ``translate``
  transform (2310.02446 / MultiJail 2310.06474) appear in the *mutate* selector.
  ``translate`` carries a friendly note when its optional offline translator
  (argostranslate) is absent, exactly like the CLI;
* the named adaptive attackers — ``pair`` / ``tap`` / ``autodan`` / ``gptfuzzer``
  (black-box) and ``gcg`` (white-box gradient suffix; AmpleGCG 2404.07921) — are
  listed in an attacker dropdown for discoverability, with ``gcg`` noting it
  needs torch/transformers + a HuggingFace target. Like the v0.2.0 adaptive
  toggle these stay CLI-only to drive (the GUI exposes no attacker-model fields),
  so the GUI never blocks on a missing local model;
* the five-class response framework (SoK Prompt Hacking 2410.13901 / StrongREJECT)
  is surfaced as a per-scan breakdown via
  :func:`~injectkit.evaluators.response_class.classify_result`, alongside the
  existing boolean pass/fail.

DEFENSIVE / AUTHORIZED USE ONLY. Only scan endpoints you own or are explicitly
authorized to test. The server binds to localhost only.
"""

from __future__ import annotations

import argparse
import html
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs

from . import __version__
from .benchmark import BenchmarkResult
from .config import load_config
from .engine import Engine
from .models import ScanReport

# Reuse the CLI's builders so the GUI and CLI behave identically.
from .cli import _build_detectors, _build_reporter, _build_target, _load_attacks

TECHNIQUES = [
    "direct_injection",
    "indirect_injection",
    "jailbreak",
    "system_prompt_leak",
    "tool_abuse",
    "data_exfiltration",
]

# v0.2.0 adds three local (no-API-key) model adapters alongside the originals.
# Keep this in sync with cli._TARGET_KINDS (mock first so it is the default).
TARGET_KINDS = ["mock", "ollama", "openai", "hf", "http", "anthropic", "mcp"]

FAIL_ON = ["info", "low", "medium", "high", "critical"]

# The two run modes the GUI offers.
MODES = ["scan", "benchmark"]

# Friendly one-liners shown under the target dropdown so a first-time user knows
# which kinds need a key/network and which run fully offline.
_TARGET_HELP = {
    "mock": "built-in vulnerable demo target (no key, no network)",
    "ollama": "a model on your local <b>ollama serve</b> (no API key)",
    "openai": "an OpenAI-compatible local server (vLLM / LM Studio; no key)",
    "hf": "an in-process HuggingFace transformers model (loads locally)",
    "http": "your own endpoint URL",
    "anthropic": "a Claude model (needs ANTHROPIC_API_KEY)",
    "mcp": "a Model Context Protocol server you run",
}


def _dep_available(module: str) -> bool:
    """True if an optional dependency ``module`` can be imported (no side effects).

    Used to decide whether to show a friendly "needs <dep>" note next to the
    transforms/attackers that lazy-import a heavy optional dependency
    (``argostranslate`` for ``translate``, ``torch``/``transformers`` for ``gcg``).
    Checking the spec never imports the module, so this stays offline and cheap.
    """
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001 - a broken finder must never crash the GUI
        return False


def _register_v3_transforms() -> None:
    """Idempotently register the v0.3.0 cipher / art / translate transforms.

    The cipher and translate transforms are NOT auto-registered at import (unlike
    the encoders), so the GUI registers them here before listing — exactly the
    integrator seam ``register_builtin_ciphers()`` / ``register_translate()``
    document. Both calls are idempotent (a name already present is skipped), so
    calling this on every form render is safe. Registration imports nothing heavy:
    the translator's argostranslate dep stays lazy until a translate run.
    """
    try:
        from .transforms import register_builtin_ciphers, register_translate

        register_builtin_ciphers()
        register_translate()
    except Exception:  # noqa: BLE001 - a registry hiccup must not blank the form
        pass


def _list_transforms() -> list[str]:
    """Transform names available for the mutate selector (without the no-op).

    Ensures the v0.3.0 cipher / art / translate transforms are registered first
    so they appear alongside the v0.2.0 encoders.
    """
    _register_v3_transforms()
    try:
        from .transforms.base import list_transforms

        return [t for t in list_transforms() if t != "identity"]
    except Exception:  # noqa: BLE001 - the GUI must still render if a registry is empty
        return []


#: Transforms whose optional dependency must be present to actually run, mapped to
#: the importable module that backs them. Shown with a friendly note in the form
#: when the dep is absent (the transform still lists, like the CLI's ``--mutate``).
_TRANSFORM_OPTIONAL_DEPS: dict[str, str] = {"translate": "argostranslate"}


def _list_attackers() -> list[tuple[str, str, bool]]:
    """(name, kind, runnable) triples for the named adaptive-attacker dropdown.

    ``runnable`` is False for an attacker whose optional dependency is missing
    (``gcg`` needs ``torch``/``transformers``) so the form can annotate it. The
    list is informational: the GUI exposes no attacker-model fields, so driving an
    attacker stays a CLI flow (``injectkit scan --attacker <name>``), mirroring the
    v0.2.0 adaptive toggle. A missing/empty registry degrades to an empty list.
    """
    try:
        from .attackers.registry import list_attackers, registry
    except Exception:  # noqa: BLE001
        return []

    out: list[tuple[str, str, bool]] = []
    for name in list_attackers():
        try:
            spec = registry.spec(name)
            kind = spec.kind
        except Exception:  # noqa: BLE001
            kind = "black_box"
        runnable = True
        if name == "gcg":
            # white-box GCG needs torch + transformers AND a HuggingFace target.
            runnable = _dep_available("torch") and _dep_available("transformers")
        out.append((name, kind, runnable))
    return out


def _list_defenses() -> list[str]:
    """Defense names available for the defense selector ("none" first)."""
    try:
        from .defenses.base import list_defenses

        names = list(list_defenses())
    except Exception:  # noqa: BLE001
        names = ["none"]
    names = sorted(n for n in names if n != "none")
    return ["none", *names]


def _list_multiturn_strategies() -> list[str]:
    """Multi-turn strategy names for the multiturn selector."""
    try:
        from .attacks.multiturn import MULTI_TURN_STRATEGIES

        return list(MULTI_TURN_STRATEGIES.keys())
    except Exception:  # noqa: BLE001
        return ["crescendo", "many_shot", "context_overflow", "persona_priming"]


def _list_research_datasets() -> list[tuple[str, str]]:
    """(key, human description) pairs for the gated research-benchmark selector."""
    try:
        from .research.registry import KNOWN_DATASETS

        return [(k, ref.name) for k, ref in KNOWN_DATASETS.items()]
    except Exception:  # noqa: BLE001
        return []


def _research_disclaimer() -> str:
    """The canonical research-use disclaimer shown beside the opt-in checkbox."""
    try:
        from .research.base import RESEARCH_DISCLAIMER

        return RESEARCH_DISCLAIMER
    except Exception:  # noqa: BLE001
        return (
            "Research datasets reference potentially harmful prompts and are "
            "downloaded from their own official sources for authorized, ethical "
            "research only."
        )


#: Human labels + a colour class for each of the five response classes, ordered
#: worst-for-the-attacker to best. Drives the 5-class breakdown on the scan page.
_RESPONSE_CLASS_LABELS: tuple[tuple[str, str, str], ...] = (
    ("reject_irrelevant", "off-task", "good"),
    ("reject_safety", "refused (safe)", "good"),
    ("too_long", "truncated", "warn"),
    ("partial", "partial", "warn"),
    ("full", "full bypass", "bad"),
)


def _response_class_counts(report: ScanReport) -> dict[str, int]:
    """Tally a scan report's results into the five graded response classes.

    Uses :func:`~injectkit.evaluators.response_class.classify_result`, the frozen
    seam that grades a scored result without mutating it: ``full`` coincides with
    the engine's boolean ``success`` (a strong concrete-proof hit), so the
    ``full`` count equals the report's vulnerable count for the offline core. The
    other four classes add fidelity (why a non-success happened) on top of the
    pass/fail the page already shows. Returns a ``name -> count`` mapping over all
    five class keys (zeros included). Degrades to an empty mapping if the
    classifier is unavailable, so the page still renders without the breakdown.
    """
    try:
        from .evaluators.response_class import ResponseClass, classify_result
    except Exception:  # noqa: BLE001 - the page must render even without the seam
        return {}

    counts: dict[str, int] = {c.value: 0 for c in ResponseClass}
    for result in report.results:
        try:
            cls = classify_result(result)
        except Exception:  # noqa: BLE001 - one odd result must not blank the tally
            continue
        counts[cls.value] = counts.get(cls.value, 0) + 1
    return counts


# Last rendered HTML report, served at /report and embedded in the results page.
_LAST_REPORT_HTML: Optional[str] = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Form parsing helpers
# --------------------------------------------------------------------------- #
def _one(form: dict[str, list[str]], name: str) -> Optional[str]:
    """First non-blank value of ``name`` in a parsed query form, else None."""
    vals = form.get(name)
    if not vals:
        return None
    val = vals[0].strip()
    return val or None


def _checked(form: dict[str, list[str]], name: str) -> bool:
    """True when a checkbox/flag field is present with a truthy value."""
    vals = form.get(name)
    if not vals:
        return False
    return vals[0].strip().lower() not in ("", "0", "false", "off", "no")


def _build_config_from_form(form: dict[str, list[str]]):
    """Assemble a Config + filtered attack list from the submitted form.

    Shared by both the scan and the benchmark path so the target/technique
    selection is identical. Returns ``(config, attacks, target_obj, detectors)``.
    Raises ``ValueError`` if the technique filter excludes everything.
    """
    target: dict = {"kind": _one(form, "kind") or "mock"}
    for f in ("url", "model", "system"):
        if _one(form, f):
            target[f] = _one(form, f)

    techniques = [t for t in form.get("technique", []) if t in TECHNIQUES]

    overrides: dict = {
        "target": target,
        "use_judge": _checked(form, "judge"),
        "fail_on": _one(form, "fail_on") or "high",
        "report_format": "html",
    }
    if techniques:
        overrides["techniques"] = techniques

    config = load_config(cli_overrides=overrides)
    attacks = _load_attacks(config)
    if not attacks:
        raise ValueError("No attacks matched your technique selection.")

    target_obj = _build_target(config)
    detectors = _build_detectors(config)
    return config, attacks, target_obj, detectors


def _selected_transforms(form: dict[str, list[str]]) -> list:
    """Instantiate the transforms the user ticked under *mutate* (ordered)."""
    names = [n for n in form.get("mutate", []) if n and n.strip()]
    if not names:
        return []
    from .transforms.base import get_transform

    out = []
    for n in names:
        try:
            out.append(get_transform(n))
        except Exception:  # noqa: BLE001 - skip an unknown/removed transform name
            continue
    return out


def _selected_defense(form: dict[str, list[str]]):
    """Instantiate the chosen defense, or None for the 'none' baseline."""
    name = _one(form, "defense") or "none"
    if name == "none":
        return None
    try:
        from .defenses.base import get_defense

        return get_defense(name)
    except Exception:  # noqa: BLE001
        return None


def _selected_multiturn(form: dict[str, list[str]]):
    """Resolve the multi-turn strategy the user requested, or None.

    Only returns a strategy when the *multiturn* checkbox is ticked AND a known
    strategy name is selected. An unknown/removed strategy name degrades to None
    (single-shot) rather than raising, so the GUI never crashes on stale form
    state. The strategy still drives the benign-canary proxy end to end.
    """
    if not _checked(form, "multiturn"):
        return None
    name = _one(form, "multiturn_strategy")
    if not name:
        return None
    try:
        from .cli_robustness import build_strategy_for

        return build_strategy_for(name)
    except Exception:  # noqa: BLE001 - a bad strategy name falls back to single-shot
        return None


def _wrap_target_for_scan(target_obj, transforms: list, defense, strategy=None):
    """Wrap a target with the selected transforms + defense + multi-turn delivery.

    Reuses the exact wrapper classes the CLI/benchmark runner use so the GUI's
    transform/defense/multi-turn behaviour is identical to ``injectkit scan``:
    the multi-turn strategy wraps the innermost target (it delivers the
    engine-rendered, canary-bearing prompt as a conversation), then transforms
    are composed left-to-right over each outgoing turn, then the defense's hooks
    wrap each send. The benign canary is preserved throughout. Returns the
    (possibly unchanged) target plus the list of human-readable transform names
    applied.
    """
    applied_names: list[str] = []
    wrapped = target_obj
    if strategy is not None:
        from .cli_robustness import wrap_target_for_multiturn

        wrapped = wrap_target_for_multiturn(wrapped, strategy)
    if transforms:
        from .benchmark_runner import _TransformingTarget
        from .transforms.base import Compose

        combo = transforms[0] if len(transforms) == 1 else Compose(*transforms)
        wrapped = _TransformingTarget(wrapped, combo)
        applied_names = [getattr(t, "name", "?") for t in transforms]
    if defense is not None:
        from .benchmark_runner import _DefendedTarget

        wrapped = _DefendedTarget(wrapped, defense)
    return wrapped, applied_names


# --------------------------------------------------------------------------- #
# Scan execution (reuses the CLI pipeline)
# --------------------------------------------------------------------------- #
def run_scan(form: dict[str, list[str]]) -> ScanReport:
    """Build a Config from form fields and run a scan, returning the report.

    Honours the v0.2.0 robustness toggles: any ticked *mutate* transforms, a
    selected *defense*, and the *multi-turn* delivery strategy wrap the target
    before the engine scores it (the same wrappers the CLI/benchmark runner use),
    preserving the benign-canary proxy. The adaptive attacker is intentionally
    not engaged here because the GUI exposes no attacker-model fields; it stays a
    CLI-only flow so the GUI never blocks on a missing local model.
    """
    config, attacks, target_obj, detectors = _build_config_from_form(form)

    transforms = _selected_transforms(form)
    defense = _selected_defense(form)
    strategy = _selected_multiturn(form)
    target_obj, _applied = _wrap_target_for_scan(
        target_obj, transforms, defense, strategy
    )

    engine = Engine(
        target_obj,
        detectors,
        use_judge=config.use_judge,
        tool_version=__version__,
    )
    return engine.run(attacks)


# --------------------------------------------------------------------------- #
# Benchmark execution (reuses the benchmark runner + scorecard reporter)
# --------------------------------------------------------------------------- #
def run_benchmark(form: dict[str, list[str]]) -> BenchmarkResult:
    """Build a Config from form fields and run the ASR benchmark sweep.

    Sweeps the corpus across the selected transforms (Identity is always the
    baseline) and the selected defense plus the undefended baseline, returning a
    :class:`~injectkit.benchmark.BenchmarkResult`. Offline-first: with the mock
    target this runs with no network and no API key. Multi-turn / adaptive toggles
    are recorded for context; the adaptive attacker is only engaged when a local
    attacker model is actually available (otherwise the sweep runs without it so
    the GUI never blocks on a missing optional dependency).
    """
    config, attacks, target_obj, detectors = _build_config_from_form(form)

    transforms = _selected_transforms(form)
    defense = _selected_defense(form)
    defenses = [defense] if defense is not None else None

    from .benchmark_runner import run_benchmark as _run_benchmark

    return _run_benchmark(
        target_obj,
        attacks,
        detectors,
        transforms=transforms or None,
        defenses=defenses,
        use_judge=config.use_judge,
        tool_version=__version__,
    )


def run_research_benchmark(form: dict[str, list[str]]) -> BenchmarkResult:
    """Run a GATED research-dataset benchmark — only after explicit acknowledgment.

    Mirrors the CLI's ``--research-benchmark`` + ``--i-am-authorized`` contract:
    refuses unless the *acknowledge* checkbox is ticked, prints/echoes the
    research-use disclaimer, lazy-loads the dataset loader (which downloads from
    the dataset's own official source — never bundled), and benchmarks the loaded
    behaviours against the configured target. Tests stub the loader so nothing is
    downloaded offline.
    """
    if not _checked(form, "research_ack"):
        from .research.base import ResearchAcknowledgmentError

        raise ResearchAcknowledgmentError(
            "Research benchmark refused: you must tick the acknowledgment box to "
            "confirm authorized, ethical, research-only use before any dataset is "
            "downloaded.\n\n" + _research_disclaimer()
        )

    dataset = _one(form, "research_dataset")
    if not dataset:
        raise ValueError("Pick a research dataset to benchmark against.")

    try:
        limit = int(_one(form, "research_limit") or "25")
    except (TypeError, ValueError):
        limit = 25

    config, _attacks, target_obj, detectors = _build_config_from_form(form)

    # Lazy import keeps the gated research surface out of the offline core path.
    from .research import get_loader

    loader = get_loader(dataset)
    attacks = loader.load(acknowledge=True, limit=limit)
    if not attacks:
        raise ValueError(
            f"The {dataset} loader returned no behaviours (nothing to benchmark)."
        )

    transforms = _selected_transforms(form)
    defense = _selected_defense(form)
    defenses = [defense] if defense is not None else None

    from .benchmark_runner import run_benchmark as _run_benchmark

    return _run_benchmark(
        target_obj,
        attacks,
        detectors,
        transforms=transforms or None,
        defenses=defenses,
        use_judge=config.use_judge,
        tool_version=__version__,
    )


# --------------------------------------------------------------------------- #
# HTML pages
# --------------------------------------------------------------------------- #
_STYLE = """
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #0e1116; color: #e6edf3; }
.wrap { max-width: 880px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h1 .v { color: #768390; font-size: 14px; font-weight: 400; }
.sub { color: #adbac7; margin: 0 0 20px; }
.banner { background: #2d2410; border: 1px solid #5c4813; color: #e3b341;
          padding: 10px 14px; border-radius: 8px; font-size: 13px; margin: 0 0 24px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 20px; margin: 0 0 20px; }
label { display: block; font-weight: 600; margin: 14px 0 4px; font-size: 13px; }
input[type=text], input[type=number], select { width: 100%; padding: 8px 10px;
        border-radius: 6px; border: 1px solid #30363d; background: #0e1116;
        color: #e6edf3; font-size: 14px; }
.row { display: flex; gap: 16px; } .row > div { flex: 1; }
.techs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }
.techs label { font-weight: 400; display: flex; align-items: center; gap: 8px; margin: 0; }
.hint { color: #768390; font-size: 12px; margin: 4px 0 0; }
button { background: #238636; color: #fff; border: 0; padding: 11px 22px;
         border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 22px; }
button:hover { background: #2ea043; }
a { color: #58a6ff; } .chk { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
.chk label { margin: 0; font-weight: 400; }
.summary { display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 18px; }
.stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 18px; min-width: 110px; }
.stat .n { font-size: 26px; font-weight: 700; } .stat .l { color: #768390; font-size: 12px; }
.bad { color: #f85149; } .good { color: #3fb950; } .warn { color: #e3b341; }
.warnbox { background: #2d2410; border: 1px solid #5c4813; color: #e3b341;
           padding: 12px 16px; border-radius: 8px; font-size: 14px; margin: 0 0 18px; }
.gatebox { background: #20131b; border: 1px solid #5c1a3a; color: #ff9ecb;
           padding: 12px 16px; border-radius: 8px; font-size: 13px; margin: 14px 0 0; }
iframe { width: 100%; height: 1400px; border: 1px solid #30363d; border-radius: 10px; background: #fff; }
.err { background: #2d1416; border: 1px solid #5c1a1f; color: #ff7b72;
       padding: 14px 16px; border-radius: 8px; }
code { background: #0e1116; padding: 1px 5px; border-radius: 4px; }
fieldset { border: 1px solid #30363d; border-radius: 8px; margin: 16px 0 0; padding: 10px 14px 14px; }
legend { font-weight: 600; font-size: 13px; padding: 0 6px; color: #adbac7; }
"""

_BANNER = (
    "&#9888; Defensive / authorized use only — scan only endpoints you own "
    "or are explicitly authorized to test."
)


def _page(body: str) -> bytes:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>injectkit</title><style>{_STYLE}</style></head>"
        f"<body><div class=wrap>{body}</div></body></html>"
    ).encode("utf-8")


def _options(values, selected: str) -> str:
    """Build <option> markup, marking ``selected`` as the chosen value."""
    return "".join(
        f"<option value='{html.escape(str(v), quote=True)}'"
        f"{' selected' if v == selected else ''}>{html.escape(str(v))}</option>"
        for v in values
    )


def form_page(notice: str = "") -> bytes:
    techs = "".join(
        f"<label><input type=checkbox name=technique value='{t}'> {t}</label>"
        for t in TECHNIQUES
    )
    kinds = _options(TARGET_KINDS, "mock")
    fail = _options(FAIL_ON, "high")
    modes = _options(MODES, "scan")

    # Mutate transform checkboxes (a grid of the registry's transforms). The
    # v0.3.0 cipher/art/translate transforms are registered by _list_transforms()
    # so they appear here; translate is annotated when its offline translator dep
    # is missing (it still lists, but the note explains it will not run).
    transform_names = _list_transforms()
    if transform_names:
        labels = []
        missing_dep = False
        for t in transform_names:
            dep = _TRANSFORM_OPTIONAL_DEPS.get(t)
            note = ""
            if dep and not _dep_available(dep):
                note = (
                    f" <span class=hint>(needs <code>{html.escape(dep)}</code>)</span>"
                )
                missing_dep = True
            labels.append(
                "<label><input type=checkbox name=mutate "
                f"value='{html.escape(t, quote=True)}'> {html.escape(t)}{note}</label>"
            )
        mutate = "".join(labels)
        dep_hint = (
            "<p class=hint>A transform marked <i>needs &lt;dep&gt;</i> lists but "
            "will not run until you install that optional, offline extra.</p>"
            if missing_dep
            else ""
        )
        mutate_block = (
            "<label>Mutate <span class=hint>(obfuscation/semantic transforms — "
            "ticking any measures robustness against input filtering; identity is "
            "always the baseline)</span></label>"
            f"<div class=techs>{mutate}</div>{dep_hint}"
        )
    else:
        mutate_block = ""

    defenses = _list_defenses()
    defense_select = (
        "<div><label>Defense</label>"
        f"<select name=defense>{_options(defenses, 'none')}</select>"
        "<p class=hint>Wrap the target in a mitigation before scoring "
        "(measures ASR with the defense). Default: none.</p></div>"
    )

    mt_strategies = _list_multiturn_strategies()
    multiturn_select = "".join(
        f"<option value='{html.escape(s, quote=True)}'>{html.escape(s)}</option>"
        for s in mt_strategies
    )

    # Named adaptive-attacker dropdown (v0.3.0). Informational: the GUI exposes no
    # attacker-model fields, so the dropdown stays disabled (drive an attacker
    # from the CLI). gcg is annotated when torch/transformers is missing.
    attackers = _list_attackers()
    if attackers:
        opts = []
        attacker_dep_note = False
        for name, kind, runnable in attackers:
            extra = "" if runnable else " — needs torch/transformers"
            if not runnable:
                attacker_dep_note = True
            opts.append(
                f"<option value='{html.escape(name, quote=True)}'>"
                f"{html.escape(name)} ({html.escape(kind)}){extra}</option>"
            )
        attacker_select = (
            "<div style='margin-top:10px'><label>Adaptive attacker strategy "
            "<span class=hint>(named automated red-teamers)</span></label>"
            "<select name=attacker disabled style='max-width:340px'>"
            f"{''.join(opts)}</select>"
            "<p class=hint>pair/tap/autodan/gptfuzzer drive a local attacker model; "
            "<b>gcg</b> is a white-box gradient suffix (HuggingFace target only). "
            "All optimise toward the <b>benign canary</b> marker, never harmful "
            "content. <b>CLI-only</b>: run <code>injectkit scan --attacker &lt;name&gt;"
            "</code> &mdash; the GUI exposes no attacker-model fields, so this stays "
            "informational here."
            + (
                " <i>gcg needs the optional <code>torch</code>/<code>transformers"
                "</code> extra installed to run.</i>"
                if attacker_dep_note
                else ""
            )
            + "</p></div>"
        )
    else:
        attacker_select = ""

    # Per-target config hints rendered as a small legend so the user knows what
    # the URL/model fields mean for each kind.
    target_help = " &middot; ".join(
        f"<b>{html.escape(k)}</b> = {v}" for k, v in _TARGET_HELP.items()
    )

    # The gated research-benchmark block (collapsed-looking fieldset). It only
    # does anything when the acknowledgment box is ticked AND mode=benchmark.
    datasets = _list_research_datasets()
    if datasets:
        ds_opts = "".join(
            f"<option value='{html.escape(k, quote=True)}'>"
            f"{html.escape(k)} — {html.escape(name)}</option>"
            for k, name in datasets
        )
        research_block = (
            "<fieldset><legend>Research benchmark (gated, opt-in)</legend>"
            "<p class=hint>Benchmark against an official public research dataset. "
            "The dataset is downloaded from its OWN source (never bundled) and "
            "only when you explicitly acknowledge the terms below. Benchmark mode "
            "only.</p>"
            "<label>Dataset</label>"
            f"<select name=research_dataset>{ds_opts}</select>"
            "<div class=row><div><label>Limit "
            "<span class=hint>(max behaviours)</span></label>"
            "<input type=number name=research_limit value=25 min=1></div></div>"
            "<div class=chk><input type=checkbox name=research_ack value=1 "
            "id=ack><label for=ack>I acknowledge the research-use terms "
            "(required to load any dataset).</label></div>"
            f"<div class=gatebox>{html.escape(_research_disclaimer())}</div>"
            "</fieldset>"
        )
    else:
        research_block = ""

    return _page(
        f"<h1>injectkit <span class=v>v{__version__}</span></h1>"
        "<p class=sub>Red-team your own LLM app for prompt injection — "
        "now with transforms, defenses, multi-turn, adaptive, and an ASR "
        "scorecard.</p>"
        f"<div class=banner>{_BANNER}</div>"
        f"{notice}"
        "<form method=post action='/scan'><div class=card>"
        "<div class=row>"
        f"<div><label>Mode</label><select name=mode>{modes}</select>"
        "<p class=hint><b>scan</b> = one report. <b>benchmark</b> = ASR "
        "robustness scorecard (sweeps transforms &times; defenses).</p></div>"
        f"<div><label>Target</label><select name=kind>{kinds}</select></div>"
        f"<div><label>Fail-on (CI gate)</label><select name=fail_on>{fail}</select>"
        "<p class=hint>Lowest severity that counts as a failed gate (scan mode)."
        "</p></div>"
        "</div>"
        f"<p class=hint>{target_help}</p>"
        "<label>Endpoint / server URL "
        "<span class=hint>(http, or the ollama/openai server base URL)</span></label>"
        "<input type=text name=url placeholder='http://localhost:11434  or  "
        "https://your-app.example.com/api/chat'>"
        "<div class=row>"
        "<div><label>Model <span class=hint>(optional)</span></label>"
        "<input type=text name=model placeholder='llama3.1 / local-model / "
        "claude-opus-4-8'></div>"
        "<div><label>System prompt <span class=hint>(optional)</span></label>"
        "<input type=text name=system placeholder=\"You are a helpful assistant.\"></div>"
        "</div>"
        "<label>Techniques <span class=hint>(none = run all 6)</span></label>"
        f"<div class=techs>{techs}</div>"
        # ---- robustness fieldset (mutate / defense / multiturn / adaptive) ----
        "<fieldset><legend>Robustness (v0.2.0)</legend>"
        f"{mutate_block}"
        "<div class=row style='margin-top:10px'>"
        f"{defense_select}"
        "<div><label>Multi-turn <span class=hint>(deliver as a conversation)"
        "</span></label>"
        "<div class=chk><input type=checkbox name=multiturn value=1 id=mt>"
        "<label for=mt>Enable</label>"
        f"<select name=multiturn_strategy style='max-width:200px'>{multiturn_select}"
        "</select></div></div>"
        "</div>"
        "<div class=chk><input type=checkbox name=adaptive value=1 id=adapt "
        "disabled>"
        "<label for=adapt>Adaptive attacker (local-model-first, structure-only, "
        "benign-canary objective) &mdash; <b>CLI-only</b>: run "
        "<code>injectkit scan --adaptive</code> with a local attacker model. The "
        "GUI exposes no attacker-model fields, so this stays disabled here.</label>"
        "</div>"
        f"{attacker_select}"
        "</fieldset>"
        # ---- judge ----
        "<div class=chk><input type=checkbox name=judge value=1 id=judge>"
        "<label for=judge>Use LLM judge (sharper grading — needs an Anthropic API "
        "key; off = fully offline)</label></div>"
        # ---- gated research ----
        f"{research_block}"
        "<button type=submit>Run</button>"
        "</div></form>"
        "<p class=hint>Tip: leave everything default and click <b>Run</b> to watch "
        "injectkit attack the built-in mock target with zero setup. Switch "
        "<b>Mode</b> to <b>benchmark</b> for the ASR scorecard.</p>"
    )


def results_page(report: ScanReport) -> bytes:
    failed = report.failed
    errored = report.errored
    worst = report.highest_severity
    worst_s = worst.value if worst else "none"
    fcls = "bad" if failed else "good"
    ecls = "warn" if errored else ""

    # An all-errored scan never reached the target — say so plainly instead of
    # implying the target defended everything.
    if report.all_errored:
        notice = (
            "<div class=warnbox>&#9888; Target unreachable — all "
            f"{report.total} attack(s) errored (no usable responses). This scan "
            "could not be graded and is <b>not</b> a pass. Check the target URL "
            "or credentials.</div>"
        )
    elif errored:
        notice = (
            f"<div class=warnbox>&#9888; {errored} attack(s) could not reach the "
            "target (errors) and are not counted as defended.</div>"
        )
    else:
        notice = ""

    errored_stat = (
        f"<div class=stat><div class='n {ecls}'>{errored}</div>"
        "<div class=l>errored</div></div>"
        if errored
        else ""
    )

    # 5-class response breakdown (v0.3.0): grade every result into one of the five
    # classes. `full` coincides with the boolean vulnerable count; the rest add
    # fidelity (why a non-success happened). Only rendered when the seam is present.
    class_counts = _response_class_counts(report)
    if class_counts:
        cells = "".join(
            f"<div class=stat><div class='n {cls}'>{class_counts.get(key, 0)}</div>"
            f"<div class=l>{html.escape(label)}</div></div>"
            for key, label, cls in _RESPONSE_CLASS_LABELS
        )
        class_block = (
            "<p class=hint style='margin-top:8px'>5-class response breakdown "
            "<span class=hint>(SoK Prompt Hacking 2410.13901 / StrongREJECT; "
            "<b>full</b> = a benign-canary bypass = the vulnerable count)</span></p>"
            f"<div class=summary>{cells}</div>"
        )
    else:
        class_block = ""

    return _page(
        "<h1>Scan results</h1>"
        f"<p class=sub>Target: <code>{html.escape(report.target_name)}</code>"
        + (f" &middot; <code>{html.escape(report.target_model)}</code>" if report.target_model else "")
        + "</p>"
        f"{notice}"
        "<div class=summary>"
        f"<div class=stat><div class=n>{report.total}</div><div class=l>attacks</div></div>"
        f"<div class=stat><div class='n good'>{report.passed}</div><div class=l>defended</div></div>"
        f"<div class=stat><div class='n {fcls}'>{failed}</div><div class=l>vulnerable</div></div>"
        f"{errored_stat}"
        f"<div class=stat><div class='n {fcls}'>{html.escape(worst_s)}</div><div class=l>worst severity</div></div>"
        "</div>"
        f"{class_block}"
        "<p><a href='/'>&larr; New run</a> &nbsp;&middot;&nbsp; "
        "<a href='/report' target=_blank>open full report in a new tab</a></p>"
        "<iframe src='/report' title='injectkit report'></iframe>"
    )


def benchmark_results_page(result: BenchmarkResult, *, research: bool = False) -> bytes:
    """Render the ASR scorecard summary page (the full scorecard is in /report)."""
    from .reporters.scorecard import robustness_grade

    overall = result.overall("none")
    overall_asr = result.overall_asr("none")
    grade = robustness_grade(overall_asr) if overall and overall.attempts else "N/A"
    gcls = "good" if grade in ("A+", "A", "B") else ("warn" if grade in ("C", "D") else "bad")

    succeeded = overall.successes if overall else 0
    attempts = overall.attempts if overall else 0
    errored = overall.errored if overall else 0
    m = result.metadata

    title = "Research benchmark results" if research else "Benchmark results"
    research_note = (
        "<div class=gatebox>This scorecard used an opt-in research dataset, "
        "downloaded from its official source under the research-use terms you "
        "acknowledged. ASR remains the benign-canary proxy.</div>"
        if research
        else ""
    )

    return _page(
        f"<h1>{title}</h1>"
        f"<p class=sub>Target: <code>{html.escape(m.target_name)}</code>"
        + (f" &middot; <code>{html.escape(m.target_model)}</code>" if m.target_model else "")
        + "</p>"
        f"{research_note}"
        "<div class=summary>"
        f"<div class=stat><div class='n {gcls}'>{html.escape(grade)}</div>"
        "<div class=l>robustness grade</div></div>"
        f"<div class=stat><div class='n {gcls}'>{overall_asr * 100:.1f}%</div>"
        "<div class=l>overall ASR</div></div>"
        f"<div class=stat><div class=n>{succeeded}/{attempts}</div>"
        "<div class=l>succeeded</div></div>"
        + (
            f"<div class=stat><div class='n warn'>{errored}</div>"
            "<div class=l>errored</div></div>"
            if errored
            else ""
        )
        + "</div>"
        "<p class=hint>transforms: <code>"
        f"{html.escape(', '.join(m.transforms) or 'identity')}</code> &middot; "
        f"defenses: <code>{html.escape(', '.join(m.defenses) or 'none')}</code>"
        + (f" &middot; corpus <code>{html.escape(m.corpus_hash[:12])}</code>" if m.corpus_hash else "")
        + "</p>"
        "<p><a href='/'>&larr; New run</a> &nbsp;&middot;&nbsp; "
        "<a href='/report' target=_blank>open the full scorecard in a new tab</a></p>"
        "<iframe src='/report' title='injectkit scorecard'></iframe>"
    )


def error_page(message: str) -> bytes:
    return _page(
        "<h1>Run failed</h1>"
        f"<div class=err>{html.escape(message)}</div>"
        "<p style='margin-top:18px'><a href='/'>&larr; Back</a></p>"
        "<p class=hint>The <b>mock</b> target always works offline. <b>ollama</b>/"
        "<b>openai</b> need a local server running; <b>hf</b> loads a local model "
        "(needs transformers/torch); <b>anthropic</b> needs "
        "<code>ANTHROPIC_API_KEY</code>; <b>http</b> needs a reachable URL.</p>"
    )


# --------------------------------------------------------------------------- #
# Dispatch: scan vs benchmark vs gated research benchmark
# --------------------------------------------------------------------------- #
def handle_submit(form: dict[str, list[str]]) -> tuple[bytes, Optional[str]]:
    """Run the requested mode and return ``(page_bytes, report_html)``.

    ``report_html`` is the standalone HTML to serve at ``/report`` (a scan report
    or an ASR scorecard), or ``None`` when the page already carries everything
    (an error page). All exceptions are surfaced as a friendly error page rather
    than propagated, so a misconfigured target never crashes the server.
    """
    mode = (_one(form, "mode") or "scan").lower()
    research = _checked(form, "research_ack") and _one(form, "research_dataset")

    try:
        if mode == "benchmark" and research:
            result = run_research_benchmark(form)
            from .reporters.scorecard import ScorecardHtmlReporter

            return benchmark_results_page(result, research=True), (
                ScorecardHtmlReporter().render(result)
            )
        if mode == "benchmark":
            result = run_benchmark(form)
            from .reporters.scorecard import ScorecardHtmlReporter

            return benchmark_results_page(result), ScorecardHtmlReporter().render(result)
        # Default: a single scan.
        report = run_scan(form)
        reporter = _build_reporter("html")
        return results_page(report), reporter.render(report)
    except Exception as exc:  # noqa: BLE001 - surface any failure as a friendly page
        return error_page(f"{type(exc).__name__}: {exc}"), None


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        global _LAST_REPORT_HTML
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(form_page())
        elif path == "/report":
            with _LOCK:
                report = _LAST_REPORT_HTML
            if report is None:
                self._send(form_page("<div class=banner>No run has happened yet.</div>"))
            else:
                self._send(report.encode("utf-8"))
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        global _LAST_REPORT_HTML
        if self.path.split("?", 1)[0] != "/scan":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        form = parse_qs(raw, keep_blank_values=True)
        page, report_html = handle_submit(form)
        if report_html is not None:
            with _LOCK:
                _LAST_REPORT_HTML = report_html
        self._send(page)

    def log_message(self, *args) -> None:  # keep the console quiet
        return


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the local GUI server (blocks until Ctrl-C)."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"injectkit GUI running at {url}  (Ctrl-C to stop)")
    print("Defensive / authorized use only — scan only what you own.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m injectkit.web",
        description="Launch the injectkit local web GUI (localhost only).",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default localhost).")
    p.add_argument("--port", type=int, default=8765, help="Port (default 8765).")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open a browser.")
    args = p.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
