# injectkit v0.3 — BASELINE (recon, gates all v0.4 work)

This document is an **audit of the injectkit code as it actually exists on disk at
v0.3.0**, produced by CHUNK 0-recon. It maps the v0.3 codebase to the claims v0.4
work will build on, records the **actual** test count, and audits the judge /
default-classifier licensing posture so later chunks build on facts, not memory.

It is descriptive, not aspirational: every section below was verified against the
source files cited (paths are repo-relative, package root is the **flat-layout**
`injectkit/` at the repo root — confirmed, see §1). Anything not yet implemented is
called out explicitly as a **GAP**.

- Audited at: package version `0.3.0` (`injectkit/__init__.py` `__version__`,
  `pyproject.toml` `version`).
- Test result at audit time: **1276 passed, 10 skipped, 0 failed** — the pytest
  process exits **0** (clean). 9 of those skips are the GitHub-Action entrypoint
  tests, which now self-skip on the one host configuration that cannot run them
  (a Windows Python whose `bash` resolves to the WSL launcher shim); see §6.

---

## 1. Top-level package layout — FLAT, at repo root (confirmed)

The importable package is `injectkit/` **at the repo root** (NOT `src/injectkit`).
Confirmed by direct listing: `injectkit/__init__.py` sits beside `pyproject.toml`
at the repo root, and `pyproject.toml` declares `name = "injectkit"` with no
`src/` layout. `injectkit.egg-info/` at the root corroborates a root-level
editable install.

Top-level modules (files directly under `injectkit/`):

| Module | Role |
| --- | --- |
| `__init__.py` | Public API surface; re-exports the `models` dataclasses; `__version__ = "0.3.0"`. |
| `__main__.py` | `python -m injectkit` entry. |
| `models.py` | Core dataclasses: `Attack`, `AttackResult`, `DetectorVerdict`, `Finding`, `ScanReport`, `Severity`, `TargetConfig`, `TargetResponse`, `Verdict`. |
| `config.py` | Run config incl. `DEFAULT_JUDGE_MODEL`. |
| `engine.py` | The scan engine / pipeline (see §3). |
| `cli.py`, `cli_robustness.py`, `web.py` | CLI + web/robustness front-ends. |
| `benchmark.py`, `benchmark_runner.py` | Benchmark harness. |
| `indirect.py` | Indirect-injection plumbing. |

Sub-packages under `injectkit/`:

| Package | Contents |
| --- | --- |
| `attackers/` | Black-box adaptive attackers (`pair`, `tap`, `autodan`, `gptfuzzer`) + white-box `gcg`; `base.py`, `whitebox_base.py`, `registry.py`, `adaptive.py`. |
| `attacks/` | Multi-turn strategy contract (`base.py`, `multiturn.py`). |
| `corpus/` | YAML attack corpus + `loader.py` (see §3). |
| `defenses/` | Mitigations (`base.py`, `mitigations.py`). |
| `evaluators/` | Detectors + scoring + 5-class grader (see §4, §5). |
| `reporters/` | `terminal`, `json`, `markdown`, `html`, `sarif`, `scorecard`, `base`. |
| `research/` | Gated opt-in dataset loaders + ASR judge (`base.py`, `datasets.py`, `registry.py`). |
| `targets/` | Adapters: `anthropic_target`, `openai_compat`, `ollama`, `hf`, `http`, `mcp`, `conversational`, `base`. |
| `transforms/` | `encoders`, `ciphers`, `translate`, `base`. |

---

## 2. The GCG interface + loss

White-box GCG/AmpleGCG family. Two files:

- **Contract / base — `injectkit/attackers/whitebox_base.py`:**
  - `WhiteBoxModel` (a `runtime_checkable` `Protocol`) is the white-box seam.
    Methods: `token_ids(text) -> ids`, `decode(ids) -> str`,
    `target_loss(input_ids, target_ids) -> loss`,
    `token_gradients(input_ids, target_ids, suffix_slice) -> [suffix_len, vocab]`,
    plus a `name` attribute.
  - **The loss** is the seam method `WhiteBoxModel.target_loss`: the LM loss of
    producing `target_ids` after `input_ids` (lower ⇒ closer to emitting the
    target string). GCG **minimises** it over candidate suffix tokens. The
    gradient signal is `token_gradients` (top-k most-negative entries per slot are
    the candidate replacements). The concrete numeric loss (cross-entropy) lives
    in a real HF `WhiteBoxModel` implementation behind the seam; in tests a
    `StubWhiteBoxModel` returns deterministic fake logits/grads/loss.
  - `GCGConfig` (dataclass): `max_steps=50`, `suffix_len=20`, `batch_size=64`,
    `top_k=256`, `target_string=None` (defaults to the **benign** marker),
    `init_suffix=None`, `trigger=DEFAULT_TRIGGER` (`"INJECTOK-"`), `seed=0`.
  - `GCGStep` (dataclass): `step`, `suffix`, `loss`, `succeeded` — the per-step
    trajectory recorded for the transcript.
  - `WhiteBoxGCGAttacker` is the **abstract base**: `run` and `_optimize_suffix`
    raise `NotImplementedError` (they are overridden by the concrete subclass).
  - `import_torch_transformers()` lazy-imports `torch` + `transformers` with a
    friendly `AttackerError` if missing (white-box ⇒ HF-only, compute-heavy).

- **Concrete — `injectkit/attackers/gcg.py`:** `GCGSuffixAttacker` implements the
  greedy-coordinate-gradient loop entirely through the `WhiteBoxModel` seam:
  `_optimize_suffix` per step calls `token_gradients`, takes `top_k` candidates
  per slot, samples a `batch_size` batch of single-token swaps, keeps the
  lowest-`target_loss` swap, records a `GCGStep`, and stops at `max_steps` or on
  the first benign-marker success. `make_gcg_attacker(model, config, **opts)` is
  the registry factory (requires a white-box `model`); it registers `"gcg"` on
  the attacker registry at import time.

**Safety posture (verified in source):** the optimisation objective is **always**
the per-run **benign canary marker** `<trigger><canary>` (a robustness probe),
never a harmful string. **Zero** harmful AdvBench/AmpleGCG suffix artifacts are
bundled. `load_amplegcg_suffixes(acknowledge=...)` is the only path to published
suffixes and is **gated** through `injectkit.research.require_acknowledgment`;
even when loaded a suffix is used only as benign *initial filler* whose objective
is re-pointed at the benign marker. It currently returns `[]` (the gate is fully
exercised; the concrete download is a research-loader responsibility — **GAP**:
no dataset-specific download wired yet).

---

## 3. The corpus → engine → detectors pipeline

**Corpus (`injectkit/corpus/`).** Six YAML technique files loaded by
`corpus/loader.py`:

| File | Technique | Attack count |
| --- | --- | --- |
| `direct_injection.yaml` | direct_injection | 12 |
| `indirect_injection.yaml` | indirect_injection | 12 |
| `jailbreak.yaml` | jailbreak | 14 |
| `data_exfiltration.yaml` | data_exfiltration | 9 |
| `system_prompt_leak.yaml` | system_prompt_leak | 9 |
| `tool_abuse.yaml` | tool_abuse | 9 |
| **Total** | | **65** |

`load_corpus(path)` accepts a file or directory (dir loads all `*.yaml`/`*.yml`
sorted, dedupes ids). `load_attack_file` validates each entry against the schema
(required: `id`, `technique`, `name`, `description`, `severity`, `payload`;
optional `success_conditions`, `references`, `tags`, `system`, `context`) and
raises `CorpusError` on malformed input. Valid `success_conditions` keys:
`marker_echoed`, `canary_in_output`, `regex`, `refusal_absent`,
`system_prompt_leaked`.

**Engine (`injectkit/engine.py`).** `Engine(target, detectors=..., use_judge=...,
canary_factory=make_canary, ...)`. `make_canary()` mints a fresh per-run canary.
Pipeline:

1. `run(attacks)` iterates attacks, calling `run_one` per attack, builds a
   `ScanReport` (one `AttackResult` per attack, a `Finding` per success).
2. `run_one(attack)`: mint canary → `attack.render(canary)` (and render
   `system`/`context` with the canary) → `_send` to the target → run each
   detector via `_evaluate` (defensive: a raising detector becomes a non-success
   verdict; a raising target becomes an errored `TargetResponse`) → `score(...)`.
3. Extra paths: `run_strategy` (multi-turn / adaptive reply-referencing),
   `run_transformed` (transforms), `run_defended` (defenses), `run_adaptive`
   (black-box attackers). `run_scan(...)` is the module-level convenience entry.

Default detector when none supplied: a single offline
`HeuristicDetector()` — the engine works with zero config and no API key.

**Detectors (`injectkit/evaluators/`).**
- `heuristics.py` — `HeuristicDetector` (offline). `DEFAULT_TRIGGER = "INJECTOK-"`.
  `_STRONG_CONDITIONS` = strong concrete-proof conditions. Always-on.
- `judge.py` — `JudgeDetector`, an **optional** LLM-backed detector. **Anthropic**
  (default model `claude-haiku-4-5`, `DEFAULT_JUDGE_MODEL`); `anthropic` SDK is
  lazy-imported; off unless `--judge`/`use_judge` is set. Reports under the
  reserved detector name `"judge"`.
- `research/datasets.py` — `RefusalComplianceDetector`, an **offline refusal-vs-
  compliance ASR judge** (keyword `REFUSAL_MARKERS`) used by the gated research
  benchmark path. This is a heuristic, not an ML classifier (see §7).

---

## 4. The 5-class grader

`injectkit/evaluators/response_class.py` freezes the 5-class response framework
(StrongREJECT / SoK-style), on top of — not replacing — the boolean `success`:

`ResponseClass` (str-Enum, worst→best for the attacker):
`REJECT_IRRELEVANT`, `REJECT_SAFETY`, `TOO_LONG`, `PARTIAL`, `FULL`.

- `ResponseClass.is_success` is `True` **only** for `FULL` (frozen invariant).
- `FULL` is defined to coincide with the engine's heuristic boolean `success`: a
  strong concrete-proof condition fired (`STRONG_SUCCESS_CONDITIONS` =
  `marker_echoed`, `canary_in_output`, `regex`, `system_prompt_leaked`) or the
  full marker text is present.
- `PARTIAL` = a weaker signal (judge-only success, or a bare-canary leak with no
  strong condition).
- `REJECT_SAFETY` = `response.refused`; `TOO_LONG` = length-capped stop reason
  (`LENGTH_STOP_REASONS`); `REJECT_IRRELEVANT` = off-task/no signal/transport
  error.

Pure helpers: `classify_response(attack, response, canary, verdicts, *, trigger)`
and `classify_result(result, *, trigger)` — no network/clock/RNG, do not mutate
the result. `classify_result(r).is_success` agrees with `r.success` for the
offline heuristic core. Re-exported from `evaluators/__init__.py`.

**Scoring (`evaluators/scoring.py`).** `score_verdicts` / `score` combine verdicts
deterministically: refusal ⇒ defender wins; judge precedence when enabled and the
judge produced signal, else heuristics; per-technique severity floors
(`TECHNIQUE_MIN_SEVERITY`); confidence from the deciding verdicts.

---

## 5. Attacker registry (context for §2)

`injectkit/attackers/registry.py` pre-declares five attacker specs and exposes
`register_attacker(name, factory)`: `pair`, `tap`, `autodan`, `gptfuzzer`
(black-box adaptive), and `gcg` (white-box, HF-only). Each concrete attacker
module wires its factory at import time.

---

## 6. Actual current test count (pytest)

Run with the repo venv interpreter; the pytest process exits **0**:

```
$ ./.venv/Scripts/python.exe -m pytest -q
...
1276 passed, 10 skipped in ~141s
```

The colored summary line above is **not** flushed into a redirected pipe through
this host's WSL→Windows Python interop (only the per-test dots and the final exit
code reach captured stdout), so the count is recorded from pytest's own machine-
readable JUnit report rather than from screen-scraping. That report is the
authoritative, reproducible source for the figures below:

```
$ ./.venv/Scripts/python.exe -m pytest -q --junit-xml=.pytest_junit.xml
$ # <testsuite ... tests="1286" errors="0" failures="0" skipped="10" ...>
$ #   -> passed = 1286 - 0 - 0 - 10 = 1276
```

- **Effective passing suite: 1276 passed, 10 skipped, 0 failed, 0 errors**
  (1286 collected; 49 `tests/test_*.py` files). Process **exit code 0**.
- **History (resolved):** prior to this fix the local run reported `9 failed,
  1276 passed, 1 skipped` and exited non-zero (rc 1). The 9 failures were all in
  `tests/test_action_entrypoint.py` and were **environment-only, never code
  regressions**: that test resolves `bash` via `shutil.which("bash")`, which on a
  Windows Python under WSL finds `C:\Windows\System32\bash.EXE` (the WSL launcher
  shim). The shim launches the Linux distro's bash, which only sees the Linux
  filesystem and so cannot resolve the **Windows-style** path
  (`C:/Users/dukot/injectkit/entrypoint.sh`), yielding `returncode 127` /
  "No such file or directory". `entrypoint.sh` exists and is executable; no
  injectkit code is involved.
- **Fix applied (this chunk):** `tests/test_action_entrypoint.py` now detects that
  exact shim — `shutil.which("bash")` whose basename is `bash.exe` under a Windows
  `System32` directory — and **skips** (via `pytestmark`) instead of failing, so
  the **local exit code now matches CI (0)**. Real Git Bash, Cygwin/MSYS bash
  (`.../usr/bin/bash.exe`), and native Linux bash (`/usr/bin/bash`, including the
  GitHub-Action CI runner) are **not** matched and run the suite normally. The 9
  entrypoint tests therefore run green on CI and contribute 9 of the 10 local
  skips here (the 10th skip pre-dates this chunk).

This chunk changed exactly two files beyond the baseline doc itself: the one
skip-guard in `tests/test_action_entrypoint.py` (no production / `injectkit/`
code touched) and this §6 / header record. **No code regressions.**

---

## 7. Judge licensing audit — default judge MUST be MIT from-scratch (GAP)

The v0.4 plan requires the **default** ASR judge to be bundleable under injectkit's
MIT license, which rules out license-encumbered classifiers:

- **Llama-Guard-3** — gated (Hugging Face access request) and governed by the
  **Llama 3.1 Community License**, not OSI-approved/permissive. **NON-bundleable.**
- **HarmBench classifier (HarmBench-cls)** — derived from **Llama-2** and governed
  by the **Llama 2 Community License**. **NON-bundleable.**

Conclusion: the **default** judge must be an **MIT, from-scratch classifier**
(call it `clean_cls`) that injectkit can ship and that carries no upstream license
encumbrance.

**Audit of current code — the default `clean_cls` judge does NOT yet exist. GAP:**

- A repo-wide search for `clean_cls` / `Llama-Guard` / `HarmBench-cls` /
  "from-scratch classifier" finds **no such classifier in code**. The only matches
  are: (a) `injectkit/research/datasets.py` + `registry.py`, where **HarmBench** is
  referenced solely as a *gated, never-bundled, download-on-demand* benchmark
  **dataset** (not a judge model and not shipped); and (b) CLI/docs strings listing
  `harmbench` as a selectable research dataset key.
- The judges that **do** exist today are: `JudgeDetector` (optional **Anthropic**
  API LLM judge, `evaluators/judge.py`) and `RefusalComplianceDetector` (offline
  **keyword** refusal-vs-compliance heuristic, `research/datasets.py`). Neither is
  an MIT from-scratch ML classifier; there is **no** Llama-Guard / HarmBench-cls
  model bundled (good — nothing non-bundleable is shipped).
- **GAP to close in a later v0.4 chunk:** implement an MIT, from-scratch
  `clean_cls` classifier and make it the default ASR judge. Until then the offline
  default grading is the heuristic detectors (`HeuristicDetector` +
  `RefusalComplianceDetector`), and any model-backed grading is the opt-in
  Anthropic `JudgeDetector`.

**Net licensing posture today: clean.** injectkit bundles no Llama-derived or
otherwise non-permissive judge model or dataset; the gap is the *absence* of the
required MIT default classifier, not the *presence* of an encumbered one.

---

## 8. Decisions of record this baseline locks in for v0.4

1. Package stays **flat-layout** `injectkit/` at the repo root.
2. Benign-canary methodology is load-bearing: GCG and all attackers optimise for /
   detect the **benign marker** `<trigger><canary>`, never harmful content.
3. The boolean `success` invariant is frozen: `ResponseClass.is_success` is `FULL`
   and nothing else; `FULL` ⇔ the engine's heuristic boolean success.
4. No non-permissive judge model or dataset may be bundled. The **default** judge
   must be the MIT from-scratch `clean_cls` (to be built) — Llama-Guard-3 and
   HarmBench-cls are NON-bundleable and are referenced (datasets only) strictly
   through the gated, download-on-demand research loader.
