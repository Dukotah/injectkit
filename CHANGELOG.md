# Changelog

All notable changes to **injectkit** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.3.0] — 2026-06-16

injectkit widens the attack surface it can measure with more obfuscation
ciphers, a semantic translation transform, reply-aware multi-turn escalation, a
named-attacker registry covering the canonical automated-jailbreak papers, an
optional white-box GCG suffix optimizer, and a richer 5-class response grade —
still benign-canary based, still offline-first, still **defensive /
authorized-use only**. Every technique cites a primary research source in the
new `docs/RESEARCH.md`.

### Added

- **Cipher & encoding transforms** — six new canary-preserving, deterministic
  transforms on the `--mutate` axis (`transforms/ciphers.py`): `caesar`,
  `atbash`, `morse`, `unicode_escape`, `artprompt` (ASCII-art masking), and
  `selfcipher` (role-play cipher framing). Registered through the existing
  `TransformRegistry` via `register_builtin_ciphers()`. Grounded in CipherChat
  (arXiv:2308.06463) and ArtPrompt (arXiv:2402.11753).
- **Semantic translation transform** — a `translate` transform
  (`transforms/translate.py`) routes the payload through a low-resource language
  (default Swahili) to probe cross-lingual robustness; a *semantic* transform,
  not a character cipher. Uses a lazy, offline `argostranslate` backend behind a
  `Translator` protocol, with a friendly error if it is not installed. Grounded
  in low-resource-language jailbreak findings (arXiv:2310.02446 / MultiJail
  arXiv:2310.06474).
- **Reply-aware crescendo strategies** — two new multi-turn strategies
  (`attacks/multiturn.py`), both added to `MULTI_TURN_STRATEGIES`:
  `crescendo_reply` escalates by quoting the model's own prior replies before the
  scored seed-payload ask, and `crescendo_decompose` breaks the benign objective
  into a chain of individually-benign, canary-free sub-tasks and only carries the
  live marker on the final scored turn (the agent-decomposition variant). Grounded
  in Crescendo (arXiv:2404.01833).
- **Named-attacker registry** — a pre-seeded `AttackerRegistry`
  (`attackers/registry.py`) declaring the canonical automated jailbreak
  techniques, each carrying its citation: the black-box **PAIR**
  (arXiv:2310.08419), **TAP** (arXiv:2312.02119), **AutoDAN** (arXiv:2310.04451),
  and **GPTFUZZER** (arXiv:2309.10253), plus the white-box **GCG**. They optimize
  attack *structure* against the benign canary proxy, never harmful content.
- **White-box GCG suffix optimizer (optional, HuggingFace-only)** — a
  `WhiteBoxGCGAttacker` base (`attackers/whitebox_base.py`) that optimizes an
  adversarial suffix from model gradients so a **local** white-box HuggingFace
  model emits the **benign** canary marker — a robustness test, never harmful
  output. `torch` / `transformers` are lazy-imported and the optimization is
  compute-heavy (GPU recommended). **No harmful suffix artifact is bundled.**
  Grounded in GCG / AmpleGCG (arXiv:2404.07921) and Mask-GCG (arXiv:2509.06350).
- **5-class response scoring** — `evaluators/response_class.py` grades each reply
  into `reject_irrelevant` · `reject_safety` · `too_long` · `partial` · `full`,
  while keeping the boolean headline **frozen**: a scan succeeds only on `full`
  (`ResponseClass.is_success`). Grounded in SoK Prompt Hacking (arXiv:2410.13901)
  and StrongREJECT.
- **Docs** — new `docs/RESEARCH.md` (the cited 2023–2026 research map, with the
  honest frontier-robustness caveat that the "90%+ on flagship models" narrative
  is overstated); expanded `docs/TAXONOMY.md` and `docs/BENCHMARK.md` for the new
  families, attackers, and 5-class grade; README and landing page updated with a
  v0.3.0 section that keeps the ethics / research-use posture and frontier caveat
  prominent.

## [0.2.0] — 2026-06-15

injectkit grows from a single pass/fail scan into a reproducible **robustness
benchmark** — still offline-first, still benign-canary based, still
**defensive / authorized-use only**.

### Added

- **`injectkit bench` command** — sweep the corpus across transforms and defenses
  and emit a per-technique, per-defense **attack-success-rate (ASR)** scorecard
  with a reproducibility stamp (tool version, corpus hash, seed). Shares the
  robustness flags with `scan`. See `docs/BENCHMARK.md`.
- **`injectkit gui` command** — point-and-shoot launch of the local web UI in
  your browser (`injectkit gui`, with `--host` / `--port` / `--no-open`). The
  existing `python -m injectkit.web` entry point still works.
- **Offline local-model targets** — point injectkit at a self-hosted or
  in-process model with no API key: `ollama` (local Ollama server),
  `openai` (OpenAI-compatible local server: vLLM / LM Studio / etc.), and `hf`
  (in-process HuggingFace Transformers). The core and new targets stay
  offline-first with lazy-imported heavy SDKs.
- **Attack transforms** — canary-preserving, deterministic obfuscation modifiers
  (`base64`, `rot13`, `hex`, `leetspeak`, `homoglyph`, `zero_width`, `reversed`,
  `split`, plus the `identity` baseline) layered onto base attacks to stress
  input filtering, registered through a process-wide `TransformRegistry` and
  selected with `--mutate`.
- **Multi-turn attacks & strategies** — `crescendo`, `many_shot`,
  `context_overflow`, and `persona_priming` conversational strategies
  (`--multiturn`) scored on the same benign-canary proxy as single-shot attacks.
- **Adaptive attacker** — a local-model-first attacker (`--adaptive`, default
  backend a local Ollama server) that optimizes attack *structure* (never
  harmful content) against the benign proxy, graded by the detectors.
- **Defenses** — pluggable mitigations selected with `--defense`
  (`hardened_system`, `sandwich`, `input_sanitizer`, `output_filter`, plus the
  `none` baseline) measured as ASR-with vs ASR-without via `defense_delta`.
- **ASR benchmark data model** — `injectkit/benchmark.py` (`ASRCell`,
  `BenchmarkRunMetadata`, `BenchmarkResult`): pure data, no heavy imports, no
  network. Rollups by technique and by defense with a reproducibility stamp.
- **Research-dataset interface** — opt-in, gated references to the official
  academic datasets (AdvBench, HarmBench, JailbreakBench, In-The-Wild
  Jailbreaks, Tensor Trust) by name + URL only; never bundled. Gated behind
  `--research-benchmark DATASET` **plus** `--i-am-authorized`; downloads from
  each dataset's own source and prints the research-use disclaimer.
- **Packaging extras** — new lazy-imported extras `injectkit[ollama]`,
  `injectkit[openai]`, `injectkit[hf]`, and `injectkit[research]` alongside the
  existing `anthropic` / `mcp` / `all`.
- **Docs** — new `docs/BENCHMARK.md` (ASR methodology + reproduction); expanded
  `docs/TAXONOMY.md` and `docs/RESEARCH-USE.md`; README and landing page updated
  for the v0.2.0 capabilities and a prominent research-use / ethics section.

### Fixed

- **Honest grading for unreachable targets** — when every attack errors (wrong
  URL, missing API key, unreachable host) the scan is no longer scored as a
  clean `A`/all-defended. `ScanReport` now distinguishes `passed` (genuinely
  defended) from `errored` (no usable response) and exposes `all_errored`. The
  terminal, HTML, and Markdown reports render the grade as `N/A` with a
  "target unreachable" notice when every attack errored, surface an errored
  count when only some errored, and the CLI prints a stderr warning so errors
  are never silently counted as defenses. The web GUI shows the same.

## [0.1.0] — 2026-06-15

Initial public release. injectkit red-teams LLM applications you own or are
authorized to test for prompt injection, then reports which attacks got through.
**Defensive / authorized-use only.**

### Added

- **Data-driven attack corpus** — 36 attacks across 6 techniques, one YAML file
  each: `direct_injection`, `indirect_injection`, `jailbreak`,
  `system_prompt_leak`, `tool_abuse`, `data_exfiltration`. The community grows
  the corpus by PRing YAML; no Python required.
- **Engine** — orchestrates the scan: loads the corpus, renders each attack's
  per-run `{canary}`, sends it to the target, evaluates the response, and
  aggregates a `ScanReport`.
- **Targets** — pluggable `Target` protocol with adapters for a generic HTTP
  chat endpoint (configurable request template + JSONPath reply extraction), the
  Anthropic Messages API (default `claude-opus-4-8`), and MCP servers / agents
  (tool-abuse + exfiltration). A deterministic built-in `MockTarget` powers the
  offline demo and the test suite.
- **Detectors** — offline heuristics (marker/canary echo, refusal detection,
  system-prompt-leak markers, regex success conditions) plus an optional
  Anthropic LLM judge (default `claude-haiku-4-5`) for subtler successes such as
  paraphrased system-prompt leaks and partial compliance. Refusals are scored as
  the defender successfully winning, not as findings.
- **Scoring** — combines detector verdicts into a success boolean, severity, and
  confidence per attack.
- **CLI** — `injectkit scan` / `list` / `init`, with `--target`, `--url`,
  `--model`, `--judge` / `--judge-model`, `--fail-on`, `--format`, and `--out`.
  Exit codes: `0` clean, `1` a finding met `--fail-on`, `2` a setup/run error.
- **Reporters** — `terminal`, `json`, `markdown`, `sarif`, and `html` output.
  Every report embeds the authorized-use notice.
- **GitHub Action** (`action.yml` + Docker entrypoint) — gate pull requests on
  injection regressions and upload SARIF to the repository's Security tab.
- **Configuration** — `.injectkit.yaml` config loader merged with CLI flags into
  a typed config; `injectkit init` scaffolds a starter file.
- **Packaging** — installable from PyPI with lazy-imported optional extras:
  `injectkit[anthropic]`, `injectkit[mcp]`, `injectkit[all]`. The core CLI works
  with no optional SDKs installed.
- **Docs & policy** — README quickstart and how-it-works, a GitHub Pages landing
  page (`docs/index.html`), `SECURITY.md` (authorized-use notice +
  responsible-disclosure), and `CONTRIBUTING.md` (attack-YAML schema).
- **CI** — test workflow plus a self-scan workflow that runs injectkit against
  its own bundled demo target.

[Unreleased]: https://github.com/Dukotah/injectkit/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Dukotah/injectkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Dukotah/injectkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dukotah/injectkit/releases/tag/v0.1.0
