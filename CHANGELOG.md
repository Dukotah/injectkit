# Changelog

All notable changes to **injectkit** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.5 judge-in-the-loop attackers

**Judge-in-the-loop white-box attackers** on the v0.4 ABC/registry: the offline judge
layer becomes the **in-loop reward signal** that steers token optimisation, not just a
post-hoc grader. **Library-complete and CPU-tested offline** against tiny / offline
seams + the deterministic mock judge; the at-scale run (real 7–20B + a real judge
reward) is **DEFERRED-NO-GPU** — code path exists and is exercised, not faked. Every
objective remains the **benign canary marker**; no harmful target is set, sampled, or
rewarded. Still offline-first and **defensive / authorized-use only**.

### Added

- **REINFORCE-GCG** (`reinforce_gcg`, `injectkit/attacks/whitebox/judge_loop.py`) —
  judge-in-the-loop GCG. The per-candidate objective becomes `nll − reward_weight ·
  reward`, where the reward is the in-loop judge's StrongREJECT-style score of the
  model's *own generated continuation* (the adaptive/distributional/semantic
  objective). At `reward_weight = 0` it reduces exactly to plain GCG (the golden-loss
  tie-in). **Reuses** the hardened `gcg_hard.py` machinery + the proven GCG inner loop
  — no duplicated optimiser. Grounded in REINFORCE-GCG (arXiv:2502.17924).
- **UJA — Universal Jailbreak Adversarial** (`uja`, same module) — optimises one
  **universal** suffix across a *set* of benign-canary behaviors, re-ranked by the
  **mean** in-loop judge reward across the batch; success requires transfer to a
  majority of behaviors. Grounded in the universal/transferable GCG objective
  (arXiv:2307.15043 §universal).
- **Optimisation-judge ≠ evaluation-judge circularity firewall**
  (`assert_opt_judge_distinct` / `OptJudgeCircularityError`) — the in-loop OPT judge
  (`substring`) must differ from the leaderboard EVAL judge (`clean_cls`) so the
  optimiser cannot game its own grader. Grounded in the judge-circularity finding
  (arXiv:2502.11910). See `docs/JUDGES.md` §6.
- **Typed configs** `ReinforceGCGConfig` / `UJAConfig` (extend `GCGConfig`) with the
  REINFORCE/universal knobs (`opt_judge_id`, `reward_weight`, `num_samples` /
  `behaviors_per_step`, `judge_n_tokens`).
- **Registry + CLI + bench surface** — both attacks register on the white-box attack
  registry and run through `injectkit attack` / `injectkit capability`
  (`--attack reinforce_gcg` / `--attack uja`) and the bench harness, fully offline on
  the demo seam (the demo seam now also exposes the gradient seam). Dense-only
  (gradient family); the zoo's dense models list them under `supported_attacks`.

### DEFERRED-NO-GPU

- Real 7–20B model + a real judge as the reward signal; the full sampled REINFORCE
  distributional estimate; universal-transfer ASR over a large held-out behavior set.
  Implemented in code and exercised against tiny/offline seams + the deterministic mock
  judge — not executed at scale on this no-GPU host. The honest frontier-robustness
  caveat is preserved (`docs/RESEARCH.md`, `docs/REPRODUCE.md` §5b).

## [Unreleased] — v0.4 white-box core integration

The **white-box research core**: a license-clean, fully-offline white-box attack +
judge + leaderboard stack, **library-complete and CPU-tested end-to-end**. Parts
that genuinely require a 24 GB GPU + multi-GB downloads are marked
**DEFERRED-NO-GPU** — their code paths exist and are exercised against tiny CPU
models / offline seams, but are not executed at scale on the development host.
Every objective is still the **benign canary marker**; no harmful target is set,
and **no harmful artifact or Llama-derived weight is bundled**. Still offline-first
and **defensive / authorized-use only**.

### Added

- **White-box attack ABC + registry + typed configs** (`injectkit/whitebox/`) — a
  single typed `Attack` contract (`run(model, tokenizer, messages, target, cfg,
  defense) -> AttackResult`), a name registry, and Pydantic configs. Additive to
  the v0.3 `attackers/` package; the v0.3 GCG is re-wrapped onto it.
- **Model zoo** (`injectkit/whitebox/zoo.py` + `zoo.yaml`) — white-box models
  pinned by revision + quantisation for reproducible loads, with an offline CPU
  demo seam. Real 7–20B loads are **DEFERRED-NO-GPU** (loader/seam tested offline).
- **Hardened GCG (nanoGCG parity) + AdvPrefix** (`whitebox/gcg_hard.py`,
  `whitebox/targets.py`) — one-hot token gradients, `top_k=256` /
  `search_width=512` candidate sampling, an attack buffer, and two mandatory
  correctness traps (round-trip `filter_ids`; tokenizer-agnostic chat-template
  slice location across Llama-3 / Qwen2.5 / Gemma-2 / Mistral-v0.3 / Phi-4). A
  **golden-loss tripwire** pins optimiser numerics within 5% on GPT-2 / Pythia-160M
  (CPU). `probe_sampling` and full-8B ASR parity are **DEFERRED-NO-GPU** (knob
  present, off by default). Grounded in nanoGCG (arXiv:2410.15362) and AdvPrefix
  (arXiv:2412.10321). See `docs/REPRODUCE.md`.
- **Prefill attack** (`injectkit/attacks/whitebox/prefill.py`) — an affirmative-
  prefix prefill family registered on the white-box registry.
- **Offline judge layer + calibration gate** (`injectkit/judge/`) — three signals
  reported **separately and never collapsed**: substring-ASR, judge-ASR (default
  `clean_cls`, an **MIT from-scratch** classifier, the only bundleable model
  judge), and StrongREJECT-mean. A publication gate fails the build below
  **κ ≥ 0.6 / agreement ≥ 0.85** or on a frozen prompt/feature-hash drift
  (`assert_calibrated`). The Llama-derived `harmbench_cls` / `llama_guard` judges
  are **optional gated loaders that never bundle weights** and error gracefully
  without HF auth. The production ModernBERT/DeBERTa `clean_cls` backbone and the
  LLM StrongREJECT autograder are **DEFERRED-NO-GPU**. See `docs/JUDGES.md`.
- **Deterministic, backend-locked generation runner** (`injectkit/generate/`) —
  greedy, byte-reproducible decoding through one seam, with a same-backend
  invariant so a judge cannot score `hf`-generated text under a `vllm` backend (or
  vice-versa). The `vllm` backend is **DEFERRED-NO-GPU**; the `hf` path is verified
  on a tiny CPU model.
- **Generalized bench harness + leaderboard + 8-field repro stamp**
  (`injectkit/bench/`) and a new **`injectkit attack`** CLI subcommand — one cell
  (`attack × model × behaviors × seeds × judge`) aggregates the three signals with
  Wilson confidence intervals and the mandatory 8-field stamp (version,
  corpus-hash, model-revision, seed, quant, judge-id, attack-id, backend; `quant`
  never blank). Runs on a tiny CPU model offline; 8B fp16-vs-4bit anchor cells are
  **DEFERRED-NO-GPU**.
- **Capability-paradox bench harness + `injectkit capability` CLI subcommand**
  (`injectkit/bench/capability.py`) — runs one attack across a **set of target
  models** ordered along a configurable capability axis (default: `params_b`) and
  emits the **ASR-vs-capability curve** + a model × attack leaderboard, with a
  per-model Wilson CI and the full 8-field stamp on every point, plus a
  monotonicity verdict (`capability_paradox` / `inverse` / `flat`). Surfaces the
  **MCPTox** finding (arXiv:2508.14925) that more-capable models can be *more*
  susceptible to tool poisoning — the portfolio's validation of deterministic
  enforcement. The offline demo ladder runs fully on CPU (mock-seam driven, what
  the test suite exercises); the **actual frontier sweep** over the pinned zoo /
  live `anthropic`/`ollama`/`openai` targets is a documented one-command step
  marked **DEFERRED-NO-GPU** (loader seam real and exercised offline, not faked).
  See `docs/BENCHMARK.md` §6.
- **Docs** — `docs/BASELINE.md` (v0.3 recon that gates all v0.4 work),
  `docs/REPRODUCE.md` (GCG parity / tolerance bands / honesty ledger), and
  `docs/JUDGES.md` (judge licences, calibration floor, frozen hashes — assertions
  enforced by the test suite).

### Deferred (honest, DEFERRED-NO-GPU — implemented, not executed at scale here)

- Loading/attacking any real 7–20B white-box model; published leaderboard numbers
  for flagship-scale models.
- Full GCG ASR parity (±10 abs pp) vs reference GCG; `probe_sampling` draft-model
  loop; full-scale 8B AdvPrefix prefix mining.
- The transformer `clean_cls` backbone; the LLM-backed StrongREJECT autograder;
  the gated `llama_guard` / `harmbench_cls` 8B/13B loads; the vLLM backend.
- The **actual frontier capability-paradox curve** (`injectkit capability --models
  zoo`) over real 7–20B models / live API targets — the offline demo ladder is
  CPU-tested; the frontier numbers need a GPU + downloads or API keys.
- The version string remains `0.3.0` (repro stamp included) until the release is
  cut. Judge-in-the-loop attacks (REINFORCE-GCG, UJA) are scoped to v0.5.

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
