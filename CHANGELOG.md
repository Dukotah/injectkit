# Changelog

All notable changes to **injectkit** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.5 efficiency + frontier attacks (builds on the v0.4 core)

v0.5 completes the white-box stack the v0.4 core started: the **efficiency
primitives** (Probe Sampling; the I-GCG / Faster-GCG / Mask-GCG GCG-family
refinements) and the **objective-frontier, judge-in-the-loop attacks**
(REINFORCE-GCG, UJA) plus the **continuous embedding / soft-prompt attack**. Like
v0.4, it is **library-complete and CPU-tested end-to-end**: every attack registers
on the white-box registry, runs through the `injectkit attack` bench harness on a
pure-Python offline seam (no torch, no download), and is unit-tested. The
**headline ASR / wall-clock NUMBERS** for each paper genuinely need a 24 GB GPU +
a real 7–8B target, so they are marked **DEFERRED-NO-GPU** — the code paths exist
and run on tiny CPU models / offline seams, they are *not* faked, but the
flagship-scale measurement is not run here. Every objective remains the **benign
canary marker**; no harmful target is set, no harmful suffix/soft-prompt artifact
is bundled, and the version string stays `0.3.0` until the release is cut.

### Added (v0.5)

- **Probe Sampling efficiency primitive** (`whitebox/probe_sampling.py`; CHUNK 8,
  arXiv:2403.01251). A drop-in draft-model acceleration wrapper over the shared GCG
  candidate-scoring seam: a cheap draft model pre-ranks the `search_width`
  candidates, the kept fraction is sized *dynamically* by draft↔target agreement,
  and only that probe set is re-scored on the expensive target. With probe sampling
  **off (default)** GCG scores the full batch exactly as before. The headline
  **3.5×–6.3× speedup / non-degraded ASR on Llama-2-7B** is **DEFERRED-NO-GPU**;
  the full path is verified on a tiny CPU draft+target pair.
- **I-GCG + Faster-GCG GCG-family refinements** (`whitebox/igcg.py`,
  `whitebox/faster_gcg.py`; CHUNK 9). Registered as `igcg` (arXiv:2405.21018,
  ICLR 2025 — diverse benign-target templates, auto multi-coordinate update,
  easy-to-hard init) and `faster_gcg` (arXiv:2410.15362 — distance-regularised
  candidate scoring, temperature sampling, visited-set dedup). Both reuse the
  proven GCG coordinate-descent core verbatim and are torch-free / CPU-unit-tested;
  their full-scale efficiency/ASR numbers are **DEFERRED-NO-GPU**.
- **Optional GCG variant tier** — `mask_gcg` (arXiv:2509.06350, token-position
  pruning) plus flag-gated momentum / MAGIC / SM-GCG primitives
  (`whitebox/gcg_variants.py`; arXiv:2405.01229 / arXiv:2412.08615). They ship as
  **flags on `GCGConfig`, never blockers**; with every flag at default the
  behaviour is byte-for-byte plain GCG. Numbers **DEFERRED-NO-GPU**.
- **Continuous embedding / soft-prompt attack** (`whitebox/embedding.py`; CHUNK 10,
  arXiv:2402.09063, NeurIPS 2024). Registered as `embedding`: optimises the input
  embeddings directly (Adam on a `ℝ^{k×d}` soft prompt, no discrete projection) as
  the **capability ceiling** — what full weight + embedding access can reach. Uses a
  small additional `EmbeddingModel` seam (`HFEmbeddingModel` in production, a
  pure-Python from-scratch-Adam fallback on CPU). The **embedding-ASR ≥ GCG-ASR at
  lower wall-clock on an 8B model** claim is **DEFERRED-NO-GPU**; the optimiser
  converges on a tiny CPU model.
- **Objective-frontier, judge-in-the-loop attacks** (`whitebox/reinforce_gcg.py`,
  `whitebox/uja.py`, `whitebox/objective_judge.py`; CHUNK 11). `reinforce_gcg`
  (arXiv:2502.17254, ICML 2025) replaces GCG's fixed-target NLL with a REINFORCE
  objective over sampled completions graded by a small **in-loop** judge; `uja`
  (arXiv:2510.02999) drops the affirmative target entirely and maximises the
  in-loop judge's score directly. A **test-enforced circularity firewall** requires
  the in-loop OPT judge to differ from the leaderboard EVAL judge
  (`opt_judge_id != eval_judge_id`), asserted at config construction and again
  before the in-loop judge loads. Real completion sampling from a 7–8B target and
  the ASR-parity numbers are **DEFERRED-NO-GPU**; the seam wiring + judge plumbing
  run offline on the stub seam.
- **Bench-harness + CLI integration** — all eight white-box attacks (`gcg`,
  `igcg`, `faster_gcg`, `mask_gcg`, `prefill`, `embedding`, `reinforce_gcg`, `uja`)
  resolve from `injectkit.whitebox.registry` and run end-to-end through
  `injectkit attack --attack <key>` on the offline CPU demo seam. The demo seam now
  satisfies the prefill/generation, discrete-white-box, and continuous-embedding
  seam contracts, so every registered attack drives the full
  registry → seam → judge → aggregate → 8-field-stamp path with no GPU.

## v0.4 white-box core integration (Unreleased)

The **white-box research core**: a license-clean, fully-offline white-box attack +
judge + leaderboard stack, **library-complete and CPU-tested end-to-end**. Parts
that genuinely require a 24 GB GPU + multi-GB downloads are marked
**DEFERRED-NO-GPU** — their code paths exist and are exercised against tiny CPU
models / offline seams, but are not executed at scale on the development host.
Every objective is still the **benign canary marker**; no harmful target is set,
and **no harmful artifact or Llama-derived weight is bundled**. Still offline-first
and **defensive / authorized-use only**.

### Added (v0.4)

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
- **Docs** — `docs/BASELINE.md` (v0.3 recon that gates all v0.4 work),
  `docs/REPRODUCE.md` (GCG parity / tolerance bands / honesty ledger), and
  `docs/JUDGES.md` (judge licences, calibration floor, frozen hashes — assertions
  enforced by the test suite).

### Deferred (honest, DEFERRED-NO-GPU — implemented, not executed at scale here)

- Loading/attacking any real 7–20B white-box model; published leaderboard numbers
  for flagship-scale models.
- Full GCG ASR parity (±10 abs pp) vs reference GCG; full-scale 8B AdvPrefix
  prefix mining. (The `probe_sampling` draft-model loop landed in v0.5 above; only
  its 8B speedup NUMBER stays DEFERRED-NO-GPU.)
- The transformer `clean_cls` backbone; the LLM-backed StrongREJECT autograder;
  the gated `llama_guard` / `harmbench_cls` 8B/13B loads; the vLLM backend.
- The version string remains `0.3.0` (repro stamp included) until the release is
  cut. (The judge-in-the-loop attacks REINFORCE-GCG and UJA that were scoped to
  v0.5 are now implemented and registered — see the v0.5 section above; only their
  GPU-scale ASR numbers remain DEFERRED-NO-GPU.)

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
