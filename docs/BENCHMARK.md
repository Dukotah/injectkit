# injectkit Benchmark Methodology (v0.3.0)

> DEFENSIVE / AUTHORIZED USE ONLY. The benchmark measures **attack-success rate
> (ASR)** against a target you own or are explicitly authorized to test, using
> the **benign canary proxy** — every "success" is a unique per-run marker the
> target was told to withhold echoing back. It quantifies instruction-bypass
> robustness, never harmful content.

The benchmark turns a single pass/fail scan into a reproducible **measurement**:
how often do attacks bypass instructions, broken down by technique family and by
defense, with a stable reproducibility stamp so two runs of the same setup are
comparable. The data model and rollups live in `injectkit/benchmark.py` (pure
data — no heavy imports, no network).

## 1. What ASR means here

For a group of attempts against a target:

```
ASR = successes / attempts          # attempts that ERRORED are excluded
```

`compute_asr(successes, attempts)` returns `0.0` when `attempts <= 0`. An
**attempt** is one attack rendered with a fresh canary and sent to the target. A
**success** is a graded bypass — the detectors (offline heuristics, optionally
the LLM judge) confirmed the benign marker came back / the planted constraint was
broken. Refusals count as the **defender winning** (not a success). Attempts whose
`TargetResponse.error` is set (unreachable target, timeout) are **excluded from
the denominator** and counted separately as `errored`, so a flaky endpoint can
never be misread as either "defended" or "vulnerable".

This is the standard ASR methodology used across the prompt-injection /
jailbreak literature (AdvBench, HarmBench, JailbreakBench), applied to the benign
canary proxy instead of harmful behaviors.

> **Honest baselines.** Lower ASR is the *defender* winning, and a robust
> frontier model legitimately scores near zero — the survey
> ([`RESEARCH.md`](./RESEARCH.md)) verifies that the "90%+ even on flagship aligned
> models" narrative is **overstated** (those high numbers are GPT-4-era / open /
> mid-tier). injectkit's value is *measuring* robustness — including proving a
> model is robust — so set expected-bypass baselines accordingly.

### Grading: boolean bypass vs 5-class scoring

A "success" is still the boolean bypass the ASR formula counts. v0.3.0 adds an
optional **5-class grade** per reply (`evaluators/response_class.py::ResponseClass`
— `reject_irrelevant` / `reject_safety` / `too_long` / `partial` / `full`; SoK
Prompt Hacking arXiv:2410.13901; StrongREJECT) for richer ASR fidelity. The boolean
remains **derivable** and frozen: `is_success` is True only for `full`, so the ASR
numerator is unchanged. The extra classes explain *why* a non-success happened
(safety refusal vs off-task vs length-capped) without moving the headline. See
[`TAXONOMY.md`](./TAXONOMY.md) Axis 5 for the full mapping.

## 2. The data model (`injectkit/benchmark.py`)

| Type | Role |
|------|------|
| `ASRCell` | One `(group, defense)` measurement: `attempts`, `successes`, `errored`, `highest_severity`, and an `asr` property. `group` is a technique/family name or `"overall"`. Built with `ASRCell.from_results(group, defense, results)`. |
| `BenchmarkRunMetadata` | The reproducibility stamp: `tool_version`, `target_name`/`target_model`, `corpus_hash`, `transforms`, `defenses`, `seed`, `attacker_model`, `used_judge`, `started_at`/`finished_at` (+ `duration_s`). |
| `BenchmarkResult` | The full report: `metadata` + a list of `cells`, with rollup helpers. |

`BenchmarkResult` rollups:

- `overall(defense="none")` / `overall_asr(defense="none")` — the single
  headline number for a defense.
- `by_technique(defense="none")` — a `{technique: ASRCell}` breakdown.
- `defenses()` — the defenses that appear in the cells.
- `defense_delta(defense)` — `baseline("none").asr - defended.asr`. **Positive
  means the defense helped** (it reduced ASR); negative means it made things
  worse.

`ASRCell.from_results` is the bridge from a scan: feed it the `AttackResult`s for
one `(technique, defense)` slice and it tallies attempts/successes/errored and
tracks the worst severity seen.

## 3. The benchmark matrix

A benchmark sweeps three axes (see [`TAXONOMY.md`](./TAXONOMY.md) for the full
taxonomy) and records one `ASRCell` per cell of the resulting grid:

- **Technique families** (the rollup groups): `direct_injection`,
  `indirect_injection`, `jailbreak`, `system_prompt_leak`, `tool_abuse`,
  `data_exfiltration`.
- **Transforms** (obfuscation/restructuring modifiers): `identity` (baseline),
  encodings, unicode tricks, framing, splitting — each a canary-preserving
  `Transform` from `injectkit/transforms/`. v0.3.0 adds the cipher family
  (`caesar` / `atbash` / `morse` / `unicode_escape` / `artprompt` / `selfcipher`
  — CipherChat 2308.06463, ArtPrompt 2402.11753) and the semantic `translate`
  transform (low-resource translation, 2310.02446 / MultiJail 2310.06474), so the
  obfuscation axis now spans byte/char encodings, ciphers, ASCII-art, and a
  *semantic* low-resource translation. See [`TAXONOMY.md`](./TAXONOMY.md) Axis 2.
- **Defenses** evaluated as ASR-with vs ASR-without: `none` (undefended
  baseline) plus the mitigations registered in `injectkit/defenses/`
  (`hardened_system`, `sandwich`, `input_sanitizer`, `output_filter`).

Two further axes are optional inputs to the same grid rather than rollup groups:

- **Delivery** (`--multiturn`): single-shot (default) or a multi-turn strategy,
  including the v0.3.0 `crescendo_reply` (reply-referencing crescendo) and
  `crescendo_decompose` (agent-decomposition crescendo) variants (2404.01833).
- **Attacker** (`--attacker`): the adaptive propose/refine loop, including the
  v0.3.0 named attackers `pair` / `tap` / `autodan` / `gptfuzzer` (black-box) and
  `gcg` (white-box, HF-only). All optimise toward the benign canary; the
  `attacker_model` and `used_judge` flags are stamped into the metadata because
  they change the numbers. See [`TAXONOMY.md`](./TAXONOMY.md) Axis 3/3b.

The interesting numbers are the **deltas**: ASR with a transform vs without
(does the obfuscation get past your filter?) and ASR with a defense vs the `none`
baseline (`defense_delta` — does the mitigation actually help, and against which
technique?).

## 4. Reproducibility

Every benchmark carries a `BenchmarkRunMetadata` stamp so a run is reproducible
and two runs are comparable:

- **`tool_version`** — the injectkit version that produced the numbers.
- **`corpus_hash`** — fingerprints the exact corpus, so a corpus edit is visible.
- **`seed`** — seeds every RNG-bearing transform and the adaptive attacker.
  Transforms are pure/deterministic given their seed, so a fixed seed reproduces
  the exact attack payloads.
- **`transforms` / `defenses`** — the axes that were swept.
- **`attacker_model` / `used_judge`** — which local attacker model generated
  adaptive candidates and whether the LLM judge graded results (both change the
  numbers, so both are stamped).
- **`target_name` / `target_model`** — what was scanned.
- **`started_at` / `finished_at` / `duration_s`** — timing.

Because transforms are deterministic and the corpus is hashed, a benchmark with
the same `tool_version`, `corpus_hash`, `seed`, transforms, defenses, target, and
judge setting reproduces the same ASR cells (modulo target nondeterminism — pin a
local model and a temperature of 0 for bit-stable numbers).

## 5. How to reproduce a benchmark

The `injectkit bench` command drives the whole sweep. Its robustness flags
(`--mutate`, `--defense`, `--multiturn`, `--adaptive`, `--seed`) are shared with
`injectkit scan`, and `--research-benchmark <dataset> --i-am-authorized` swaps in
a gated official dataset (see [`RESEARCH-USE.md`](./RESEARCH-USE.md)).

```sh
# Offline, zero-setup: sweep every transform and A/B a defense against the mock target.
injectkit bench --target mock --mutate all --defense hardened_system --seed 0

# A local model with no API key, single transform, terminal scorecard:
injectkit bench --target ollama --model llama3.1 --mutate base64 --defense sandwich
```

1. **Pick an offline target.** A local/self-hosted model (`--target ollama` /
   `openai` / `hf`) needs no API key. The built-in `mock` target makes the whole
   pipeline run with zero setup.
2. **Choose the axes.** Filter techniques with `--technique` (or run all six),
   pick transforms with `--mutate` (`all` sweeps every built-in; `identity` is
   always measured as the baseline), and compare defenses with `--defense`
   against the `none` baseline.
3. **Fix a seed** (`--seed`) so transforms and the adaptive attacker are
   deterministic and the run is reproducible.
4. **Run the benchmark.** The engine renders each attack (with its canary)
   through each transform, applies each defense's three hooks
   (`wrap_system` -> `filter_input` -> target.send -> `filter_output`), grades
   the filtered output, and tallies an `ASRCell` per `(technique, defense)`.
5. **Read the rollups.** `overall_asr()` for the headline, `by_technique()` for
   where you're weak, and `defense_delta()` for whether each mitigation earned
   its place.

### Reading results

- **Lower ASR is better** — it means more attacks were defended.
- **A positive `defense_delta`** means the defense reduced ASR (it helped); a
  per-technique breakdown shows *which* families it helped, since a defense that
  blocks `direct_injection` may do nothing for `indirect_injection`.
- **Watch `errored`.** A high errored count means the target was flaky — fix
  connectivity before trusting the ASR, since errored attempts are excluded from
  the denominator.
- **Compare like with like.** Only compare benchmarks that share `tool_version`,
  `corpus_hash`, seed, target, and judge setting; the metadata stamp makes
  mismatches obvious.

## 6. Capability-paradox sweep (ASR vs model capability)

> Why this matters. MCPTox (arXiv:2508.14925) found that **more-capable models can
> be *more* susceptible** to tool poisoning — the arms race cannot be won by better
> models alone. injectkit's `capability` mode measures that curve directly: it runs
> one attack across a **set of target models** ordered along a configurable
> **capability axis** and reports ASR-vs-capability with a per-model Wilson CI.

`injectkit capability` (and the library `injectkit.bench.run_capability_sweep`)
generalises the single-cell harness over a model set:

```sh
# Offline, zero-setup: sweep one attack across the synthetic demo capability ladder.
injectkit capability --attack prefill --models demo --seeds 2

# Frontier sweep over the pinned zoo (DEFERRED-NO-GPU — needs a GPU + downloads):
injectkit capability --attack prefill --models zoo --seeds 5 --export-dir out/
```

For each model the sweep runs the existing `run_cell` (same attack registry, judge
layer, generation seam, and the **three never-collapsed signals** — substring-ASR /
judge-ASR / StrongREJECT-mean, each with a Wilson CI and the full **8-field repro
stamp**), records the model's **capability score** (an explicit value, else the
zoo entry's `params_b`), and sorts the points along the capability axis into a
`CapabilityCurve`. The curve exposes:

- **`series()`** — the ordered `(capability, judge-ASR ± CI)` series the plot/table
  consumes.
- **`leaderboard()`** — the model × attack matrix (one column per capability rung),
  rendered to CSV / JSON / Markdown by the existing exporters, every cell carrying
  its 8-field stamp.
- **`verdict()`** — a monotonicity read: `capability_paradox` (ASR rises with
  capability — the MCPTox finding), `inverse` ("bigger is safer"), or `flat`.

> **Honest caveat (read this).** The verdict is an **indicative** read of a handful
> of seeded points, **not** a significance test, and the "90%+ even on flagship
> aligned models" narrative remains overstated (see [`RESEARCH.md`](./RESEARCH.md)).
> A robust frontier model legitimately scores near zero. The value is *measuring*
> the curve — including proving the paradox does *not* hold for your stack — not
> manufacturing it.

### DEFERRED-NO-GPU — the actual frontier run

The offline `--models demo` path is the deterministic CPU done-check (a synthetic
capability ladder over the offline seam; no torch, no download, no API key) and is
what the test suite drives. The **actual frontier-model curve** — sweeping the
pinned zoo entries (`llama-3.1-8b`, `qwen2.5-7b`, `gemma-2-9b`, `mistral-7b-v0.3`,
`phi-4`, `gpt-oss-20b`) or the live `anthropic` / `ollama` / `openai` targets —
needs a GPU + multi-GB downloads or API keys and is **DEFERRED-NO-GPU**. The code
path is real (the `ModelSpec.loader` seam is the same one the zoo loader and the
live targets plug into) and is exercised here against tiny/offline models; it is
**not faked**. The one-command step on a GPU host:

```sh
pip install "injectkit[zoo]"          # transformers + accelerate + bitsandbytes
injectkit capability --attack prefill --models zoo --seeds 5 \
  --quant fp16 --export-dir out/      # writes capability.{csv,json,md}
```

## 7. Offline-first & test posture

`injectkit/benchmark.py` is pure data with no heavy imports, so it loads with no
SDKs and no network. The benchmark unit tests construct `AttackResult`s in-memory
and assert the rollups — they never call a model or hit the network. The adaptive
attacker that feeds the benchmark uses a **local** model; its tests use a stub.
The whole benchmark suite runs fully offline and deterministically.

## 8. References

- OWASP Top 10 for LLM Applications — **LLM01: Prompt Injection**.
- The ASR methodology mirrors the prompt-injection / jailbreak benchmark
  literature (AdvBench, HarmBench, JailbreakBench, Tensor Trust). Those datasets
  are **referenced, not bundled**, and load only on explicit opt-in — see
  [`RESEARCH-USE.md`](./RESEARCH-USE.md).
- **MCPTox** (arXiv:2508.14925) — the capability-paradox finding (more-capable
  models can be *more* susceptible to tool poisoning) that the `capability` sweep
  (§6) measures.
