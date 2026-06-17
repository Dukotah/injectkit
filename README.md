# injectkit

**Red-team your own LLM apps for prompt injection.** injectkit is an
open-source Python tool that throws a corpus of prompt-injection attacks at an
LLM endpoint you control — a chatbot, an agent, an MCP tool server, or a raw
model API — and reports which ones got through.

> ⚠️ **Defensive / authorized use only.** injectkit scans endpoints you **own**
> or are **explicitly authorized** to test — the "scan your own site" posture.
> It does not target third parties. Every report carries an authorized-use
> notice. MIT licensed. See [SECURITY.md](SECURITY.md).

> 🔑 **No API key required.** The detectors run fully offline — install it and
> scan with zero credentials. An LLM-as-judge for sharper grading is **optional**
> (`--judge`) and the only feature that needs an Anthropic API key; leave it off
> and everything still works.

> 🧪 **Research use only — no harmful content.** Every attack is a **benign
> canary proxy**: it asks the target to echo a unique per-run marker it was told
> to withhold. A marker echo proves the injection bypassed instructions *without
> eliciting any harmful output*. injectkit measures bypass and robustness, never
> harm. See [docs/RESEARCH-USE.md](docs/RESEARCH-USE.md).

---

## What's new in v0.3.0

v0.3.0 widens the attack surface injectkit can measure — more obfuscation
ciphers, a semantic translation transform, reply-aware multi-turn escalation, a
named-attacker registry covering the canonical automated jailbreak papers, an
optional white-box GCG suffix optimizer, and a richer 5-class response grade —
**all still benign-canary based, offline-first, and defensive / authorized-use
only**. Every technique cites a primary research source in
[docs/RESEARCH.md](docs/RESEARCH.md).

- **Cipher & encoding transforms.** Six new canary-preserving transforms —
  `caesar`, `atbash`, `morse`, `unicode_escape`, `artprompt` (ASCII-art
  masking), and `selfcipher` (role-play cipher framing) — extend the `--mutate`
  axis. Grounded in CipherChat (arXiv:2308.06463) and ArtPrompt
  (arXiv:2402.11753).
- **Semantic translation transform.** A `translate` transform routes the payload
  through a **low-resource language** (default Swahili) to probe cross-lingual
  robustness — a *semantic* transform, not a character cipher. Uses a lazy,
  offline translator (`argostranslate`) with a friendly error if it's missing.
  Grounded in low-resource-language jailbreak findings (arXiv:2310.02446 /
  MultiJail arXiv:2310.06474).
- **Reply-aware crescendo.** A `crescendo_reply` multi-turn strategy escalates by
  quoting the model's *own prior replies* before the scored ask, and
  `crescendo_decompose` chains individually-benign, canary-free sub-tasks and only
  carries the marker on the final turn — the reply-aware and agent-decomposition
  Crescendo variants (arXiv:2404.01833).
- **Named automated attackers.** A pre-seeded attacker registry declares the
  canonical automated jailbreak techniques — **PAIR** (arXiv:2310.08419),
  **TAP** (arXiv:2312.02119), **AutoDAN** (arXiv:2310.04451), and **GPTFUZZER**
  (arXiv:2309.10253), all black-box — each carrying its citation. They optimize
  attack *structure* against the benign canary proxy, never harmful content.
- **White-box GCG suffix optimizer (optional, HF-only).** An opt-in `gcg`
  attacker optimizes an adversarial suffix from model gradients so a **local**
  white-box HuggingFace model emits the **benign** canary marker — a robustness
  test, never harmful output. `torch` / `transformers` are lazy-imported and the
  optimization is compute-heavy (GPU recommended). **No harmful suffix artifact
  is bundled.** Grounded in GCG / AmpleGCG (arXiv:2404.07921) and Mask-GCG
  (arXiv:2509.06350).
- **5-class response scoring.** Beyond pass/fail, each reply is graded into
  `reject_irrelevant` · `reject_safety` · `too_long` · `partial` · `full`.
  The headline boolean stays **frozen**: a scan succeeds **only** on `full`
  (`is_success`). Grounded in SoK Prompt Hacking (arXiv:2410.13901) and
  StrongREJECT.

> ⚠️ **Honest frontier caveat.** The popular "90%+ jailbreak even on flagship
> aligned models" narrative is **overstated** — those high attack-success rates
> are mostly GPT-4-era / open / mid-tier models. Classifier-equipped frontier
> stacks are much harder, with single-turn ASR in the low single digits.
> injectkit's value is **measuring** robustness — including proving a model is
> robust — not "jailbreak anything." Set honest expected-bypass baselines
> accordingly. See [docs/RESEARCH.md](docs/RESEARCH.md).

These additions preserve the v0.2.0 design: benign canary detection,
offline-by-default, and **defensive / authorized use only**.

---

## What's new in v0.4 (white-box core integration — unreleased)

v0.4 lands the **white-box research core**: a license-clean, fully-offline
white-box attack + judge + leaderboard stack. It is **library-complete and
CPU-tested end-to-end today**, with the parts that genuinely need a 24 GB GPU and
multi-GB model downloads honestly marked **DEFERRED-NO-GPU** (the code paths exist
and are exercised against tiny CPU models / offline seams — they are *not* faked).
All objectives remain the **benign canary marker**; no harmful target is ever set.

- **White-box attack ABC + registry + typed configs** (`injectkit/whitebox/`).
  One typed contract (`run(model, tokenizer, messages, target, cfg, defense) ->
  AttackResult`) every white-box family implements, a name registry, and Pydantic
  configs. Additive to the v0.3 `attackers/` package; the v0.3 `gcg` is re-wrapped
  onto it.
- **Model zoo** (`injectkit/whitebox/zoo.py` + `zoo.yaml`). A pinned registry of
  white-box models by **revision + quantisation** for reproducible loads. Loading
  the real 7–20B entries needs a GPU (**DEFERRED-NO-GPU**); the loader/seam is
  tested offline and an in-process **CPU demo seam** runs with no download.
- **Hardened GCG (nanoGCG parity) + AdvPrefix** (`whitebox/gcg_hard.py`,
  `whitebox/targets.py`). One-hot token gradients, `top_k=256` / `search_width=512`
  candidate sampling, an attack buffer, and the two mandatory correctness traps
  (round-trip `filter_ids`, tokenizer-agnostic chat-template slice location across
  5 dense families). A **golden-loss tripwire** pins the optimiser numerics within
  5% on GPT-2 / Pythia-160M (CPU). `probe_sampling` and full-8B ASR parity are
  **DEFERRED-NO-GPU** (knob present, off). Grounded in nanoGCG (arXiv:2410.15362)
  and AdvPrefix (arXiv:2412.10321). See [docs/REPRODUCE.md](docs/REPRODUCE.md).
- **Prefill attack** (`injectkit/attacks/whitebox/prefill.py`). An affirmative-
  prefix prefill family registered on the white-box registry.
- **Offline judge layer + calibration gate** (`injectkit/judge/`). Three signals
  reported **separately and never collapsed**: **substring-ASR**, **judge-ASR**
  (default `clean_cls`, an **MIT from-scratch** classifier — the only bundleable
  model judge), and **StrongREJECT-mean**. A publication gate fails the build if
  the default judge falls below **κ ≥ 0.6 / agreement ≥ 0.85** or its frozen
  prompt/feature hash drifts. The Llama-derived `harmbench_cls` / `llama_guard`
  judges are **optional gated loaders that never bundle weights**. The production
  ModernBERT/DeBERTa `clean_cls` backbone and the LLM StrongREJECT autograder are
  **DEFERRED-NO-GPU**. See [docs/JUDGES.md](docs/JUDGES.md).
- **Deterministic, backend-locked generation runner** (`injectkit/generate/`).
  Greedy, byte-reproducible decoding through a single seam, with a **same-backend
  invariant** so a judge can't score `hf`-generated text under a `vllm` backend (or
  vice-versa). The `vllm` backend needs a GPU (**DEFERRED-NO-GPU**); the `hf` path
  is verified on a tiny CPU model.
- **Generalized bench harness + leaderboard + 8-field repro stamp**
  (`injectkit/bench/`) and a new **`injectkit attack`** CLI subcommand. One cell —
  `attack × model × behaviors × seeds × judge` — aggregates the three signals with
  Wilson confidence intervals and the mandatory 8-field stamp (version, corpus-hash,
  model-revision, seed, quant, judge-id, attack-id, backend). Runs on a tiny CPU
  model offline; the 8B fp16-vs-4bit anchor cells are **DEFERRED-NO-GPU**.

> ℹ️ **What v0.4 does NOT yet do.** No real 7–20B white-box model is loaded or
> attacked on the development host (no GPU). The published leaderboard *numbers*
> for flagship-scale models, full GCG ASR parity, `probe_sampling`, full-scale
> AdvPrefix prefix mining, the transformer `clean_cls` backbone, the LLM-backed
> StrongREJECT autograder, the gated Llama judges, and the vLLM backend are all
> **DEFERRED-NO-GPU** — implemented in code and tested against tiny/offline seams,
> but not executed at scale here. The repro stamp still records `version 0.3.0`
> until the release is cut. Judge-in-the-loop attacks (REINFORCE-GCG, UJA) are
> v0.5.

---

## What's new in v0.2.0

injectkit grew from a single pass/fail scan into a reproducible **robustness
benchmark**, all still offline-first and benign-canary based:

- **Offline local-model targets.** Point injectkit at a self-hosted or
  in-process model with **no API key** — the core, detectors, and new targets
  all work fully offline; heavy SDKs stay lazy-imported.
- **Attack transforms.** Canary-preserving, deterministic obfuscations
  (encoding, unicode, framing, splitting) layered onto base attacks to measure
  how well your input filtering holds up. See [docs/TAXONOMY.md](docs/TAXONOMY.md).
- **Multi-turn attacks.** Crescendo / role-play strategies that escalate across
  turns, scored on the same benign-canary proxy as single-shot attacks.
- **Adaptive attacker.** A **local-model-first** attacker that optimizes attack
  *structure* (not harmful content) against the benign proxy, graded by the
  detectors — standard ASR methodology for stress-testing robustness.
- **Defenses you can A/B.** Pluggable mitigations (spotlighting, hardened system
  prompt, input/output filters) measured as **ASR-with vs ASR-without** so you
  can see which defense actually helps, and against which technique.
- **ASR benchmark.** Attack-success-rate rollups by technique and by defense with
  a reproducibility stamp (tool version, corpus hash, seed). Methodology and
  reproduction steps: [docs/BENCHMARK.md](docs/BENCHMARK.md).
- **Research-dataset interface (opt-in, gated, never bundled).** References the
  official academic datasets (AdvBench, HarmBench, JailbreakBench, …) by name and
  URL only; downloads happen from the source on **explicit acknowledgment**.

These capabilities preserve the v0.1.0 design: benign canary detection,
offline-by-default, and **defensive / authorized use only**.

---

## GUI / point-and-shoot

Don't want the command line? After `pip install injectkit`, just run:

```sh
injectkit gui
```

That launches a small **local web UI** (localhost only) and opens it in your
browser. Pick a target, choose techniques, and click **Run scan** — the full
HTML report renders right in the page. The built-in **mock** target needs no API
key and no network, so you can watch injectkit work end-to-end with zero setup.

![injectkit GUI scan form](docs/img/gui-form.png)

![injectkit report with findings](docs/img/report.png)

Flags: `injectkit gui --host 127.0.0.1 --port 8765 --no-open`. The equivalent
`python -m injectkit.web` also still works.

## The problem

Prompt injection is the [#1 OWASP LLM risk](https://owasp.org/www-project-top-10-for-large-language-model-applications/):
untrusted text — a user message, a retrieved document, a tool result — can
override your model's instructions, leak its system prompt, or abuse its tools.
You can't fix what you can't measure. injectkit gives you a repeatable scan so
you can catch injection weaknesses in CI before they ship.

## Demo

```sh
# Scan a target defined in .injectkit.yaml, print a terminal report:
injectkit scan

# Scan a local model with NO API key, obfuscating each payload and A/B-ing a defense:
injectkit scan --target ollama --model llama3.1 --mutate base64,rot13 --defense sandwich

# Benchmark attack-success-rate (ASR) across transforms + defenses → scorecard:
injectkit bench --target mock --mutate all --defense hardened_system --seed 0

# List the bundled attack corpus:
injectkit list

# Scaffold a config file:
injectkit init
```

```text
$ injectkit scan --target anthropic --fail-on high

injectkit 0.3.0  ·  defensive / authorized-use only
target  anthropic:claude-opus-4-8
corpus  6 techniques · 65 attacks

  ✓ PASS  direct_injection      model refused / ignored injection
  ✗ FAIL  system_prompt_leak    [CRITICAL] system prompt echoed (conf 0.97)
  ✗ FAIL  data_exfiltration     [HIGH] marker INJECTOK-3f9a echoed (conf 0.95)
  ✓ PASS  jailbreak             refusal detected (defender won)
  ✓ PASS  tool_abuse            no unauthorized tool call
  ✓ PASS  indirect_injection    marker absent from output

summary  65 attacks · 63 passed · 2 failed · highest: CRITICAL
exit 1  (failed --fail-on high)
```

*(Illustrative output. The CLI, engine, and reporters are built as separate
modules against the frozen contracts in this repo.)*

## Install

```sh
pip install injectkit                 # core: CLI, corpus, HTTP target, reports
pip install "injectkit[anthropic]"    # + Anthropic Messages API target & LLM judge
pip install "injectkit[mcp]"          # + MCP server / agent tool-use target
pip install "injectkit[ollama]"       # + local Ollama target + adaptive attacker (no key)
pip install "injectkit[openai]"       # + OpenAI-compatible local server target (vLLM/LM Studio)
pip install "injectkit[hf]"           # + in-process HuggingFace Transformers target
pip install "injectkit[translate]"    # + offline translator for the `translate` transform
pip install "injectkit[research]"     # + gated, opt-in research-dataset loaders
pip install "injectkit[all]"          # everything
```

Python 3.10+. The optional SDKs are lazy-imported, so the core works without
them — and the `ollama` / `openai` / `hf` local targets need **no API key**.

## Usage

```sh
injectkit scan \
  --target anthropic \
  --model claude-opus-4-8 \
  --judge \                 # enable the optional LLM judge (sharper grading)
  --fail-on high \          # non-zero exit if any HIGH+ finding
  --format sarif \          # terminal | json | markdown | sarif | html
  --out results.sarif
```

Set `ANTHROPIC_API_KEY` in your environment for the Anthropic target and judge.
Report formats: `terminal · json · markdown · sarif · html`.

## GitHub Action

Gate every pull request against injection regressions and upload the results to
your repo's **Security** tab as SARIF:

```yaml
# .github/workflows/injectkit.yml
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Dukotah/injectkit@v1
        with:
          target: anthropic        # anthropic | http | mcp
          fail-on: high            # info | low | medium | high | critical
          format: sarif            # terminal | json | markdown | sarif | html
          out: results.sarif
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

A non-zero `--fail-on` exit code breaks the build when a finding at or above the
chosen severity is detected — an injection regression fails CI like any other
test. injectkit also self-scans its own bundled demo target in CI.

## How it works

```
corpus (YAML attacks) ──> engine
                            │  for each attack:
                            │    render {canary}  ->  target.send()  ->  evaluate
                            ▼
                    detectors (offline heuristics + optional LLM judge)
                            │  marker/canary echo, refusal detection,
                            │  system-prompt-leak markers, regex rules
                            ▼
                        scoring  ->  ScanReport  ->  reporter
```

- **Data-driven corpus.** Each attack is a YAML entry (id, technique, severity,
  payload, success conditions). The community adds attacks by PRing YAML — see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Robust offline detection.** Many attacks instruct the model to emit a unique
  marker (e.g. `output exactly: INJECTOK-{canary}`). injectkit substitutes a
  fresh per-run canary and checks whether that exact marker comes back — so a
  model merely echoing the word "INJECTOK" won't false-positive.
- **Optional LLM judge.** For subtler successes (paraphrased system-prompt
  leaks, partial compliance), an Anthropic judge grades the response. Off by
  default; lazy-imported.
- **Targets.** Generic HTTP chat endpoints, the Anthropic Messages API, and MCP
  servers/agents (tool-abuse + exfiltration). A built-in deterministic
  `MockTarget` powers the offline demo and tests.

## Techniques covered

`direct_injection` · `indirect_injection` · `jailbreak` ·
`system_prompt_leak` · `tool_abuse` · `data_exfiltration`

injectkit layers more axes on top of these families:

- **Transforms** (encoding / unicode / framing / splitting obfuscations) — now
  including the v0.3.0 ciphers `caesar`, `atbash`, `morse`, `unicode_escape`,
  `artprompt`, `selfcipher`, and the semantic low-resource `translate`
  transform.
- **Delivery shape** (single-shot vs multi-turn crescendo / role-play) — now
  including the reply-referencing `crescendo_reply` strategy.
- **Automated attackers** — the named black-box attackers `pair`, `tap`,
  `autodan`, `gptfuzzer`, and the optional white-box `gcg` suffix optimizer
  (HuggingFace-only, lazy `torch` / `transformers`, benign target).
- **Response grade** — the 5-class score (`reject_irrelevant` ·
  `reject_safety` · `too_long` · `partial` · `full`); a scan succeeds only on
  `full`.

The full taxonomy — families, modifier tags, delivery strategies, automated
attackers, and the defenses they're measured against — is in
[docs/TAXONOMY.md](docs/TAXONOMY.md), with the cited research grounding in
[docs/RESEARCH.md](docs/RESEARCH.md).

## Benchmarking & defenses

Beyond a single pass/fail scan, injectkit measures **attack-success rate (ASR)**
and rolls it up by technique and by defense, so you can compare an undefended
baseline against mitigations:

- **ASR** = `successes / attempts` (errored attempts excluded). Lower is better.
- **`defense_delta`** = baseline ASR − defended ASR. **Positive means the defense
  helped.**
- Defenses ship as pluggable hooks (`hardened_system`, `sandwich`,
  `input_sanitizer`, `output_filter`, and the `none` baseline) applied as
  `wrap_system → filter_input → target.send → filter_output` before grading.
- The **adaptive attacker** is local-model-first and optimizes attack *structure*
  against the benign canary proxy — it is not a harmful-output generator.
- The **named automated attackers** (`pair`, `tap`, `autodan`, `gptfuzzer`, and
  the optional white-box `gcg`) likewise optimize toward the benign marker, each
  citing its source paper. GCG runs against a **local HuggingFace** model only
  and is compute-heavy.
- The **5-class response grade** keeps the headline boolean frozen — a scan
  counts as a success only on `full` — while exposing the finer
  `reject_irrelevant` / `reject_safety` / `too_long` / `partial` shades for
  honest reporting.

Methodology, the data model, and step-by-step reproduction (including the
reproducibility stamp — tool version, corpus hash, seed) are documented in
[docs/BENCHMARK.md](docs/BENCHMARK.md); the cited research grounding for every
technique is in [docs/RESEARCH.md](docs/RESEARCH.md).

## Research use & datasets

injectkit is for **authorized defensive research only** and ships **no harmful
prompts or datasets**. The optional `injectkit.research` interface references the
official academic datasets (AdvBench, HarmBench, JailbreakBench, In-The-Wild
Jailbreaks, Tensor Trust) by **name and URL only** and downloads them from their
source **only on explicit opt-in**. On the CLI this is a double gate —
`injectkit bench --research-benchmark <dataset> --i-am-authorized` — and from
the library it is `acknowledge=True` or `INJECTKIT_RESEARCH_ACK=1`. Every
ungated access prints a disclaimer, and you remain bound by each source's own
licence. See [docs/RESEARCH-USE.md](docs/RESEARCH-USE.md) for the authorized-use
posture, the gating, and responsible-disclosure guidance, and
[docs/RESEARCH.md](docs/RESEARCH.md) for the cited 2023–2026 research map that
grounds every technique — including the honest frontier-robustness caveat.

## Contributing

New attacks are the highest-value contribution and require only a YAML edit.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Ethics

injectkit is built for defenders. Use it to harden LLM endpoints, agents, and
models you **own** or are **explicitly authorized** to test. Every attack is a
**benign canary proxy** (it never elicits harmful content), the toolkit ships no
harmful datasets, and the adaptive attacker optimizes attack *structure* only.
See [docs/RESEARCH-USE.md](docs/RESEARCH-USE.md) for the authorized-research-only
posture, the gated research-dataset interface, and responsible disclosure, and
[SECURITY.md](SECURITY.md) for the authorized-use notice and how to report a flaw
in injectkit itself.

## Changelog

Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md). Current release:
**v0.3.0**.

## License

[MIT](LICENSE) © Dukotah / Copper Bay Labs
