# injectkit Attack Taxonomy (v0.2.0)

> DEFENSIVE / AUTHORIZED USE ONLY. Every attack below is delivered as a **benign
> canary proxy**: the payload asks the target to emit a unique per-run marker it
> was instructed to withhold. A marker echo proves the injection bypassed the
> instruction — *without eliciting any harmful content*. injectkit measures
> **bypass and robustness**, never harm.

This taxonomy is the contract the expanded v0.2.0 corpus follows. Every attack
declares a `technique` (a top-level family below) and may carry `tags` from the
modifier and delivery axes. The benchmark rolls attack-success-rate (ASR) up by
`technique`, so the families here are the reporting groups — see
[`BENCHMARK.md`](./BENCHMARK.md) for how ASR is computed and reproduced.

## Axis 1 — Technique families (the `technique` field)

| `technique`          | Definition (benign-proxy framing)                                                | Severity floor* |
|----------------------|-----------------------------------------------------------------------------------|-----------------|
| `direct_injection`   | The user prompt directly overrides the system instruction ("ignore previous…").   | low             |
| `indirect_injection` | The override rides in untrusted *context* (retrieved doc, tool output, web page).  | medium          |
| `jailbreak`          | Bypass a planted policy/constraint via persona, fiction, emotion, obfuscation.     | medium          |
| `system_prompt_leak` | Coax the target into disclosing its hidden system prompt / configuration.          | high            |
| `tool_abuse`         | Induce an unauthorized tool call / action the agent should not take.               | high            |
| `data_exfiltration`  | Cause the target to leak a planted secret to an attacker-controlled sink.          | high            |

\* Severity floors are enforced by `evaluators/scoring.py::TECHNIQUE_MIN_SEVERITY`.

### New / expanded families to populate in v0.2.0
- `indirect_injection`: more carriers — markdown/HTML comments, image alt-text,
  CSV/JSON fields, email quoting, RAG chunk boundaries.
- `tool_abuse`: argument injection, confused-deputy, tool-chaining.
- `data_exfiltration`: markdown-image beaconing, URL-encoded sinks, summary
  smuggling. (All sinks are inert canary markers — no real exfiltration.)

## Axis 2 — Modifier techniques (the `tags` field; applied via `transforms/`)

Obfuscations/restructurings layered onto a base attack. Each shipped modifier is
a `Transform` (registered in `injectkit/transforms/`, selected with `--mutate`)
so the benchmark can measure ASR per modifier. The taxonomy below is the design
space; **bold** entries are the transforms that ship in v0.2.0 (the rest are the
roadmap for community PRs).

- **Encoding**: **`base64`**, **`rot13`**, **`hex`**, **`leetspeak`**.
- **Unicode**: **`homoglyph`**, **`zero_width`**, bidi, fullwidth.
- **Reordering**: **`reversed`** (character-reversed payload).
- **Splitting**: **`split`** (payload split across fragments), multi-part,
  acrostic.
- **Framing**: roleplay, hypothetical, translation, refusal-suppression,
  forced-prefix.
- **In-context**: many-shot, few-shot-priming.

The benchmark always measures the **`identity`** transform as the baseline.

## Axis 3 — Delivery shape (single-shot vs multi-turn; the `attacks/` strategies)

Single-shot is the v0.1.0 default; the multi-turn strategies are selected with
`--multiturn <strategy>` and live in `injectkit/attacks/multiturn.py`
(`MULTI_TURN_STRATEGIES`):

- **single_shot** — one user turn (the default; no `--multiturn`).
- **crescendo** — innocuous turns escalating toward the target ask.
- **persona_priming** — establish a persona over turns, then exploit it.
- **many_shot** — many real alternating turns priming compliance.
- **context_overflow** — bury the planted instruction under bulk context.

## Axis 4 — Defenses evaluated against (the `defenses/`)

Each is measured as ASR-with vs ASR-without (`benchmark.py::defense_delta`, where
a positive delta means the defense reduced ASR). A `Defense` exposes three hooks,
applied in order `wrap_system` -> `filter_input` -> `target.send` ->
`filter_output`, any of which may be a no-op. The built-ins (registered in
`injectkit/defenses/mitigations.py`, selected with `--defense`):
`none` (baseline / `NullDefense`), `hardened_system` (defensive system prefix),
`sandwich` (delimiter-fence untrusted input between trusted instructions),
`input_sanitizer` (strip/flag suspicious input markers),
`output_filter` (scrub leaked markers from the response).

## Mapping to external references

The families align with **OWASP LLM Top 10 — LLM01: Prompt Injection** and the
prompt-injection / jailbreak literature. Research datasets that exercise these
families (loaded only on opt-in, never bundled) are listed in
[`RESEARCH-USE.md`](./RESEARCH-USE.md).
