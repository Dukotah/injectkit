# injectkit — Research Use & Responsible Disclosure

> **injectkit is a defensive tool for authorized research only.** Use it to scan
> LLM endpoints, models, and agents that **you own or are explicitly authorized
> to test**. Do not target third-party production systems.

## 1. Authorized-research-only

By using injectkit you confirm that:

- You have **explicit authorization** to test every target you point it at
  (your own local/self-hosted model, your own endpoint, or a system you have
  written permission to assess).
- You will **not** use it against third-party systems you do not control.
- You understand the toolkit measures **instruction-bypass robustness** via a
  **benign canary proxy** — it is built to find weaknesses so you can fix them,
  not to generate harmful content.

These notices also appear in the CLI help, the README, the SECURITY policy, and
every rendered report.

## 2. Benign-canary methodology (no harmful content)

injectkit's bundled corpus and all generated attacks (transforms, multi-turn,
adaptive) default to a **benign success marker**: the payload asks the target to
emit a unique per-run token it was told to withhold. A marker echo proves the
injection bypassed the instruction — with **zero** harmful output. The adaptive
attacker optimises attack *structure* against this benign proxy; it is **not** a
harmful-output generator.

## 3. Research datasets — OPT-IN and GATED (never bundled)

injectkit ships **no** harmful prompts, exploits, or harm-behavior datasets. For
authorized academic research, the `injectkit.research` package offers an
*interface* to load the official public datasets from the literature, with these
rules:

- **No bundling.** Only dataset **names and canonical URLs** ship (see the table
  below and `injectkit/research/registry.py`). The data is downloaded from its
  **official source** on demand; injectkit never redistributes it.
- **Explicit acknowledgment required.** Loading is gated by
  `injectkit.research.base.require_acknowledgment`, satisfied only by **one** of:
  - the `--research-benchmark` CLI flag,
  - constructing the loader with `acknowledge=True`, or
  - setting the environment variable `INJECTKIT_RESEARCH_ACK=1`.

  Without it, the loader raises `ResearchAcknowledgmentError` carrying the
  disclaimer below.
- **Honour each source's licence/terms.** You are responsible for complying with
  the licence and acceptable-use terms of every dataset you download.

### Disclaimer (shown on every ungated access attempt)

> Research datasets are loaded ONLY for authorized defensive research on targets
> you own or are explicitly permitted to test. These datasets are maintained by
> third parties, are downloaded from their official sources (injectkit does not
> redistribute them), and may contain sensitive or offensive material. By
> acknowledging, you confirm authorized, ethical, research-only use and
> acceptance of each dataset's own licence/terms.

### Referenced official datasets (pointers only — see `registry.py`)

| Key                     | Dataset                          | Official source |
|-------------------------|----------------------------------|-----------------|
| `advbench`              | AdvBench (GCG)                    | https://github.com/llm-attacks/llm-attacks |
| `harmbench`             | HarmBench                        | https://www.harmbench.org/ |
| `jailbreakbench`        | JailbreakBench (JBB-Behaviors)   | https://jailbreakbench.github.io/ |
| `in_the_wild_jailbreaks`| In-The-Wild Jailbreak Prompts    | https://github.com/verazuo/jailbreak_llms |
| `tensor_trust`          | Tensor Trust (prompt injection)  | https://tensortrust.ai/paper/ |

## 4. Responsible disclosure

When injectkit helps you find a real weakness in a system you are authorized to
assess:

1. **Report privately first.** Notify the system owner / vendor through their
   security contact or a coordinated-disclosure channel before any public
   mention. Do not publish reproduction details that enable abuse.
2. **Give time to remediate.** Allow a reasonable window (commonly 90 days)
   before public discussion, and coordinate timelines with the owner.
3. **Minimise impact.** Use the benign canary proxy; never exfiltrate real data,
   degrade service, or pivot beyond the authorized scope.
4. **Share fixes, not weapons.** Prefer publishing the mitigation (a defense in
   `injectkit/defenses/`) over a turnkey exploit.

To report a vulnerability **in injectkit itself**, see
[`SECURITY.md`](https://github.com/Dukotah/injectkit/blob/main/SECURITY.md).

## 5. Out of scope / prohibited

- Targeting systems you do not own or lack written authorization to test.
- Using injectkit to produce or distribute harmful content.
- Framing transforms/attackers as detection-evasion against third-party
  production defenses.
