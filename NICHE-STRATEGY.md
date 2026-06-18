# MCP-Security Portfolio — Niche Strategy & Research-Grounded Roadmap

> Shared strategy document for the three-project portfolio. Identical copy lives in
> `mcp-range`, `mcp-warden`, and `injectkit`. Last updated 2026-06-18.

This is a planning artifact, not a spec. The binding build specs remain each
repo's own `ROADMAP.md`. This document explains **how the three fit together**,
**where the research literature says the roadmap should go next**, and **the
proposed path to a single product**.

---

## 1. The portfolio in one line

We are building the **offense → deterministic-enforcement → neutral-scorekeeper
triad** for one specific, currently-uncornered niche: **deterministic,
reproducible security for MCP / tool-using agents.**

| Project | Role | Form | License |
|---|---|---|---|
| **injectkit** | Offense / measurement (does the attack get through?) | Python CLI + library + GUI + GitHub Action | MIT |
| **mcp-warden** | Runtime enforcement (block it deterministically, no LLM in the block path) | Single Go binary, inline proxy | Apache-2.0 |
| **mcp-range** | Neutral benchmark (grade *any* defender, reproducibly, OWASP-1:1) | Lab range + scorer | MIT |

The niche-cementing value is in how they interlock, not in any one repo.

---

## 2. Built-out verdict (maturity scorecard)

The spine of all three exists; the **authority layer** (published numbers,
leaderboard, the missing OWASP category, the missing transports) does not.

| Project | Shipped | ~% to v1.0 | Done | Missing keystone |
|---|---|---|---|---|
| **injectkit** | v0.3 public, v0.4 staged | ~65% | Offense corpus, transforms, named attackers, white-box core (CPU-tested) | GPU-scale numbers (DEFERRED-NO-GPU), judge-in-loop attacks (v0.5), leaderboard |
| **mcp-warden** | v0.2 (stdio + taint) | ~35% | Deterministic core: scope, raw-bytes pin, taint exfil block, signed audit | **HTTP + SSE transports** (tri-transport moat is 1 of 3), server-id/shadowing, **published AgentDojo/MCPTox numbers** |
| **mcp-range** | v0.2 (9 of 10 OWASP cats) | ~40% | Two-axis scorer, OWASP coverage minus MCP07, Mode-B harness, crosswalk | **MCP07 auth lab**, **FPR corpus → Coverage Score**, **regenerated public leaderboard** |

---

## 3. Why the niche is uncornered (and the papers that prove it)

- **The threat is empirically large.** First large-scale study of 1,899 MCP
  servers: 7.2% general vulns, **5.5% MCP-specific tool poisoning**
  (arXiv:2506.13538). MCPSecBench: **protection success <30%**, **>85% of
  attacks compromise ≥1 host** (arXiv:2508.13220).
- **The problem worsens as models improve.** MCPTox: **more-capable models are
  *more* vulnerable** to tool poisoning (arXiv:2508.14925). This is the single
  strongest validation of the deterministic-enforcement thesis — the arms race
  cannot be won by better models.
- **Every competitor is a single leg.** Scanners only passively log; LLM-guardrail
  proxies defend an instruction channel with another instruction-follower
  (unsound); PipeLab's "State of MCP Security" matrix is hand-authored; AgentDojo
  / MCPSecBench measure *model* susceptibility, not *defender* coverage. Nobody
  has the offense → deterministic-enforce → neutral-reproducible-score triad keyed
  1:1 to the OWASP MCP Top 10.

---

## 4. What the research says about the roadmap

Holding the roadmap against what the papers actually found:

### Validated — stay the course
- **Deterministic, model-free enforcement** (warden no-LLM block path; range
  frozen Mode-A) is backed by MCPTox's capability-paradox finding.
- **Static-vs-runtime headline** (range) and **raw-bytes pin incl. non-standard
  fields** (warden DR-3) are exactly what the STRIDE/DREAD threat-model paper
  (arXiv:2603.22489) names as the core failure ("insufficient static validation
  and parameter visibility"). CyberArk full-schema poisoning underwrites DR-3.
- **Tool-poisoning / pin priority** (MCP03, shipped) is confirmed by the
  1,899-server study.

### Reprioritized — the ordering changes
1. **The utility/FPR axis is now table stakes, not polish.** AgentDojo
   (arXiv:2406.13352) set the field's bar by measuring *utility AND robustness
   together*: a defender that blocks by breaking legitimate flows is worthless.
   warden's `<2% false-block` and range's FPR-weighted Coverage Score are the same
   idea. **range's FPR/benign corpus → Coverage Score jumps to #1** — it is the
   metric every reviewer will demand on day one, and no headline score exists
   without it.
2. **Mode-B deserves more weight than "correlation-only" implies.** MCPSecBench's
   most-cited numbers are *per-host* (Claude blocks injection; Cursor/OpenAI
   don't). Keep the discipline (Mode-A frozen, Mode-B never folded into the
   score), but treat MCPSecBench as the **calibration target** and invest in
   Mode-B presentation — it's the part the market quotes.
3. **Run injectkit against a frontier model** to demonstrate the MCPTox
   capability-paradox curve. Cheap, high-signal, feeds the thesis directly.

### Newly exposed frontier — widest open ground
The literature has thoroughly mapped *attacks* (MCPSecBench 17, MCPTox 10,
MCP-38's 38 classes incl. **Function Return Injection**). The uncornered ground
is therefore **reproducible defense measurement** + the surfaces papers haven't
reached:
- **Async `Tasks` surface** (MCP 2026-07-28 RC). No paper, benchmark, or firewall
  covers async tool execution yet. First-mover wide open.
- **Semantic transformation / causality laundering.** Sound IFC (FIDES, CaMeL)
  needs a trusted agent runtime; warden reconstructs provenance from the wire and
  honestly concedes (in `LIMITS.md`) it cannot catch paraphrase-laundered exfil.
  That gap is the hard, defensible research frontier.

---

## 5. Revised top-of-roadmap sequencing (next quarter)

0. **Credibility gate first** — resolve warden's §10 citation gate and range's
   `STANDARDS.lock` (`nullable-until-harvested` fields). Cheap; unblocks all
   public claims.
1. **range: FPR/benign corpus → Coverage Score** — the literature's verdict; the
   critical path. ≥500 items, registry-sourced, license-clean, HarmBench-style
   held-out split, Wilson 95% CI.
2. **The integrated benchmark run** (injectkit → warden → range) → regenerated
   `LEADERBOARD.md` with static-vs-runtime + utility-retention numbers. The
   artifact the field is implicitly asking for; satisfies both range v1.0 and
   warden v1.0 (published AgentDojo/MCPTox numbers).
3. **injectkit frontier-model run** — the MCPTox capability-paradox curve.
4. **warden Streamable HTTP** (makes the EchoLeak flagship demo real on the
   flagship transport) → **range MCP07 auth lab** (closes OWASP to 10/10).
5. **Ecosystem capture** — PR mcp-range into the OWASP MCP Top 10 repo as the
   reference lab; contribute exfil tasks upstream to AgentDojo.

Note: MCP07 and HTTP slid *down* relative to a naive completeness-first plan. They
matter, but no paper makes them the blocker — whereas every paper, read together,
says the missing thing in this niche is a reproducible defender score with a
utility axis.

---

## 6. Path to a single product

**Decision: yes for two, no for the third — and not yet.**

- **injectkit + warden → one product.** "Test your MCP setup for injection, then
  enforce deterministically at runtime" is one coherent story: offense feeds
  defense, same buyer, same install.
- **mcp-range stays independent and neutral — this is non-negotiable.** range's
  only moat is neutrality: it grades warden *alongside* competitors and "sits
  above the arms race as the neutral scorekeeper." Folding range into the same
  product as warden turns it into "the vendor benchmark where the vendor's firewall
  wins" — instant credibility loss. The product *cites* range's public neutral
  results; it never *contains* range. Target: **OWASP-governed**.
- **Timing: after each project hits v1.0.** Merging mid-milestone (all at v0.2)
  taxes every remaining milestone. "At one point" = post-1.0.
- **How (cross-language reality):** warden is Go, injectkit/range are Python — a
  single binary is a fool's errand. Instead:
  - **Now (no-regret first step):** extract a shared **`mcp-sec-core`** package —
    the OWASP taxonomy, crosswalk, and wire-parsing are duplicated across all
    three today. This is the real DRY win and a move toward a merge you can take
    *before* committing to one.
  - **Post-1.0:** monorepo with separate packages + **one umbrella CLI**
    (`mcpsec scan` → injectkit, `mcpsec enforce` → warden), shared brand and
    distribution. range remains a separate repo/product.

The intellectual honesty already baked into all three (warden's deterministic-vs-
advisory DR-5 boundary, range's frozen-Mode-A-vs-indicative-Mode-B split,
injectkit's benign-canary discipline) is the durable moat in a hype-saturated
field. The remaining work is less invention than *finishing and publishing*.

---

## 7. Citation map

| Paper / source | arXiv / ref | Roadmap relevance |
|---|---|---|
| MCPTox — tool poisoning on real servers | 2508.14925 | Capability paradox → validates deterministic enforcement |
| MCPSecBench — 17 attacks / 4 surfaces | 2508.13220 | Per-host variance → Mode-B calibration target |
| AgentDojo — utility + robustness eval | 2406.13352 | Utility/FPR axis → promotes range Coverage Score to #1 |
| MCP threat modeling (STRIDE/DREAD) | 2603.22489 | "Insufficient static validation" → static-vs-runtime headline |
| 1,899 MCP servers study | 2506.13538 | 5.5% tool poisoning → market sizing, MCP03 priority |
| MCP-38 taxonomy | 2603.18063 | Function Return Injection → new lab/enforcement target |
| HarmBench | 2402.04249 | Held-out anti-gaming discipline for the FPR corpus |
| FIDES / CaMeL (sound IFC) | — | Policy shape adopted; sound-IFC gap = warden frontier |
| EchoLeak (M365 Copilot zero-click) | advisory | warden taint demo archetype (HTTP-class) |
| OWASP MCP Top 10 | OWASP repo | range keying; ecosystem-capture PR target |
