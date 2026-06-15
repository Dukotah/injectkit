# Changelog

All notable changes to **injectkit** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`injectkit gui` command** — point-and-shoot launch of the local web UI in
  your browser (`injectkit gui`, with `--host` / `--port` / `--no-open`). The
  existing `python -m injectkit.web` entry point still works.

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

[Unreleased]: https://github.com/Dukotah/injectkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Dukotah/injectkit/releases/tag/v0.1.0
