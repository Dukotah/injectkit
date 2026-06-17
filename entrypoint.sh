#!/usr/bin/env bash
# injectkit GitHub Action entrypoint.
#
# DEFENSIVE / AUTHORIZED USE ONLY. This runs the injectkit prompt-injection
# scanner against an endpoint you own or are explicitly authorized to test.
#
# Reads INJECTKIT_* environment variables (populated from the action.yml
# inputs), assembles an `injectkit scan` command, runs it, then publishes the
# scan's exit code and a few summary outputs to $GITHUB_OUTPUT and the job
# summary. The severity gate (failing the build) is re-applied by action.yml so
# the SARIF upload can run first; this script never `exit`s non-zero just
# because the scan found something — it records the code as an output.
#
# The script is deliberately self-contained and side-effect-light so it can be
# unit-tested offline by stubbing `injectkit` on PATH and pointing GITHUB_OUTPUT
# at a temp file.
set -uo pipefail

# Resolve a Python interpreter once. Honor an explicit ${PYTHON} override, then
# prefer `python`, then fall back to `python3` (modern Linux/CI images often
# ship only `python3`). Used both to invoke the CLI (`python -m injectkit`) and
# the stdlib report parser below.
if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"  # last resort; parse_report degrades gracefully if absent
  fi
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Emit "name=value" to the GitHub outputs file when one is configured. Multi-line
# values are not expected here, so the simple form is safe.
set_output() {
  local name="$1"
  local value="$2"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf '%s=%s\n' "$name" "$value" >>"$GITHUB_OUTPUT"
  fi
}

# Append a line to the job summary when running on GitHub.
summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n' "$1" >>"$GITHUB_STEP_SUMMARY"
  fi
}

# True when a value is a "truthy" string (case-insensitive true/1/yes/on).
is_true() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    true | 1 | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

# Resolve the injectkit CLI command. Prefer an explicit override (used in tests
# and for pinned installs), then a `injectkit` on PATH, then `python -m`.
injectkit_cmd() {
  if [ -n "${INJECTKIT_CLI:-}" ]; then
    printf '%s' "$INJECTKIT_CLI"
  elif command -v injectkit >/dev/null 2>&1; then
    printf 'injectkit'
  else
    printf '%s -m injectkit' "${PYTHON:-python}"
  fi
}

# ---------------------------------------------------------------------------
# Build the scan argument vector from INJECTKIT_* inputs.
# Echoes one argument per line so callers (and tests) can capture it exactly.
# Empty/false inputs are skipped so they don't clobber config-file values.
# ---------------------------------------------------------------------------
build_scan_args() {
  local -a args=(scan)

  [ -n "${INJECTKIT_CONFIG:-}" ] && args+=(--config "$INJECTKIT_CONFIG")
  [ -n "${INJECTKIT_TARGET:-}" ] && args+=(--target "$INJECTKIT_TARGET")
  [ -n "${INJECTKIT_URL:-}" ] && args+=(--url "$INJECTKIT_URL")
  [ -n "${INJECTKIT_MODEL:-}" ] && args+=(--model "$INJECTKIT_MODEL")
  [ -n "${INJECTKIT_FAIL_ON:-}" ] && args+=(--fail-on "$INJECTKIT_FAIL_ON")
  # The Action exposes a single comma-separated "techniques" input, but the CLI
  # flag is --technique (singular, repeatable, one name per occurrence and it
  # does NOT split commas itself). Split the input on commas and emit one
  # --technique per non-empty, trimmed name so filtering actually works.
  if [ -n "${INJECTKIT_TECHNIQUES:-}" ]; then
    local _old_ifs="$IFS"
    IFS=','
    # shellcheck disable=SC2206  # deliberate word-split on commas
    local -a _techs=(${INJECTKIT_TECHNIQUES})
    IFS="$_old_ifs"
    local _t
    for _t in "${_techs[@]}"; do
      # Trim leading/trailing whitespace; skip empties (e.g. "a,,b" or "a, b").
      _t="${_t#"${_t%%[![:space:]]*}"}"
      _t="${_t%"${_t##*[![:space:]]}"}"
      [ -n "$_t" ] && args+=(--technique "$_t")
    done
  fi
  [ -n "${INJECTKIT_FORMAT:-}" ] && args+=(--format "$INJECTKIT_FORMAT")
  [ -n "${INJECTKIT_OUT:-}" ] && args+=(--out "$INJECTKIT_OUT")

  if is_true "${INJECTKIT_JUDGE:-}"; then
    args+=(--judge)
    [ -n "${INJECTKIT_JUDGE_MODEL:-}" ] && args+=(--judge-model "$INJECTKIT_JUDGE_MODEL")
  fi

  printf '%s\n' "${args[@]}"
}

# ---------------------------------------------------------------------------
# Parse a JSON/SARIF report for summary outputs. Pure-Python, stdlib only, so it
# works wherever injectkit is installed. Echoes "total failed highest" on one
# line; prints "0 0 none" if the file is absent or unparseable.
# ---------------------------------------------------------------------------
parse_report() {
  local path="$1"
  local fmt="$2"
  if [ -z "$path" ] || [ ! -f "$path" ]; then
    printf '0 0 none\n'
    return 0
  fi
  "${PYTHON:-python}" - "$path" "$fmt" <<'PY' 2>/dev/null || printf '0 0 none\n'
import json
import sys

path, fmt = sys.argv[1], sys.argv[2]
order = ["info", "low", "medium", "high", "critical"]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    print("0 0 none")
    raise SystemExit(0)

total = 0
failed = 0
highest = -1

def bump(sev):
    global highest
    try:
        highest = max(highest, order.index(str(sev).lower()))
    except ValueError:
        pass

if fmt == "sarif" or (isinstance(data, dict) and "runs" in data):
    for run in data.get("runs", []):
        props = run.get("properties", {})
        total = props.get("total_attacks", total) or total
        results = run.get("results", [])
        failed = max(failed, len(results))
        for res in results:
            sev = res.get("properties", {}).get("severity")
            if sev:
                bump(sev)
else:
    # injectkit JSON report shape: {"total":..,"findings":[{"severity":..}], ...}
    total = data.get("total", data.get("total_attacks", 0)) or 0
    findings = data.get("findings", []) or []
    failed = data.get("failed", len(findings)) or len(findings)
    for f in findings:
        bump(f.get("severity", "info"))

highest_name = order[highest] if highest >= 0 else "none"
print(f"{total} {failed} {highest_name}")
PY
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  local notice="injectkit is a defensive tool. Scan only endpoints you own or are explicitly authorized to test."
  echo "::notice::${notice}"

  # Mock target convenience: the CLI takes --target mock with the built-in
  # deterministic MockTarget; nothing else to set up.
  local fail_on="${INJECTKIT_FAIL_ON:-high}"
  local fmt="${INJECTKIT_FORMAT:-sarif}"
  local out="${INJECTKIT_OUT:-injectkit-results.sarif}"
  INJECTKIT_FAIL_ON="$fail_on"
  INJECTKIT_FORMAT="$fmt"
  INJECTKIT_OUT="$out"

  # Assemble the command.
  local cmd
  cmd="$(injectkit_cmd)"
  local -a scan_args
  mapfile -t scan_args < <(build_scan_args)

  echo "+ ${cmd} ${scan_args[*]}"

  # Run the scan. errexit is already off (we only use -uo pipefail), so a
  # non-zero scan exit won't abort the script; capture the code explicitly.
  # shellcheck disable=SC2086  # intentional split so "python -m injectkit" works
  $cmd "${scan_args[@]}"
  local code=$?

  # Resolve the report path (relative paths resolve against the CWD the scan
  # ran in, which is the action's working-directory).
  local report_path="$out"
  if [ -f "$report_path" ]; then
    report_path="$(cd "$(dirname "$report_path")" && pwd)/$(basename "$report_path")"
  fi

  local sarif_path=""
  if [ "$fmt" = "sarif" ] && [ -f "$out" ]; then
    sarif_path="$report_path"
  fi

  # Summary outputs (best-effort; never fail the script on a parse miss).
  local stats total failed highest
  stats="$(parse_report "$out" "$fmt")"
  total="$(printf '%s' "$stats" | awk '{print $1}')"
  failed="$(printf '%s' "$stats" | awk '{print $2}')"
  highest="$(printf '%s' "$stats" | awk '{print $3}')"

  set_output "exit-code" "$code"
  set_output "report-path" "$report_path"
  set_output "sarif-path" "$sarif_path"
  set_output "total" "${total:-0}"
  set_output "failed" "${failed:-0}"
  set_output "highest-severity" "${highest:-none}"

  summary "## injectkit prompt-injection scan"
  summary ""
  summary "- Target: \`${INJECTKIT_TARGET:-(from config)}\`"
  summary "- Attacks run: **${total:-0}**"
  summary "- Findings (attacks that got through): **${failed:-0}**"
  summary "- Highest severity: **${highest:-none}**"
  summary "- Fail-on threshold: \`${fail_on}\`"
  summary "- Exit code: \`${code}\`"
  summary ""
  summary "> ${notice}"

  # Do NOT propagate the scan exit code here; action.yml re-applies the gate
  # after uploading SARIF. Exit 0 so the outputs above are always published.
  return 0
}

main "$@"
