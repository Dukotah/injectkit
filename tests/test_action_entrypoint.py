"""Unit tests for the GitHub Action entrypoint (``entrypoint.sh``).

These tests exercise the shell entrypoint in isolation and fully offline by
stubbing the ``injectkit`` CLI with a tiny fake script (passed via the
``INJECTKIT_CLI`` override env var). No network, no real model, no API key.

We verify:
  * the scan argument vector is assembled from INJECTKIT_* env (and that
    empty/false inputs are skipped);
  * the entrypoint publishes the expected outputs to ``$GITHUB_OUTPUT``;
  * the scan's exit code is surfaced as an output but does NOT make the
    entrypoint itself exit non-zero (action.yml re-applies the gate);
  * summary parsing of a SARIF and a JSON report yields total/failed/highest.

The tests are skipped if no ``bash`` interpreter is available (e.g. a bare
Windows runner without Git Bash); on Windows with Git Bash / Cygwin they run.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"

BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None or not ENTRYPOINT.is_file(),
    reason="bash interpreter or entrypoint.sh not available",
)


def _win_to_posix(path: Path) -> str:
    """Best-effort convert a Windows path to a form bash can use.

    Cygwin/Git Bash accept ``C:/Users/...`` style forward-slash paths for most
    operations; we just normalize separators. If ``cygpath`` is present we use
    it for a fully-correct POSIX path.
    """
    cygpath = shutil.which("cygpath")
    if cygpath:
        out = subprocess.run(
            [cygpath, "-u", str(path)], capture_output=True, text=True
        )
        if out.returncode == 0:
            return out.stdout.strip()
    return str(path).replace("\\", "/")


def _make_stub_cli(tmp_path: Path, *, write: str = "", exit_code: int = 0) -> Path:
    """Create a fake injectkit CLI that records argv and optionally writes a file.

    The stub:
      * appends its argv (one per line) to ``<dir>/argv.txt``;
      * writes ``write`` (verbatim) to the path following ``--out`` if present;
      * exits with ``exit_code``.
    """
    stub = tmp_path / "fake_injectkit.sh"
    argv_file = tmp_path / "argv.txt"
    payload_file = tmp_path / "payload.txt"
    payload_file.write_text(write, encoding="utf-8")

    script = f"""#!/usr/bin/env bash
: >"{_win_to_posix(argv_file)}"
out=""
prev=""
for a in "$@"; do
  printf '%s\\n' "$a" >>"{_win_to_posix(argv_file)}"
  if [ "$prev" = "--out" ]; then out="$a"; fi
  prev="$a"
done
if [ -n "$out" ]; then
  cp "{_win_to_posix(payload_file)}" "$out"
fi
exit {exit_code}
"""
    stub.write_text(script, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_entrypoint(
    tmp_path: Path,
    env_extra: dict[str, str],
    stub_cli: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    """Run entrypoint.sh with a stubbed CLI; return (proc, parsed outputs)."""
    gh_output = tmp_path / "gh_output.txt"
    gh_summary = tmp_path / "gh_summary.txt"
    gh_output.write_text("", encoding="utf-8")
    gh_summary.write_text("", encoding="utf-8")

    env = dict(os.environ)
    # Point the entrypoint at our stub CLI and GitHub output files.
    env["INJECTKIT_CLI"] = f"bash {_win_to_posix(stub_cli)}"
    env["GITHUB_OUTPUT"] = _win_to_posix(gh_output)
    env["GITHUB_STEP_SUMMARY"] = _win_to_posix(gh_summary)
    # Make sure no leftover INJECTKIT_* from the host leaks in.
    for k in list(env):
        if k.startswith("INJECTKIT_") and k not in ("INJECTKIT_CLI",):
            env.pop(k)
    env.update(env_extra)

    proc = subprocess.run(
        [BASH, _win_to_posix(ENTRYPOINT)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )

    outputs: dict[str, str] = {}
    for line in gh_output.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            outputs[key] = val
    return proc, outputs


# --------------------------------------------------------------------------- #
# Argument assembly
# --------------------------------------------------------------------------- #

def test_builds_full_arg_vector(tmp_path: Path) -> None:
    """All provided inputs map to the expected CLI flags."""
    stub = _make_stub_cli(tmp_path, write='{"total": 0, "findings": []}', exit_code=0)
    env = {
        "INJECTKIT_TARGET": "http",
        "INJECTKIT_URL": "http://127.0.0.1:9/chat",
        "INJECTKIT_MODEL": "claude-opus-4-8",
        "INJECTKIT_FAIL_ON": "medium",
        "INJECTKIT_TECHNIQUES": "direct_injection,jailbreak",
        "INJECTKIT_FORMAT": "json",
        "INJECTKIT_OUT": "report.json",
        "INJECTKIT_JUDGE": "true",
        "INJECTKIT_JUDGE_MODEL": "claude-haiku-4-5",
    }
    proc, _ = _run_entrypoint(tmp_path, env, stub)
    assert proc.returncode == 0, proc.stderr

    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[0] == "scan"
    # Spot-check each flag/value pair is present and adjacent.
    def pair(flag: str) -> str:
        i = argv.index(flag)
        return argv[i + 1]

    assert pair("--target") == "http"
    assert pair("--url") == "http://127.0.0.1:9/chat"
    assert pair("--model") == "claude-opus-4-8"
    assert pair("--fail-on") == "medium"
    # The comma-separated "techniques" input is split into one repeated
    # --technique flag per name (the CLI flag is --technique, repeatable, and
    # does not split commas itself).
    tech_values = [argv[i + 1] for i, a in enumerate(argv) if a == "--technique"]
    assert tech_values == ["direct_injection", "jailbreak"]
    assert "--techniques" not in argv
    assert pair("--format") == "json"
    assert pair("--out") == "report.json"
    assert "--judge" in argv
    assert pair("--judge-model") == "claude-haiku-4-5"


def test_skips_empty_and_false_inputs(tmp_path: Path) -> None:
    """Empty inputs are omitted; judge=false drops --judge and --judge-model."""
    stub = _make_stub_cli(tmp_path, write="{}", exit_code=0)
    env = {
        "INJECTKIT_TARGET": "mock",
        "INJECTKIT_JUDGE": "false",
        "INJECTKIT_JUDGE_MODEL": "claude-haiku-4-5",  # must be ignored
        # No URL / model / techniques / config provided.
    }
    proc, _ = _run_entrypoint(tmp_path, env, stub)
    assert proc.returncode == 0, proc.stderr

    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--judge" not in argv
    assert "--judge-model" not in argv
    assert "--url" not in argv
    assert "--model" not in argv
    assert "--techniques" not in argv
    assert "--technique" not in argv
    assert "--config" not in argv
    # Defaults still applied: format/out/fail-on come from the entrypoint.
    assert "--target" in argv and argv[argv.index("--target") + 1] == "mock"
    assert "--fail-on" in argv  # defaulted to "high"
    assert "--format" in argv  # defaulted to "sarif"
    assert "--out" in argv  # defaulted to "injectkit-results.sarif"


def test_techniques_split_and_trimmed(tmp_path: Path) -> None:
    """A comma-separated techniques input with stray spaces/empties splits cleanly."""
    stub = _make_stub_cli(tmp_path, write="{}", exit_code=0)
    env = {
        "INJECTKIT_TARGET": "mock",
        # Stray whitespace and a doubled comma must be tolerated.
        "INJECTKIT_TECHNIQUES": " direct_injection , jailbreak ,, system_prompt_leak",
    }
    proc, _ = _run_entrypoint(tmp_path, env, stub)
    assert proc.returncode == 0, proc.stderr
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    tech_values = [argv[i + 1] for i, a in enumerate(argv) if a == "--technique"]
    assert tech_values == ["direct_injection", "jailbreak", "system_prompt_leak"]
    assert "--techniques" not in argv


def test_config_input_passed_through(tmp_path: Path) -> None:
    """A config path input becomes --config."""
    stub = _make_stub_cli(tmp_path, write="{}", exit_code=0)
    env = {"INJECTKIT_CONFIG": ".injectkit.yaml", "INJECTKIT_TARGET": "mock"}
    proc, _ = _run_entrypoint(tmp_path, env, stub)
    assert proc.returncode == 0, proc.stderr
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    i = argv.index("--config")
    assert argv[i + 1] == ".injectkit.yaml"


# --------------------------------------------------------------------------- #
# Exit-code handling and outputs
# --------------------------------------------------------------------------- #

def test_passing_scan_publishes_outputs(tmp_path: Path) -> None:
    """A clean scan (exit 0) publishes outputs and the script itself exits 0."""
    report = {"total": 6, "failed": 0, "findings": []}
    stub = _make_stub_cli(tmp_path, write=json.dumps(report), exit_code=0)
    env = {"INJECTKIT_TARGET": "mock", "INJECTKIT_FORMAT": "json", "INJECTKIT_OUT": "r.json"}
    proc, outputs = _run_entrypoint(tmp_path, env, stub)

    assert proc.returncode == 0, proc.stderr
    assert outputs["exit-code"] == "0"
    assert outputs["total"] == "6"
    assert outputs["failed"] == "0"
    assert outputs["highest-severity"] == "none"
    # JSON format -> no sarif path.
    assert outputs["sarif-path"] == ""
    assert outputs["report-path"].endswith("r.json")


def test_failing_scan_surfaces_code_but_script_exits_zero(tmp_path: Path) -> None:
    """A scan that finds issues (exit 2) is recorded but does not abort the script.

    The action.yml gate re-applies the exit code after SARIF upload, so the
    entrypoint must always exit 0 to guarantee outputs are published.
    """
    report = {
        "total": 6,
        "failed": 2,
        "findings": [
            {"severity": "high"},
            {"severity": "medium"},
        ],
    }
    stub = _make_stub_cli(tmp_path, write=json.dumps(report), exit_code=2)
    env = {"INJECTKIT_TARGET": "mock", "INJECTKIT_FORMAT": "json", "INJECTKIT_OUT": "r.json"}
    proc, outputs = _run_entrypoint(tmp_path, env, stub)

    # Entrypoint itself succeeds...
    assert proc.returncode == 0, proc.stderr
    # ...but records the scan's failing code and the parsed summary.
    assert outputs["exit-code"] == "2"
    assert outputs["failed"] == "2"
    assert outputs["highest-severity"] == "high"
    assert outputs["total"] == "6"


def test_sarif_report_parsed_and_sarif_path_set(tmp_path: Path) -> None:
    """A SARIF report yields a sarif-path output and parsed counts."""
    sarif = {
        "version": "2.1.0",
        "runs": [
            {
                "properties": {"total_attacks": 6},
                "results": [
                    {"properties": {"severity": "critical"}},
                    {"properties": {"severity": "high"}},
                    {"properties": {"severity": "low"}},
                ],
            }
        ],
    }
    stub = _make_stub_cli(tmp_path, write=json.dumps(sarif), exit_code=2)
    env = {
        "INJECTKIT_TARGET": "mock",
        "INJECTKIT_FORMAT": "sarif",
        "INJECTKIT_OUT": "out.sarif",
    }
    proc, outputs = _run_entrypoint(tmp_path, env, stub)

    assert proc.returncode == 0, proc.stderr
    assert outputs["total"] == "6"
    assert outputs["failed"] == "3"
    assert outputs["highest-severity"] == "critical"
    assert outputs["sarif-path"].endswith("out.sarif")
    assert outputs["report-path"].endswith("out.sarif")


def test_missing_report_file_degrades_gracefully(tmp_path: Path) -> None:
    """If the scan writes no file, summary outputs default to 0/0/none."""
    # Stub exits 0 but writes nothing (no --out cp because payload empty? Force
    # by removing the out file the stub created): use a stub that doesn't write.
    stub = tmp_path / "noop_cli.sh"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {
        "INJECTKIT_TARGET": "mock",
        "INJECTKIT_FORMAT": "sarif",
        "INJECTKIT_OUT": "absent.sarif",
    }
    proc, outputs = _run_entrypoint(tmp_path, env, stub)

    assert proc.returncode == 0, proc.stderr
    assert outputs["total"] == "0"
    assert outputs["failed"] == "0"
    assert outputs["highest-severity"] == "none"
    # No file -> sarif-path empty even though format is sarif.
    assert outputs["sarif-path"] == ""


def test_summary_written(tmp_path: Path) -> None:
    """The job summary file gets the human-readable block."""
    stub = _make_stub_cli(tmp_path, write='{"total": 3, "findings": []}', exit_code=0)
    env = {"INJECTKIT_TARGET": "mock", "INJECTKIT_FORMAT": "json", "INJECTKIT_OUT": "r.json"}
    gh_summary = tmp_path / "gh_summary.txt"
    # _run_entrypoint creates the summary file; capture its content after.
    proc, _ = _run_entrypoint(tmp_path, env, stub)
    assert proc.returncode == 0, proc.stderr
    text = gh_summary.read_text(encoding="utf-8")
    assert "injectkit prompt-injection scan" in text
    assert "Attacks run" in text
    assert "authorized" in text.lower()
