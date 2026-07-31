#!/usr/bin/env python
"""Run the CI workflow's inline assertion steps locally.

Motivation (2026-07-31): the `hparam_search` stream was added with a full local suite green
(336 tests, ruff clean, `make smoke`, `make schema`) and CI still failed. The C1 gate had a
*second*, independent copy of the stream invariant written inline in `smoke.yml` as
`assert len(STREAM_NAMES) == 8`. Nothing local executed it, so no amount of local testing
could have caught the break.

That is a general hazard, not a one-off: any invariant asserted only in a workflow file is
invisible to the local suite and to `pytest`, and it drifts silently from the code it guards.
The durable fixes are (a) prefer real tests in `tests/` over inline workflow assertions, and
(b) for the ones that must stay inline, run them locally — which is what this does.

This parses the workflow, extracts every `run:` step whose script contains a Python
assertion, and executes each with the same shell semantics CI uses (`bash -e`). It is wired
into `make ci-parity`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ASSERT_RE = re.compile(r"\bassert\b")


def extract_assertion_steps(workflow: Path) -> list[tuple[str, str]]:
    """Return ``(step_name, script)`` for every run-step containing a Python assertion."""
    spec = yaml.safe_load(workflow.read_text())
    out: list[tuple[str, str]] = []
    for job in spec.get("jobs", {}).values():
        for step in job.get("steps", []):
            script = step.get("run")
            if script and ASSERT_RE.search(script):
                out.append((step.get("name", "<unnamed>"), script))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflows", default=".github/workflows", type=Path)
    args = ap.parse_args()

    files = sorted(args.workflows.glob("*.yml")) + sorted(args.workflows.glob("*.yaml"))
    if not files:
        print(f"no workflows found under {args.workflows}", file=sys.stderr)
        return 1

    failures = 0
    checked = 0
    for wf in files:
        for name, script in extract_assertion_steps(wf):
            checked += 1
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
                fh.write(script)
                path = fh.name
            proc = subprocess.run(
                ["bash", "-e", path], capture_output=True, text=True, cwd=Path.cwd()
            )
            status = "ok" if proc.returncode == 0 else "FAIL"
            print(f"[{status}] {wf.name}: {name}")
            if proc.returncode != 0:
                failures += 1
                tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
                for line in tail:
                    print(f"       {line}")

    print(f"\n{checked} assertion step(s) checked, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
