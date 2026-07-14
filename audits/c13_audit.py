"""C13 — configuration-identity audit (stub).

For every reported contrast pair, checks that the two cells' *fully resolved* configs
(``resolved_config.json``, written by ``utils.conventions.serialize_resolved_config``)
differ **only** in the varied factor and its pre-registered factor-specific parameters.
Output is committed to ``audits/c13/`` alongside the figures.

Session 0: the diff engine and pass/fail rule are wired; the contrast registry (which
pairs, which varied factor, which factor-specific params are licensed) is filled when the
protocol freezes (Session 1) and consumed from Sessions 4/5/6b/7. Runs here as a no-op
that reports "no contrasts registered yet" so CI exercises the import path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Filled at protocol freeze. Each entry: contrast -> (varied factor keys, licensed
# factor-specific keys that may also differ). Everything else must be identical.
CONTRAST_REGISTRY: dict[str, dict[str, list[str]]] = {
    # "C-COHERENCE": {"varies": ["use_rule"], "licensed": []},
    # "C-PRIOR":     {"varies": ["prior"],    "licensed": ["factor_specific.prior_scale"]},
    # ...
}


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def audit_pair(cfg_a: dict, cfg_b: dict, varies: list[str], licensed: list[str]) -> dict:
    """Return {'pass': bool, 'illicit_diffs': {...}} for one contrast pair."""
    fa, fb = _flatten(cfg_a), _flatten(cfg_b)
    allowed = set(varies) | set(licensed)
    keys = (set(fa) | set(fb)) - {"_config_sha256"}
    illicit = {
        k: {"a": fa.get(k, "<absent>"), "b": fb.get(k, "<absent>")}
        for k in keys
        if fa.get(k) != fb.get(k) and k not in allowed
    }
    return {"pass": not illicit, "illicit_diffs": illicit}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=Path, default=Path("logs"),
                    help="dir tree containing resolved_config.json files")
    ap.add_argument("--out", type=Path, default=Path("audits/c13"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if not CONTRAST_REGISTRY:
        report = {"status": "no_contrasts_registered",
                  "note": "CONTRAST_REGISTRY is filled at protocol freeze (Session 1)."}
        (args.out / "c13_report.json").write_text(json.dumps(report, indent=2))
        print("C13: no contrasts registered yet (expected in Session 0).")
        return 0

    # Real audit path (exercised from Session 4 onward) omitted until the registry exists.
    raise NotImplementedError("C13 audit runs once CONTRAST_REGISTRY is frozen.")


if __name__ == "__main__":
    sys.exit(main())
