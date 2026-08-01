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

# Filled 2026-07-31, once every one of the 10 cells had a committed config (before that,
# half these pairs had no B-arm to diff against). Each entry: contrast -> the varied factor
# keys, the licensed factor-specific keys that may also differ, and the cell pairs.
# Everything else must be identical between the two arms.
#
# What is licensed, and why (freeze item 12's class-3 rule — only the varied factor's own
# parameters may differ):
#   * ``arm`` and ``run_id`` are identity labels derived from the cell, not parameters; they
#     differ in EVERY pair by construction and are excluded globally (see IDENTITY_KEYS).
#   * ``method`` is licensed wherever ``prior`` varies: the repo builds prior=off through the
#     ``bdqn`` factory and prior=on through the ``rp_bdqn`` alias, which per src/config.py is
#     "the SAME agent plus a fixed additive prior". The method string is therefore a spelling
#     of the prior factor, not an independent difference.
#   * ``factor_specific.prior_scale`` is licensed wherever ``prior`` varies (it exists only
#     when prior=on) and NOWHERE else — its value is shared across all prior=on cells.
#   * ``factor_specific.eps_schedule`` is licensed wherever ``use_rule`` varies, because
#     ``ensemble_mean`` is the only rule that consumes an ε schedule (freeze item 2).
#
# Cell ids are the canonical ``<use_rule>|<prior>|K<K>`` form (src.config._canonical_cell_id).
CONTRAST_REGISTRY: dict[str, dict[str, Any]] = {
    "C-USE": {
        "varies": ["use_rule"],
        "licensed": ["factor_specific.eps_schedule"],
        "pairs": [("episodic|off|K10", "ensemble_mean|off|K10"),
                  ("episodic|on|K10", "ensemble_mean|on|K10")],
    },
    "C-COHERENCE": {
        "varies": ["use_rule"],
        "licensed": ["factor_specific.eps_schedule"],
        "pairs": [("episodic|off|K10", "per_step|off|K10"),
                  ("episodic|on|K10", "per_step|on|K10")],
    },
    "C-PRIOR": {
        "varies": ["prior"],
        "licensed": ["method", "factor_specific.prior_scale"],
        "pairs": [("episodic|off|K10", "episodic|on|K10"),
                  ("per_step|off|K10", "per_step|on|K10"),
                  ("ensemble_mean|off|K10", "ensemble_mean|on|K10"),
                  ("episodic|off|K5", "episodic|on|K5"),
                  ("episodic|off|K20", "episodic|on|K20")],
    },
    "C-K": {
        "varies": ["K"],
        "licensed": [],
        "pairs": [("episodic|off|K5", "episodic|off|K10"),
                  ("episodic|off|K10", "episodic|off|K20"),
                  ("episodic|on|K5", "episodic|on|K10"),
                  ("episodic|on|K10", "episodic|on|K20")],
    },
}

# Derived identity labels, not parameters: they differ in every pair by construction.
IDENTITY_KEYS = frozenset({"arm", "run_id", "cell_id", "_config_sha256"})


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _is_allowed(key: str, allowed: set[str]) -> bool:
    """True if ``key`` is licensed, either exactly or as a child of a licensed subtree.

    Licensing must be subtree-wide: ``factor_specific.eps_schedule`` is a *dict* of
    (eps_start, eps_end, eps_decay_steps), and ``_flatten`` turns it into three leaf keys.
    Licensing only the exact parent name would therefore license nothing at all, and the
    C-USE pair — whose ε schedule is precisely the licensed factor-specific parameter —
    would fail on all three leaves.
    """
    return any(key == a or key.startswith(a + ".") for a in allowed)


def audit_pair(cfg_a: dict, cfg_b: dict, varies: list[str], licensed: list[str]) -> dict:
    """Return {'pass': bool, 'illicit_diffs': {...}} for one contrast pair."""
    fa, fb = _flatten(cfg_a), _flatten(cfg_b)
    allowed = set(varies) | set(licensed)
    keys = (set(fa) | set(fb)) - IDENTITY_KEYS
    illicit = {
        k: {"a": fa.get(k, "<absent>"), "b": fb.get(k, "<absent>")}
        for k in keys
        if fa.get(k) != fb.get(k) and not _is_allowed(k, allowed)
    }
    return {"pass": not illicit, "illicit_diffs": illicit}


def collect_cells(source: Path, mode: str) -> dict[str, dict]:
    """Map canonical cell_id -> resolved config dict.

    ``mode='configs'`` reads committed ``configs/*.yaml`` through the real loader, so the
    audit is runnable **before** any run exists — a contrast whose two arms differ illicitly
    is a protocol defect at commit time, not at analysis time. ``mode='runs'`` reads
    ``resolved_config.json`` files emitted by executed runs (the Session 4+ path).
    """
    cells: dict[str, dict] = {}
    if mode == "configs":
        from src import config as config_mod

        for path in sorted(source.glob("*.yaml")):
            cfg = config_mod.load_config(path)
            # ``.data`` is the resolved dict (YAML as written + derived cell_id/size_class),
            # i.e. exactly what serialize_resolved_config writes for an executed run.
            data = cfg.data
            cell = data.get("cell_id") or data.get("arm")
            if cell:
                cells[cell] = data
    else:
        for path in sorted(source.rglob("resolved_config.json")):
            data = json.loads(path.read_text())
            cell = data.get("arm") or data.get("cell_id")
            if cell:
                cells[cell] = data
    return cells


def run_audit(cells: dict[str, dict]) -> dict:
    """Audit every registered contrast pair present in ``cells``."""
    results: dict[str, Any] = {}
    n_pass = n_fail = n_skip = 0
    for contrast, spec in CONTRAST_REGISTRY.items():
        entries = []
        for a, b in spec["pairs"]:
            if a not in cells or b not in cells:
                entries.append({"pair": [a, b], "status": "skipped",
                                "missing": [c for c in (a, b) if c not in cells]})
                n_skip += 1
                continue
            res = audit_pair(cells[a], cells[b], spec["varies"], spec["licensed"])
            entries.append({"pair": [a, b],
                            "status": "pass" if res["pass"] else "FAIL",
                            "illicit_diffs": res["illicit_diffs"]})
            n_pass += res["pass"]
            n_fail += not res["pass"]
        results[contrast] = entries
    return {"status": "audited", "n_pass": n_pass, "n_fail": n_fail,
            "n_skipped": n_skip, "contrasts": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=Path, default=Path("configs"),
                    help="dir of committed *.yaml (mode=configs) or run tree (mode=runs)")
    ap.add_argument("--mode", choices=("configs", "runs"), default="configs",
                    help="audit committed configs (pre-run) or executed runs' resolved configs")
    ap.add_argument("--out", type=Path, default=Path("audits/c13"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if not CONTRAST_REGISTRY:
        report = {"status": "no_contrasts_registered",
                  "note": "CONTRAST_REGISTRY is filled at protocol freeze (Session 1)."}
        (args.out / "c13_report.json").write_text(json.dumps(report, indent=2))
        print("C13: no contrasts registered yet (expected in Session 0).")
        return 0

    cells = collect_cells(args.configs, args.mode)
    report = run_audit(cells)
    report["mode"] = args.mode
    report["source"] = str(args.configs)
    (args.out / "c13_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    for contrast, entries in report["contrasts"].items():
        for e in entries:
            a, b = e["pair"]
            if e["status"] == "FAIL":
                print(f"  FAIL  {contrast:<12} {a} vs {b}")
                for k, v in e["illicit_diffs"].items():
                    print(f"          {k}: {v['a']!r} != {v['b']!r}")
            elif e["status"] == "skipped":
                print(f"  skip  {contrast:<12} {a} vs {b}  (missing: {','.join(e['missing'])})")
            else:
                print(f"  ok    {contrast:<12} {a} vs {b}")
    print(f"\nC13 [{args.mode}]: {report['n_pass']} pass, {report['n_fail']} fail, "
          f"{report['n_skipped']} skipped")
    return 1 if report["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
