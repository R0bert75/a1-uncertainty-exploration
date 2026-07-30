"""Assert every committed config resolves under the frozen schema (spec §4 CI clause).

Spec §4 requires CI to check the "factorial config schema". The loader validates a config
when a run loads it, but nothing walks the committed set — so a config could be committed
broken and only fail at launch. This script closes that: it loads every ``configs/*.yaml``
through :func:`src.config.load_config` and reports the resolved identity of each.

Exit status is 1 if any config fails to resolve, so CI fails loudly.

Usage::

    python audits/config_schema_check.py --configs configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config as config_mod


def check_dir(configs_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(ok, failed)`` as lists of ``(filename, detail)``."""
    ok: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = config_mod.load_config(path)
        except Exception as exc:  # noqa: BLE001 — any failure is a schema failure here
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
        else:
            ok.append(
                (
                    path.name,
                    f"{cfg.method:<14} {cfg.env:<16} part {cfg.part}  "
                    f"cell {cfg.cell_id:<20} sha {cfg.config_sha256[:12]}",
                )
            )
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    args = ap.parse_args(argv)

    if not args.configs.is_dir():
        print(f"no such directory: {args.configs}", file=sys.stderr)
        return 1

    ok, failed = check_dir(args.configs)
    if not ok and not failed:
        print(f"no configs found in {args.configs}/ — nothing to check", file=sys.stderr)
        return 1

    for name, detail in ok:
        print(f"  OK    {name:<40} {detail}")
    for name, detail in failed:
        print(f"  FAIL  {name:<40} {detail}", file=sys.stderr)

    print(f"\n{len(ok)} config(s) resolved, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
