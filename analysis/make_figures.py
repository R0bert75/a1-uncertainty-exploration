"""Rebuild every figure from ``logs/*.csv`` alone (gate C2).

This script is the *only* path from data to figures. It reads committed CSVs, never
in-memory training state, never a dashboard. ``make figures`` calls it. If a figure
cannot be produced from the CSVs on disk, it does not exist.

Session 0: the real per-figure logic is stubbed. What is wired now is the contract —
discover CSVs, validate the frozen schema, group by (part, env, metric), and emit one
placeholder line plot per group into ``figures/``. Later sessions replace
``_plot_group`` with the real estimand/battery figures; the data path does not change.

Usage:
    python analysis/make_figures.py --logs logs --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.conventions import BASE_FIELDS  # noqa: E402

# Frozen schema is owned by conventions.BASE_FIELDS — never duplicated here (C2).
REQUIRED_COLUMNS = set(BASE_FIELDS)


def load_logs(logs_dir: Path) -> pd.DataFrame:
    """Concatenate every CSV under ``logs_dir`` and validate the frozen schema."""
    csvs = sorted(p for p in logs_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs found in {logs_dir}/ — nothing to plot")
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SystemExit(f"{p.name} is missing required columns: {sorted(missing)}")
        df["__source__"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _plot_group(df: pd.DataFrame, part: str, env: str, metric: str, out_dir: Path) -> Path:
    """Placeholder figure: mean±band over seeds per method vs. step.

    Replaced in later sessions by the real §1.1 estimand and battery figures.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    for method, g in df.groupby("method"):
        agg = g.groupby("step")["value"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["step"], agg["mean"], marker="o", ms=3, label=str(method))
        if (agg["count"] > 1).any():
            lo = agg["mean"] - agg["std"].fillna(0)
            hi = agg["mean"] + agg["std"].fillna(0)
            ax.fill_between(agg["step"], lo, hi, alpha=0.15)
    ax.set_xlabel("environment step")
    ax.set_ylabel(metric)
    roles = ",".join(sorted(df["role"].unique()))
    ax.set_title(f"[PLACEHOLDER] part {part} · {env} · {metric}  (role: {roles})")
    ax.legend(fontsize=8, title="method")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = f"part{part}_{env}_{metric}".replace("/", "-")
    path = out_dir / f"{safe}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args(argv)

    df = load_logs(args.logs)
    written = []
    for (part, env, metric), g in df.groupby(["part", "env", "metric"]):
        written.append(_plot_group(g, str(part), str(env), str(metric), args.out))

    print(f"rebuilt {len(written)} figure(s) from {df['__source__'].nunique()} CSV(s):")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
