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


def _plot_group(
    df: pd.DataFrame, part: str, env: str, metric: str, axis: str, out_dir: Path
) -> Path:
    """Placeholder figure: mean±band over seeds per method vs. step.

    One panel per ``(part, env, metric, axis)``. The axis is part of the grouping key because
    the two frozen reporting axes (spec §5: ``online`` primary, ``frozen_policy`` secondary)
    share the ``episode_return`` metric name — grouping without it would overplot two
    different estimands on one panel.

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
    ax.set_title(f"[PLACEHOLDER] part {part} · {env} · {metric} · {axis}  (role: {roles})")
    ax.legend(fontsize=8, title="method")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = f"part{part}_{env}_{metric}_{axis}".replace("/", "-")
    path = out_dir / f"{safe}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


PILOT_LABEL = "PILOT (development tier) — not a confirmatory result"


def _depth_table(df: pd.DataFrame) -> pd.DataFrame:
    """Join every DeepSea run's terminal discovery indicator to the depth N it ran at.

    N is recovered from the ``deep_sea_size`` **metric row** the episode lane emits once per
    run, not from a column: the frozen CSV schema (gate C2) has no env-parameter column and
    the committed cell ``run_id``s do not encode N. Runs predating that row are dropped with
    a count rather than guessed at, because inferring N from a run_id suffix would silently
    admit only the tuning-search runs (whose ids carry ``_N<size>``) and quietly exclude the
    committed factorial cells (whose ids do not) — a selection effect on the figure's x-axis.

    Discovery per (run, seed) is the **terminal** value of the cumulative ``discovery_prob``
    indicator, i.e. "did this seed ever discover within the frozen episode budget" — the
    §1.1 primary outcome. That is deliberately NOT the discovery-AUC used to *tune* the
    backbone: AUC rewards discovering early, which is the right objective for ranking
    candidates but the wrong quantity to report against depth.
    """
    size_rows = df[df["metric"] == "deep_sea_size"]
    sizes = size_rows.groupby("run_id")["value"].first().astype(int)

    disc = df[(df["metric"] == "discovery_prob") & (df["axis"] == "online")]
    if disc.empty or sizes.empty:
        return pd.DataFrame(columns=["deep_sea_size", "method", "seed", "discovered"])

    # Terminal indicator per (run_id, seed): the row at the largest checkpoint.
    idx = disc.groupby(["run_id", "seed"])["checkpoint"].idxmax()
    terminal = disc.loc[idx, ["run_id", "seed", "method", "value"]].copy()
    terminal = terminal.rename(columns={"value": "discovered"})
    terminal["deep_sea_size"] = terminal["run_id"].map(sizes)

    dropped = int(terminal["deep_sea_size"].isna().sum())
    if dropped:
        print(f"  [depth figure] dropped {dropped} run-seed(s) with no deep_sea_size row")
    terminal = terminal.dropna(subset=["deep_sea_size"])
    terminal["deep_sea_size"] = terminal["deep_sea_size"].astype(int)
    return terminal


def plot_discovery_vs_depth(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Pilot figure: discovery probability within budget vs DeepSea depth N, per method.

    The protocol's term is *discovery probability within the pre-registered episode budget*
    (§1.1); "solve-vs-depth" is informal shorthand that appears nowhere in the frozen text,
    so the axis label uses the protocol's wording. Carries a mandatory PILOT banner: these
    are development-tier runs, and the confirmatory claim is a five-size aggregate this
    figure is not.
    """
    table = _depth_table(df)
    if table.empty:
        return None

    agg = (
        table.groupby(["method", "deep_sea_size"])["discovered"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    for method, g in agg.groupby("method"):
        g = g.sort_values("deep_sea_size")
        # Wilson score interval: the estimand is a proportion of a handful of seeds, where a
        # normal-approximation band runs outside [0, 1] and collapses to zero width at 0/n
        # and n/n -- exactly the regime this figure is about (eps-greedy failing at depth).
        n, k = g["count"].to_numpy(float), g["sum"].to_numpy(float)
        z = 1.96
        denom = 1.0 + z**2 / n
        centre = (k / n + z**2 / (2 * n)) / denom
        half = (z / denom) * ((k / n * (1 - k / n) / n + z**2 / (4 * n**2)) ** 0.5)
        ax.plot(g["deep_sea_size"], g["mean"], marker="o", ms=4, label=str(method))
        ax.fill_between(g["deep_sea_size"], centre - half, centre + half, alpha=0.15)

    ax.set_xlabel("DeepSea depth $N$")
    ax.set_ylabel("discovery probability within budget")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    seeds = int(table.groupby(["method", "deep_sea_size"]).size().max())
    ax.set_title(
        f"{PILOT_LABEL}\ndiscovery probability vs depth  (\u2264{seeds} seeds/point, "
        "Wilson 95% band)",
        fontsize=9,
    )
    ax.legend(fontsize=8, title="method")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pilot_partA_discovery_vs_depth.png"
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
    # deep_sea_size is an env descriptor, not an estimand: it must not get its own panel.
    panel_df = df[df["metric"] != "deep_sea_size"]
    for (part, env, metric, axis), g in panel_df.groupby(["part", "env", "metric", "axis"]):
        written.append(_plot_group(g, str(part), str(env), str(metric), str(axis), args.out))

    depth_fig = plot_discovery_vs_depth(df[df["env"] == "deep_sea"], args.out)
    if depth_fig is not None:
        written.append(depth_fig)

    print(f"rebuilt {len(written)} figure(s) from {df['__source__'].nunique()} CSV(s):")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
