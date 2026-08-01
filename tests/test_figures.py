"""C2 round-trip: dummy CSV -> make_figures produces a PNG, reading logs only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import make_dummy_logs, make_figures  # noqa: E402


def test_dummy_logs_then_figures(tmp_path):
    logs = tmp_path / "logs"
    figs = tmp_path / "figures"
    logs.mkdir()
    csv_path = logs / "dummy_smoke.csv"

    rc = make_dummy_logs.main(["--out", str(csv_path)])
    assert rc == 0 and csv_path.exists()

    rc = make_figures.main(["--logs", str(logs), "--out", str(figs)])
    assert rc == 0
    pngs = list(figs.glob("*.png"))
    assert pngs, "make_figures produced no PNG from the dummy CSV"


def test_figures_rejects_bad_schema(tmp_path):
    import pandas as pd
    import pytest

    logs = tmp_path / "logs"
    logs.mkdir()
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(logs / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        make_figures.load_logs(logs)


# --------------------------------------------------------------------------- #
# Depth-axis pilot figure (step 6)
# --------------------------------------------------------------------------- #


def _deepsea_rows(run_id, size, seed, discover_at, n_ck=5):
    """One run's rows: a cumulative discovery indicator plus the deep_sea_size descriptor.

    ``discover_at=None`` means the seed never discovers within the budget.
    """
    base = dict(
        run_id=run_id,
        role="development",
        part="A",
        method=run_id.split("_")[0],
        env="deep_sea",
        size_class="development",
        seed=seed,
        config_sha256="0" * 64,
        axis="online",
    )
    rows = []
    for ck in range(n_ck):
        got = discover_at is not None and ck >= discover_at
        rows.append(
            {**base, "step": (ck + 1) * size, "checkpoint": ck, "is_t0": ck == 0,
             "metric": "discovery_prob", "value": float(got)}
        )
    rows.append(
        {**base, "step": n_ck * size, "checkpoint": n_ck - 1, "is_t0": False,
         "metric": "deep_sea_size", "value": float(size)}
    )
    return rows


def _depth_logs(tmp_path):
    import pandas as pd

    rows = []
    # ddqn: discovers at N=10, fails at N=30. bdqn: discovers at both.
    for seed in range(4):
        rows += _deepsea_rows("ddqn_a_N10", 10, seed, discover_at=2)
        rows += _deepsea_rows("ddqn_a_N30", 30, seed, discover_at=None)
        rows += _deepsea_rows("bdqn_a_N10", 10, seed, discover_at=1)
        rows += _deepsea_rows("bdqn_a_N30", 30, seed, discover_at=3)
    logs = tmp_path / "logs"
    logs.mkdir()
    pd.DataFrame(rows).to_csv(logs / "depth.csv", index=False)
    return logs


def test_depth_figure_reads_size_from_the_metric_row_not_the_run_id(tmp_path):
    """N must come from the ``deep_sea_size`` row, so cells whose run_id lacks ``_N`` count.

    The committed factorial cells are named ``cell_<rule>_<prior>_K<k>_deepsea_dev`` with no
    ``_N`` suffix; only the tuning-search runs carry one. Recovering N from the run_id would
    silently drop every committed cell from the figure -- a selection effect on the x-axis.
    """
    import pandas as pd

    from analysis import make_figures

    rows = []
    for seed in range(3):
        rows += _deepsea_rows("cell_episodic_off_K10_deepsea_dev", 30, seed, discover_at=1)
    df = pd.DataFrame(rows)
    table = make_figures._depth_table(df)
    assert not table.empty, "a run with no _N suffix was dropped from the depth table"
    assert set(table["deep_sea_size"]) == {30}


def test_depth_figure_uses_the_terminal_indicator_not_the_auc(tmp_path):
    """Reported discovery is "ever discovered within budget" (prereg 1.1), not discovery-AUC.

    A seed discovering at the last checkpoint contributes 1.0 here, while its tuning AUC
    would be near 0. Conflating the two would report the tuning objective as if it were the
    primary outcome.
    """
    import pandas as pd

    from analysis import make_figures

    late = pd.DataFrame(_deepsea_rows("x_N10", 10, 0, discover_at=4, n_ck=5))
    table = make_figures._depth_table(late)
    assert float(table["discovered"].iloc[0]) == 1.0


def test_depth_figure_is_written_and_labelled_pilot(tmp_path):
    from analysis import make_figures

    logs = _depth_logs(tmp_path)
    figs = tmp_path / "figures"
    rc = make_figures.main(["--logs", str(logs), "--out", str(figs)])
    assert rc == 0
    depth = figs / "pilot_partA_discovery_vs_depth.png"
    assert depth.exists(), "no depth-axis figure produced"
    assert make_figures.PILOT_LABEL.startswith("PILOT"), "figure must carry a pilot label"


def test_deep_sea_size_never_gets_its_own_panel(tmp_path):
    """It is an env descriptor, not an estimand -- a panel of it would be meaningless."""
    from analysis import make_figures

    logs = _depth_logs(tmp_path)
    figs = tmp_path / "figures"
    make_figures.main(["--logs", str(logs), "--out", str(figs)])
    assert not list(figs.glob("*deep_sea_size*.png"))
