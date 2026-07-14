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
