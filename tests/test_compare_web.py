"""LES power table, sector aggregation, figures and the web page on synthetic inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from wind_les import compare, plots, turbine, web

WS = np.arange(4.0, 11.01, 0.25)


def test_sector_of_centred_bins():
    assert list(compare.sector_of([0, 14.9, 15.0, 344.0, 359.0, 200.0])) == [0, 0, 30, 330, 0, 210]


def test_les_table_interpolates_periodically():
    runs = {0.0: np.ones((2, 6)), 180.0: np.full((2, 6), 0.8)}
    t = compare.les_table(runs, WS)
    assert t.shape == (360, len(WS), 6)
    j = int(np.flatnonzero(WS == 8.0)[0])
    assert t[0, j, 0] == pytest.approx(turbine.power_kw(8.0))
    assert t[180, j, 0] == pytest.approx(turbine.power_kw(6.4))
    assert t[90, j, 0] == pytest.approx(t[270, j, 0])            # half-way both ways round


def test_les_table_averages_power_over_blocks():
    runs = {0.0: np.array([[0.9] * 6, [1.1] * 6])}
    t = compare.les_table(runs, WS)
    j = int(np.flatnonzero(WS == 8.0)[0])
    assert t[123, j, 3] == pytest.approx(0.5 * (turbine.power_kw(7.2) + turbine.power_kw(8.8)))


def test_subsampled_keeps_sampled_directions():
    rng = np.random.default_rng(0)
    table = rng.random((360, 3, 2))
    sub = compare.subsampled(table, 10.0)
    assert np.allclose(sub[::10], table[::10])
    assert np.allclose(sub[5], 0.5 * (table[0] + table[10]))


def test_sector_losses_bootstrap():
    validation = pytest.importorskip("wind_wake_shadow.validation")
    n = 400
    rng = np.random.default_rng(1)
    wd = rng.uniform(0, 360, n)
    free = np.ones((n, 6), bool)
    free[:, 5] = wd < 180              # turbine 6 is waked for half of the directions
    power = np.full((n, 6), 1000.0)
    power[~free[:, 5], 5] = 800.0
    rec = {"wd": wd, "ws_ref": np.full(n, 8.0), "free": free,
           "days": pd.DatetimeIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) // 20, "D")),
           "measured": validation.normalise(power, free)}
    table = np.zeros((360, len(validation.WS_GRID), 6))
    table[:] = turbine.power_kw(validation.WS_GRID)[None, :, None]
    out = compare.sector_losses(rec, {"flat": table}, n_boot=50)
    total = out[out["sector_deg"].astype(str) == "all"].iloc[0]
    assert total["scada"] == pytest.approx(100 * 200 * (~free[:, 5]).sum() / (6000 * n), rel=1e-6)
    assert total["scada_lo"] <= total["scada"] <= total["scada_hi"]
    assert total["flat"] == pytest.approx(0.0, abs=1e-9)


def _fake_results(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    rows = [{"sector_deg": s, "records": 100, "days": 10, "scada": 5.0, "scada_lo": 4.0, "scada_hi": 6.0, "les": 6.0,
             "pywake_gaussian": 5.5, "pywake_jensen": 7.0, "pywake_turbopark": 7.5} for s in range(0, 360, 30)]
    rows.append({**rows[0], "sector_deg": "all"})
    pd.DataFrame(rows).to_csv(res / "sector_losses.csv", index=False)
    summary = {"wake_loss_pct": {"scada": 9.2, "scada_lo": 9.0, "scada_hi": 9.5, "les": 10.0, "pywake_gaussian": 9.7},
               "les_cost": {"median_mlups": 1000, "median_wall_s": 80}}
    (res / "run.json").write_text(json.dumps(summary))
    return res


def test_web_page_builds_without_frames(tmp_path):
    res = _fake_results(tmp_path)
    page = web.build(res, tmp_path / "docs").read_text(encoding="utf-8")
    assert "__DATA__" not in page and '"sector_deg":"all"' in page
    assert "<title>Kelmarsh GPU LES</title>" in page


def test_sector_figure(tmp_path):
    res = _fake_results(tmp_path)
    plots.sector_losses(res, tmp_path / "img")
    assert (tmp_path / "img" / "sector_losses.png").stat().st_size > 10_000
