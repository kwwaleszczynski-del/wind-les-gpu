"""Wake losses per direction sector: SCADA, PyWake and the LES, processed the same way.

The SCADA side and the PyWake models are those of wind-wake-shadow (imported, not re-implemented):

    records   10-minute intervals 2016-2021 in which all six turbines run in partial load (power > 20 kW,
              pitch < 5 deg) and the free-stream reference wind speed is 5-10 m/s
    reference turbines in free stream by the IEC 61400-12-1 disturbed-sector rule; site factors remove hub
              height and terrain differences; reference speed = inverse catalogue curve of their mean power
    models    every record is run at its own direction and reference speed, then normalised identically

The LES enters through the same door: a power table P(wd, ws, turbine). Each LES run gives the free-stream
equivalent speed of every turbine relative to the inflow hub speed, s_i(wd), per 10-minute block. Because the
LES is Reynolds-number independent and the disks use a constant C_T in partial load, the flow scales with the
wind speed, so P_i(wd, ws) = mean over blocks of P_curve(ws s_i). Directions between LES runs are interpolated
linearly.

Sector loss = 1 - sum(normalised power of all turbines) / sum(6 x reference power) over the records of a
sector. The SCADA interval is a day-block bootstrap (10-minute records within a day are strongly correlated).

Usage:
    rec = scada_records(data_dir)                       # needs the [compare] extra (wind-wake-shadow, PyWake)
    tables = pywake_tables(rec)                         # {'jensen': (360, n_ws, 6), ...}
    les = les_table({240.0: ratios_240, ...}, ws_grid)   # (360, n_ws, 6)
    table = sector_losses(rec, {"pywake gaussian": tables["gaussian"], "les": les})
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wind_les import turbine

SECTOR_DEG = 30.0
N_BOOT = 1000
log = logging.getLogger(__name__)


def _wws():
    try:
        from wind_wake_shadow import kelmarsh, pipeline, validation, wake
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("the SCADA / PyWake comparison needs the [compare] extra: "
                          "pip install -e '.[compare]'") from exc
    return kelmarsh, pipeline, validation, wake


def scada_records(data_dir: Path, years=range(2016, 2022)) -> dict:
    """Filtered SCADA records with normalised powers, as in wind-wake-shadow's validation."""
    kelmarsh, pipeline, validation, _ = _wws()
    farm = pipeline.load_farm(Path(data_dir))
    scada = kelmarsh.load_scada(list(years), Path(data_dir))
    climate, ti = pipeline.wind_climate_from_scada(scada, farm)
    rec = validation.select_records(scada, farm.static, farm.x, farm.y)
    ws_ref = validation.reference_wind_speed(validation.normalise(rec["power"], rec["free"])["ref"])
    keep = (ws_ref >= validation.WS_RANGE[0]) & (ws_ref <= validation.WS_RANGE[1])
    rec = {k: (v if k == "turbines" else v[keep]) for k, v in rec.items()}
    rec.update(ws_ref=ws_ref[keep], days=pd.DatetimeIndex(rec["time"]).normalize(), farm=farm, climate=climate,
               ti=ti, measured=validation.normalise(rec["power"], rec["free"]))
    log.info("SCADA: %d records on %d days, ambient TI %.3f", len(rec["wd"]), rec["days"].nunique(), ti)
    return rec


def pywake_tables(rec: dict) -> dict[str, np.ndarray]:
    _, _, validation, wake = _wws()
    farm = rec["farm"]
    wts, types = farm.turbines()
    site = wake.make_site(rec["climate"], ti=rec["ti"])
    return {name: wake.power_table(wake.make_model(name, site, wts), farm.x, farm.y, types, validation.WD_GRID,
                                   validation.WS_GRID) for name in wake.MODELS}


def les_table(runs: dict[float, np.ndarray], ws_grid, wd_grid=np.arange(360.0)) -> np.ndarray:
    """P(wd, ws, turbine) in kW from the LES speed ratios {wd: (blocks, turbines)}, periodic linear in wd."""
    wds = np.array(sorted(runs))
    per_run = np.array([turbine.power_kw(np.asarray(ws_grid)[None, :, None] * runs[wd][:, None, :]).mean(axis=0)
                        for wd in wds])                          # (n_runs, n_ws, n_turbines)
    ext_wd = np.concatenate([wds - 360.0, wds, wds + 360.0])
    ext = np.concatenate([per_run] * 3).reshape(3 * len(wds), -1)
    flat = np.stack([np.interp(wd_grid, ext_wd, ext[:, c]) for c in range(ext.shape[1])], axis=1)
    return flat.reshape(len(wd_grid), len(ws_grid), per_run.shape[2])


def sector_of(wd, width: float = SECTOR_DEG) -> np.ndarray:
    return (np.floor(((np.asarray(wd) + width / 2) % 360.0) / width) * width).astype(int)


def _sums(norm: dict, sector: np.ndarray, groups) -> pd.DataFrame:
    total, ref = norm["p"].sum(axis=1), norm["ref"] * norm["p"].shape[1]
    return pd.DataFrame({"total": total, "ref": ref, "sector": sector, "group": groups}).groupby(
        ["group", "sector"])[["total", "ref"]].sum()


def model_norm(rec: dict, table: np.ndarray) -> dict:
    _, _, validation, _ = _wws()
    return validation.normalise(validation.model_records(table, rec["wd"], rec["ws_ref"]), rec["free"])


def sector_losses(rec: dict, tables: dict[str, np.ndarray], width: float = SECTOR_DEG, n_boot: int = N_BOOT,
                  seed: int = 0) -> pd.DataFrame:
    """Wake loss [%] per sector and overall for SCADA (with a day-block bootstrap interval) and each table."""
    sector = sector_of(rec["wd"], width)
    rows = {}
    day_codes, days = pd.factorize(rec["days"])
    meas = _sums(rec["measured"], sector, day_codes)
    by_sector = meas.groupby("sector").sum()
    rows["scada"] = 100 * (1 - by_sector["total"] / by_sector["ref"])
    rows["scada_all"] = 100 * (1 - by_sector["total"].sum() / by_sector["ref"].sum())
    # bootstrap: resample days with replacement and recompute every sector loss
    tot = meas["total"].unstack("sector", fill_value=0.0).reindex(range(len(days)), fill_value=0.0)
    ref = meas["ref"].unstack("sector", fill_value=0.0).reindex(range(len(days)), fill_value=0.0)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_boot)   # (n_boot, n_days)
    t_b, r_b = counts @ tot.to_numpy(), counts @ ref.to_numpy()
    loss_b = 100 * (1 - t_b / r_b)
    all_b = 100 * (1 - t_b.sum(axis=1) / r_b.sum(axis=1))
    records = pd.Series(sector).value_counts().reindex(tot.columns).values
    out = pd.DataFrame({"sector_deg": tot.columns, "records": records,
                        "days": [int(((ref[c] > 0)).sum()) for c in tot.columns],
                        "scada": rows["scada"].reindex(tot.columns).values,
                        "scada_lo": np.percentile(loss_b, 2.5, axis=0),
                        "scada_hi": np.percentile(loss_b, 97.5, axis=0)})
    total = {"sector_deg": "all", "records": len(sector), "days": len(days), "scada": rows["scada_all"],
             "scada_lo": float(np.percentile(all_b, 2.5)), "scada_hi": float(np.percentile(all_b, 97.5))}
    for name, table in tables.items():
        s = _sums(model_norm(rec, table), sector, np.zeros(len(sector), int)).groupby("sector").sum()
        out[name] = (100 * (1 - s["total"] / s["ref"])).reindex(out["sector_deg"]).values
        total[name] = float(100 * (1 - s["total"].sum() / s["ref"].sum()))
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def turbine_sector_losses(rec: dict, tables: dict[str, np.ndarray], width: float = SECTOR_DEG) -> pd.DataFrame:
    """Loss of each turbine against the free-stream reference [%] per sector, for SCADA and each table.

    The farm loss of a sector is the mean of its six turbine losses, so this splits a difference between a model
    and the SCADA into the turbines that cause it.
    """
    sector = sector_of(rec["wd"], width)
    norms = {"scada": rec["measured"], **{name: model_norm(rec, t) for name, t in tables.items()}}
    out = None
    for name, norm in norms.items():
        p = pd.DataFrame(norm["p"], columns=range(1, norm["p"].shape[1] + 1)).assign(ref=norm["ref"], sector=sector)
        sums = p.groupby("sector").sum()
        sums.loc["all"] = p.drop(columns="sector").sum()
        loss = 100 * (1 - sums.drop(columns="ref").div(sums["ref"], axis=0))
        long = loss.rename_axis("sector_deg").reset_index().melt(id_vars="sector_deg", var_name="turbine",
                                                                 value_name=name)
        out = long if out is None else out.merge(long, on=["sector_deg", "turbine"])
    return out


def turbine_efficiency(rec: dict, tables: dict[str, np.ndarray], width: float = 5.0) -> pd.DataFrame:
    """Normalised power of each turbine (power / free-stream reference) per direction bin, SCADA with a 95 % CI."""
    _, _, validation, _ = _wws()
    out = validation.binned_ratio(rec["measured"], rec["wd"], rec["days"], width)
    out = out.rename(columns={"ratio": "scada", "ci95": "scada_ci95", "n": "records"})
    for name, table in tables.items():
        out[name] = validation.binned_ratio(model_norm(rec, table), rec["wd"], width=width)["ratio"]
    out = out.reset_index()
    out["turbine"] += 1
    return out


def efficiency_by_direction(rec: dict, tables: dict[str, np.ndarray], width: float = 2.0) -> pd.DataFrame:
    """Farm efficiency per direction bin for SCADA and each table (for the direction plot)."""
    _, _, validation, _ = _wws()
    out = {"scada": validation.farm_efficiency(rec["measured"], rec["wd"], width)}
    for name, table in tables.items():
        out[name] = validation.farm_efficiency(model_norm(rec, table), rec["wd"], width)
    df = pd.DataFrame(out)
    df["records"] = pd.Series(validation.direction_bins(rec["wd"], width)).value_counts()
    return df.rename_axis("wd_bin")


def subsampled(table: np.ndarray, step_deg: float) -> np.ndarray:
    """A 1-degree table sampled every ``step_deg`` and interpolated back (the LES direction resolution)."""
    wd = np.arange(table.shape[0], dtype=float)
    keep = np.arange(0.0, 360.0, step_deg)
    ext_wd = np.concatenate([keep - 360, keep, keep + 360])
    idx = np.round(keep).astype(int) % 360
    ext = np.concatenate([table[idx]] * 3)
    flat = ext.reshape(len(ext_wd), -1)
    return np.stack([np.interp(wd, ext_wd, flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape(table.shape)


def single_wake(ti: float, x_d, hub_m: float = 78.5, ws: float = 8.0) -> pd.DataFrame:
    """Hub-height centreline speed deficit 1 - U/U_inf behind one MM92 for each PyWake model."""
    from py_wake import HorizontalGrid
    from wind_wake_shadow import turbine as wt_mod
    _, _, _, wake = _wws()
    wts, types = wt_mod.farm_turbines([hub_m])
    climate = pd.DataFrame({"freq": [1.0], "A": [9.0], "k": [2.0]})
    site = wake.make_site(climate, ti=ti)
    xs = np.asarray(x_d, dtype=float) * turbine.DIAMETER_M
    out = {"x_d": np.asarray(x_d, dtype=float)}
    for name in wake.MODELS:
        model = wake.make_model(name, site, wts)
        sim = model([0.0], [0.0], type=[0], wd=[270.0], ws=[ws])
        fm = sim.flow_map(HorizontalGrid(x=xs, y=[0.0], h=hub_m), wd=[270.0], ws=[ws])
        out[name] = 1.0 - (fm.WS_eff / fm.WS).values.squeeze()
    return pd.DataFrame(out)
