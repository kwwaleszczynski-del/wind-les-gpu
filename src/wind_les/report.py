"""Comparison tables and headline numbers: SCADA vs PyWake vs LES, single wake, costs.

Usage:
    compare_all(Path("results"), Path("../wind-power-curve-kelmarsh/data"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wind_les import compare

LES_N = 8
log = logging.getLogger(__name__)


def les_runs(results: Path, n: int = LES_N) -> dict[float, dict]:
    """Ratios table rebuilt into the per-direction structure ``compare.les_table`` expects."""
    ratios = pd.read_csv(results / f"les_ratios_D{n}.csv")
    return {float(wd): g.pivot(index="block", columns="turbine", values="s").to_numpy()
            for wd, g in ratios.groupby("wd")}


def compare_all(results: Path, data_dir: Path) -> dict:
    from wind_wake_shadow import validation

    rec = compare.scada_records(data_dir)
    py = compare.pywake_tables(rec)
    runs = les_runs(results)
    step = float(np.diff(sorted(runs)).min()) if len(runs) > 1 else 360.0
    les = compare.les_table(runs, validation.WS_GRID)
    tables = {"les": les, **{f"pywake_{k}": v for k, v in py.items()}}
    sectors = compare.sector_losses(rec, tables)
    # LES sampling error: the same comparison for each 10-minute block on its own
    n_blocks = min(v.shape[0] for v in runs.values())
    per_block = compare.sector_losses(rec, {f"b{b}": compare.les_table({wd: v[b:b + 1] for wd, v in runs.items()},
                                                                       validation.WS_GRID)
                                            for b in range(n_blocks)}, n_boot=10)
    blocks = per_block[[f"b{b}" for b in range(n_blocks)]]
    sectors["les_se"] = blocks.std(axis=1, ddof=1) / np.sqrt(n_blocks)
    sectors.to_csv(results / "sector_losses.csv", index=False, float_format="%.3f")
    eff = compare.efficiency_by_direction(rec, tables, width=5.0)
    eff.to_csv(results / "efficiency_by_direction.csv", float_format="%.4f")

    # what sampling the directions every `step` degrees costs, measured on the PyWake Gaussian table
    sub = compare.sector_losses(rec, {"gaussian_full": py["gaussian"],
                                      "gaussian_subsampled": compare.subsampled(py["gaussian"], step)}, n_boot=10)
    sub[["sector_deg", "gaussian_full", "gaussian_subsampled"]].to_csv(results / "direction_sampling_check.csv",
                                                                      index=False, float_format="%.3f")

    prec = json.loads((results / f"precursor_D{LES_N}.json").read_text())
    x_d = np.round(np.arange(1.0, 15.01, 0.25), 2)
    single = {"les_ti": compare.single_wake(prec["ti_hub"], x_d), "site_ti": compare.single_wake(rec["ti"], x_d)}
    single_df = pd.concat([single["les_ti"].assign(ti=round(prec["ti_hub"], 4)),
                           single["site_ti"].assign(ti=round(rec["ti"], 4))])
    single_df.to_csv(results / "single_wake_pywake.csv", index=False, float_format="%.5f")

    sec = sectors[sectors["sector_deg"].astype(str) != "all"]
    skill = {name: {"rmse_pp": round(float(np.sqrt(((sec[name] - sec["scada"]) ** 2).mean())), 2),
                    "bias_pp": round(float((sec[name] - sec["scada"]).mean()), 2),
                    "within_scada_ci": int(((sec[name] >= sec["scada_lo"]) & (sec[name] <= sec["scada_hi"])).sum())}
             for name in tables}
    runs_df = pd.read_csv(results / f"les_runs_D{LES_N}.csv")
    all_row = sectors[sectors["sector_deg"].astype(str) == "all"].iloc[0]
    summary = {
        "scada_records": int(all_row["records"]), "scada_days": int(all_row["days"]),
        "ambient_ti_scada": round(float(rec["ti"]), 4), "ti_les_hub": round(float(prec["ti_hub"]), 4),
        "les_direction_step_deg": step, "les_directions": len(runs),
        "wake_loss_pct": {k: round(float(all_row[k]), 2) for k in ["scada", "scada_lo", "scada_hi", *tables, "les_se"]},
        "sector_skill": skill,
        "sampling_bias_pct": round(float(sub.iloc[-1]["gaussian_subsampled"] - sub.iloc[-1]["gaussian_full"]), 3),
        "les_cost": {"cells": int(runs_df["cells"].iloc[0]), "median_wall_s": float(runs_df["wall_s"].median()),
                     "median_mlups": float(runs_df["mlups"].median()),
                     "total_gpu_h": float(runs_df["wall_s"].sum() / 3600),
                     "gpu_mb": float(runs_df["gpu_mb"].max())},
    }
    (results / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wake loss: %s; sector skill: %s", summary["wake_loss_pct"], skill)
    return summary
