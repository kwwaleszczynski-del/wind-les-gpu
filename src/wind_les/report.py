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

from wind_les import abl, compare

LES_N = 8
log = logging.getLogger(__name__)


def les_runs(results: Path, n: int = LES_N, realisation: int = 0) -> dict[float, dict]:
    """Ratios table rebuilt into the per-direction structure ``compare.les_table`` expects."""
    ratios = pd.read_csv(results / f"les_ratios_{abl.file_tag(n, realisation)}.csv")
    return {float(wd): g.pivot(index="block", columns="turbine", values="s").to_numpy()
            for wd, g in ratios.groupby("wd")}


def unwaked_by_half(turb: pd.DataFrame, min_records: int = 300) -> pd.DataFrame:
    """Normalised power of each turbine in the bins where neither the LES nor the Gaussian model sees a wake on it
    (ratio > 0.99), easterly against westerly winds, record-weighted. With a site factor that works in every
    direction, the SCADA would be close to 1 in both halves."""
    free = turb[(turb["les"] > 0.99) & (turb["pywake_gaussian"] > 0.99) & (turb["records"] >= min_records)].copy()
    free["winds"] = np.where(free["bin"] < 180, "easterly (0-175)", "westerly (180-355)")
    rows = []
    for (t, half), g in free.groupby(["turbine", "winds"]):
        w = g["records"]
        rows.append({"turbine": t, "winds": half, "bins": len(g), "records": int(w.sum()),
                     "scada": float((g["scada"] * w).sum() / w.sum()), "les": float((g["les"] * w).sum() / w.sum())})
    return pd.DataFrame(rows)


def _read_ti(results: Path, realisation: int, n: int = LES_N) -> float:
    return float(json.loads((results / f"precursor_{abl.file_tag(n, realisation)}.json").read_text())["ti_hub"])


def les_realisations(results: Path, n: int = LES_N) -> list[int]:
    """Inflow realisations with a complete direction sweep (the same directions as the main one)."""
    main = set(pd.read_csv(results / f"les_ratios_D{n}.csv")["wd"])
    out = [0]
    for path in sorted(results.glob(f"les_ratios_D{n}_r*.csv")):
        r = int(path.stem.rsplit("_r", 1)[1])
        if set(pd.read_csv(path)["wd"]) == main:
            out.append(r)
        else:
            log.warning("%s: incomplete sweep, not used", path.name)
    return out


def compare_all(results: Path, data_dir: Path) -> dict:
    from wind_wake_shadow import validation

    rec = compare.scada_records(data_dir)
    py = compare.pywake_tables(rec)
    reals = les_realisations(results)
    per_real = {r: les_runs(results, LES_N, r) for r in reals}
    # the LES table is the mean over all 10-minute blocks of all inflow realisations
    runs = {wd: np.vstack([per_real[r][wd] for r in reals]) for wd in per_real[0]}
    step = float(np.diff(sorted(runs)).min()) if len(runs) > 1 else 360.0
    les = compare.les_table(runs, validation.WS_GRID)
    tables = {"les": les, **{f"pywake_{k}": v for k, v in py.items()}}
    real_tables = ({f"les_r{r}": compare.les_table(per_real[r], validation.WS_GRID) for r in reals}
                   if len(reals) > 1 else {})
    sectors = compare.sector_losses(rec, {**tables, **real_tables})
    # standard error of the pooled LES loss from the scatter of all 10-minute blocks of all realisations; the
    # blocks of one realisation scatter at least as much as the realisations differ, so they count as independent
    block_tables = {f"r{r}b{b}": compare.les_table({wd: v[b:b + 1] for wd, v in per_real[r].items()},
                                                   validation.WS_GRID)
                    for r in reals for b in range(min(v.shape[0] for v in per_real[r].values()))}
    blocks = compare.sector_losses(rec, block_tables, n_boot=10)[list(block_tables)]
    sectors["les_block_se"] = blocks.std(axis=1, ddof=1) / np.sqrt(len(block_tables))
    check = {}
    if real_tables:
        # the check: the difference between two realisations against what their block scatter predicts (sectors)
        a, b = reals[:2]
        se = {r: blocks.filter(regex=f"^r{r}b").std(axis=1, ddof=1) / np.sqrt(blocks.filter(regex=f"^r{r}b").shape[1])
              for r in (a, b)}
        in_sector = (sectors["sector_deg"].astype(str) != "all").to_numpy()
        diff = (sectors[f"les_r{a}"] - sectors[f"les_r{b}"])[in_sector]
        check = {"rms_difference_pp": round(float(np.sqrt((diff ** 2).mean())), 3),
                 "max_difference_pp": round(float(diff.abs().max()), 3),
                 "rms_predicted_by_blocks_pp": round(float(np.sqrt((se[a] ** 2 + se[b] ** 2)[in_sector].mean())), 3)}
    sectors.to_csv(results / "sector_losses.csv", index=False, float_format="%.3f")
    eff = compare.efficiency_by_direction(rec, tables, width=5.0)
    eff.to_csv(results / "efficiency_by_direction.csv", float_format="%.4f")
    tsl = compare.turbine_sector_losses(rec, tables)
    tsl.to_csv(results / "turbine_sector_losses.csv", index=False, float_format="%.3f")
    turb = compare.turbine_efficiency(rec, tables, width=5.0)
    turb.to_csv(results / "turbine_efficiency.csv", index=False, float_format="%.4f")
    unwaked_by_half(turb).to_csv(results / "unwaked_by_half.csv", index=False, float_format="%.4f")

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
             for name in {**tables, **real_tables}}
    # cost from the main sweep alone: a later realisation may have shared the GPU with other work
    runs_df = pd.read_csv(results / f"les_runs_{abl.file_tag(LES_N, 0)}.csv")
    ti_les = [_read_ti(results, r) for r in reals]
    all_row = sectors[sectors["sector_deg"].astype(str) == "all"].iloc[0]
    loss_keys = ["scada", "scada_lo", "scada_hi", *tables, *real_tables, "les_block_se"]
    east = tsl[tsl["sector_deg"].isin([60, 90, 120, 150])]
    summary = {
        "scada_records": int(all_row["records"]), "scada_days": int(all_row["days"]),
        "ambient_ti_scada": round(float(rec["ti"]), 4), "ti_les_hub": [round(t, 4) for t in ti_les],
        "les_direction_step_deg": step, "les_directions": len(runs), "les_realisations": len(reals),
        "les_blocks": len(block_tables), "les_realisation_check": check,
        "wake_loss_pct": {k: round(float(all_row[k]), 2) for k in loss_keys},
        "sector_skill": skill,
        # which turbines make the easterly sectors (60-150 deg) too lossy: mean turbine-loss error per model
        "easterly_turbine_error_pp": {
            name: {f"T{t}": round(float(v), 2)
                   for t, v in (east[name] - east["scada"]).groupby(east["turbine"]).mean().items()}
            for name in tables},
        "sampling_bias_pct": round(float(sub.iloc[-1]["gaussian_subsampled"] - sub.iloc[-1]["gaussian_full"]), 3),
        "les_cost": {"cells": int(runs_df["cells"].iloc[0]), "median_wall_s": float(runs_df["wall_s"].median()),
                     "median_mlups": float(runs_df["mlups"].median()),
                     "total_gpu_h": float(runs_df["wall_s"].sum() / 3600),
                     "gpu_mb": float(runs_df["gpu_mb"].max())},
    }
    (results / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wake loss: %s; sector skill: %s", summary["wake_loss_pct"], skill)
    return summary
