"""Study steps: verification, precursor, disk calibration, direction sweep, single wake, grid study, comparison.

Small tables go to results/ (kept in git); solver states and raw runs to results/cache/ (not kept).

Usage:
    step_verify(Path("results"), "cuda:0")
    setup = abl.Setup(cells_per_d=8)
    step_prepare(setup, Path("results"), "cuda:0")
    step_sweep(setup, Path("results"), "cuda:0", directions=np.arange(0, 360, 5))
    step_compare(Path("results"), data_dir)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp

from wind_les import abl, farm, layout, turbine, verify

DIRECTION_STEP = 5.0
FRAME_DIRECTIONS = (240.0, 300.0)
FRAMES_S = 240.0
GRID_DIRECTIONS = (240.0, 300.0)
RESOLUTIONS = (8, 12, 16)
PRECURSOR_SPIN_S = 6000.0
PRECURSOR_STATS_S = 1800.0
log = logging.getLogger(__name__)


def cache(results: Path, *parts) -> Path:
    p = Path(results, "cache", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def gpu_memory_mb(device: str) -> float | None:
    """Memory in use on the device (pool + allocations), from the CUDA driver; None on CPU."""
    if not str(device).startswith("cuda"):
        return None
    d = wp.get_device(device)
    return (d.total_memory - d.free_memory) / 2 ** 20


# --- verification ---------------------------------------------------------------------------------------------

def step_verify(results: Path, device: str) -> dict:
    tgv = pd.DataFrame([verify.taylor_green(n, device) for n in (16, 32, 64, 128)])
    pois = pd.DataFrame([{k: v for k, v in verify.poiseuille(nz, 0.1, device).items() if np.isscalar(v)}
                         for nz in (8, 16, 32)])
    rows, lines = [], []
    for n in (8, 16):
        for ctp in (0.5, 1.0, 1.5, 2.0, 2.5):
            r = turbine.isolated_disk(n, ctp, device)
            rows.append({"cells_per_d": n, "ct_prime": ctp, "M": r["M"], "a": r["a"], "ct": r["ct"], "r": r["r"],
                         "a_theory": r["theory"]["a"], "ct_theory": r["theory"]["ct"]})
            lines.append(pd.DataFrame({"cells_per_d": n, "ct_prime": ctp, "x_d": r["x_d"], "u": r["centreline"]}))
    disk = pd.DataFrame(rows)
    sizes = [(64, 64, 64), (128, 96, 64), (192, 160, 64), (256, 192, 96), (320, 256, 128), (400, 300, 160)]
    bench = pd.DataFrame(verify.benchmark_sizes(device, sizes))
    results.mkdir(parents=True, exist_ok=True)
    tgv.to_csv(results / "verify_taylor_green.csv", index=False, float_format="%.6g")
    pois.to_csv(results / "verify_poiseuille.csv", index=False, float_format="%.6g")
    disk.to_csv(results / "verify_disk.csv", index=False, float_format="%.5f")
    pd.concat(lines).to_csv(results / "verify_disk_centreline.csv", index=False, float_format="%.4f")
    bench.to_csv(results / "benchmark.csv", index=False, float_format="%.4g")
    log.info("verify: TGV nu ratio %s, Poiseuille err %s, peak %.0f MLUPS", tgv["nu_ratio"].round(4).tolist(),
             pois["max_rel_err"].round(4).tolist(), bench["mlups"].max())
    return {"tgv": tgv, "poiseuille": pois, "disk": disk, "benchmark": bench}


# --- precursor and disk calibration ---------------------------------------------------------------------------

def step_prepare(setup: abl.Setup, results: Path, device: str, spin_s: float = PRECURSOR_SPIN_S,
                 stats_s: float = PRECURSOR_STATS_S) -> dict:
    """Disk calibration and a spun-up precursor at this resolution."""
    n = setup.cells_per_d
    cal = turbine.calibrate(n, device)
    cals = _read_json(results / "disk_calibration.json")
    cals[str(n)] = cal
    _write_json(results / "disk_calibration.json", cals)

    geo = farm.geometry(setup)
    pre = abl.make_precursor(setup, geo.ny, device)
    hist = abl.spin_up(pre, setup, int(spin_s / setup.dt()), every=max(1000, int(100 / setup.dt())))
    stats = abl.profile_stats(pre, setup, int(stats_s / setup.dt()), every=max(10, int(5 / setup.dt())))
    np.save(cache(results, f"precursor_D{n}.npy"), pre.save())
    prof = pd.DataFrame({k: stats[k] for k in ("z_m", "u", "su", "sv", "sw", "uw")})
    prof.to_csv(results / f"precursor_profile_D{n}.csv", index=False, float_format="%.6g")
    pd.DataFrame(hist, columns=["step", "u_hub"]).to_csv(cache(results, f"precursor_spinup_D{n}.csv"), index=False)
    info = {"cells_per_d": n, "dx_m": setup.dx, "nz": setup.nz, "ny": geo.ny, "nx": pre.nx, "z0_m": setup.z0_m,
            "u_hub_lb": stats["u_hub"], "ti_hub": stats["ti_hub"], "ustar_lb": setup.ustar_lb,
            "dt_s": setup.dt(stats["u_hub"]), "spin_s": spin_s, "stats_s": stats_s}
    _write_json(results / f"precursor_D{n}.json", info)
    log.info("precursor D/%d: hub %.4f (lattice), TI %.3f", n, stats["u_hub"], stats["ti_hub"])
    return info


def load_prepared(setup: abl.Setup, results: Path) -> tuple[np.ndarray, dict, dict]:
    n = setup.cells_per_d
    state = np.load(cache(results, f"precursor_D{n}.npy"))
    info = _read_json(results / f"precursor_D{n}.json")
    cal = _read_json(results / "disk_calibration.json")[str(n)]
    return state, info, cal


# --- farm runs ------------------------------------------------------------------------------------------------

def run_path(results: Path, n: int, wd: float) -> Path:
    return cache(results, f"les_D{n}", f"wd_{wd:05.1f}.npz")


def step_empty(setup: abl.Setup, results: Path, device: str) -> dict:
    """The farm domain without turbines: undisturbed inflow at every possible rotor position."""
    state, info, cal = load_prepared(setup, results)
    out = farm.run_empty(setup, farm.geometry(setup), cal, state, info["u_hub_lb"], device)
    keep = ("band_blocks", "band_k0", "plane_u", "plane_su", "inflow_hub_lb", "dt", "n_spin", "n_avg", "wall_s",
            "mlups")
    np.savez_compressed(cache(results, f"empty_D{setup.cells_per_d}.npz"), **{k: np.asarray(out[k]) for k in keep})
    u = out["plane_u"] / out["inflow_hub_lb"]
    log.info("empty run D/%d: 30-min hub-plane mean U/U_hub %.3f-%.3f (sd %.3f)", setup.cells_per_d, u.min(),
             u.max(), u.std())
    return out


def load_empty(results: Path, n: int) -> dict | None:
    path = cache(results, f"empty_D{n}.npz")
    return load_run(path) if path.exists() else None


def step_sweep(setup: abl.Setup, results: Path, device: str, directions, force: bool = False,
               frame_directions=FRAME_DIRECTIONS) -> pd.DataFrame:
    state, info, cal = load_prepared(setup, results)
    geo = farm.geometry(setup)
    for wd in directions:
        path = run_path(results, setup.cells_per_d, float(wd))
        if path.exists() and not force:
            continue
        frames_s = FRAMES_S if float(wd) in frame_directions else 0.0
        out = farm.run_direction(setup, geo, cal, float(wd), state, info["u_hub_lb"], device, frames_s=frames_s)
        out["gpu_mb"] = gpu_memory_mb(device) or np.nan
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    return collect_runs(setup, results)


def load_run(path: Path) -> dict:
    with np.load(path) as z:
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def collect_runs(setup: abl.Setup, results: Path) -> pd.DataFrame:
    """Per-direction ratios (free-stream equivalent speed / inflow hub speed, per 10-minute block) and run costs."""
    n = setup.cells_per_d
    cal = _read_json(results / "disk_calibration.json")[str(n)]
    empty = load_empty(results, n)
    if empty is None:
        log.warning("no run without turbines at D/%d: ratios refer to the farm-wide inflow speed", n)
    ratios, runs = [], []
    for path in sorted(cache(results, f"les_D{n}", "x").parent.glob("wd_*.npz")):
        out = load_run(path)
        s = farm.equivalent_speed_ratio(out, cal, empty=empty, setup=setup)
        for b in range(s.shape[0]):
            for i in range(s.shape[1]):
                ratios.append({"wd": out["wd"], "block": b, "turbine": i + 1, "s": s[b, i]})
        runs.append({"wd": out["wd"], "cells": out["cells"], "steps": out["n_spin"] + out["n_avg"],
                     "wall_s": out["wall_s"], "mlups": out["mlups"], "population_mb": out["bytes"] / 2 ** 20,
                     "gpu_mb": out.get("gpu_mb", np.nan), "dt_s": out["dt"], "inflow_hub_lb": out["inflow_hub_lb"]})
    ratios, runs = pd.DataFrame(ratios), pd.DataFrame(runs)
    if len(runs):
        ratios.to_csv(results / f"les_ratios_D{n}.csv", index=False, float_format="%.5f")
        runs.to_csv(results / f"les_runs_D{n}.csv", index=False, float_format="%.5g")
    return ratios


def step_single(setup: abl.Setup, results: Path, device: str) -> pd.DataFrame:
    """One turbine in the same inflow: hub-height wake centreline and lateral profiles."""
    state, info, cal = load_prepared(setup, results)
    geo = farm.geometry(setup)
    out = farm.run_single(setup, geo, cal, state, info["u_hub_lb"], device)
    np.savez_compressed(cache(results, f"single_D{setup.cells_per_d}.npz"),
                        **{k: np.asarray(v) for k, v in out.items()})
    return single_profiles(setup, out, load_empty(results, setup.cells_per_d))


def single_profiles(setup: abl.Setup, out: dict, empty: dict) -> pd.DataFrame:
    """Centreline deficit 1 - U/U_0 on the plane nearest hub height, U_0 = the same plane without the turbine."""
    n = setup.cells_per_d
    u = out["plane_u"] / empty["plane_u"]
    xc, yc = float(out["centres"][0][0]), float(out["centres"][0][1])
    j0 = int(np.floor(yc))
    wy = yc - j0
    line = (1 - wy) * u[j0] + wy * u[j0 + 1]
    x_d = (np.arange(u.shape[1]) - xc) / n
    df = pd.DataFrame({"x_d": x_d, "deficit": 1 - line, "cells_per_d": n})
    return df[(df["x_d"] >= -3) & (df["x_d"] <= 15)]


# --- grid convergence -----------------------------------------------------------------------------------------

def step_grid(results: Path, device: str, directions=GRID_DIRECTIONS, resolutions=RESOLUTIONS) -> pd.DataFrame:
    """Same directions at several resolutions (each with its own precursor and calibration)."""
    rows = []
    for n in resolutions:
        setup = abl.Setup(cells_per_d=n)
        if not (results / f"precursor_D{n}.json").exists():
            step_prepare(setup, results, device)
        if load_empty(results, n) is None:
            step_empty(setup, results, device)
        step_sweep(setup, results, device, directions, frame_directions=())
        cal = _read_json(results / "disk_calibration.json")[str(n)]
        info = _read_json(results / f"precursor_D{n}.json")
        empty = load_empty(results, n)
        for wd in directions:
            out = load_run(run_path(results, n, float(wd)))
            s = farm.equivalent_speed_ratio(out, cal, empty=empty, setup=setup).mean(axis=0)
            row = {"cells_per_d": n, "wd": wd, "ti_hub": info["ti_hub"], "wall_s": out["wall_s"],
                   "mlups": out["mlups"], "cells": out["cells"], "gpu_mb": out.get("gpu_mb", np.nan)}
            row.update({f"s{i + 1}": v for i, v in enumerate(s)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results / "grid_convergence.csv", index=False, float_format="%.5g")
    return df


def les_efficiency(s: np.ndarray, wd: float, ws: float = 8.0) -> float:
    """Farm power over six times the mean power of the IEC free-stream turbines, at one wind speed."""
    x, y, _ = layout.kelmarsh()
    free = layout.free_stream(x, y, [wd])[0]
    p = turbine.power_kw(ws * np.asarray(s))
    return float(p.sum() / (len(p) * p[free].mean()))


# --- half-precision storage -----------------------------------------------------------------------------------

def step_fp16(results: Path, device: str, wd: float = 240.0) -> pd.DataFrame:
    """float16 populations: viscosity, throughput and one farm direction against float32."""
    tgv = pd.DataFrame([verify.taylor_green(n, device, storage=s) for s in ("fp32", "fp16") for n in (16, 32, 64, 128)])
    tgv.to_csv(results / "verify_taylor_green_storage.csv", index=False, float_format="%.6g")
    sizes = [(128, 96, 64), (256, 192, 96), (400, 300, 160), (480, 400, 192)]
    rows = verify.benchmark_sizes(device, sizes[:-1]) + verify.benchmark_sizes(device, sizes, storage="fp16")
    bench = pd.DataFrame(rows)
    bench.to_csv(results / "benchmark_storage.csv", index=False, float_format="%.4g")

    n = 8
    setup32, setup16 = abl.Setup(cells_per_d=n), abl.Setup(cells_per_d=n, storage="fp16")
    state, info, cal = load_prepared(setup32, results)
    # rounding makes the float16 inflow a different turbulence realisation, so it needs its own run without
    # turbines as the reference (which also compiles the float16 kernels before the timed direction run)
    geo16 = farm.geometry(setup16)
    empty16 = farm.run_empty(setup16, geo16, cal, state, info["u_hub_lb"], device)
    out16 = farm.run_direction(setup16, geo16, cal, wd, state, info["u_hub_lb"], device)
    out32 = load_run(run_path(results, n, wd))
    s16 = farm.equivalent_speed_ratio(out16, cal, empty=empty16, setup=setup32)
    s32 = farm.equivalent_speed_ratio(out32, cal, empty=load_empty(results, n), setup=setup32)
    df = pd.DataFrame({"turbine": np.arange(1, 7), "s_fp32": s32.mean(axis=0), "s_fp16": s16.mean(axis=0),
                       "block_sd_fp32": s32.std(axis=0, ddof=1), "block_sd_fp16": s16.std(axis=0, ddof=1),
                       "mlups_fp32": out32["mlups"],
                       "mlups_fp16": out16["mlups"], "wall_s_fp32": out32["wall_s"], "wall_s_fp16": out16["wall_s"]})
    df.to_csv(results / "fp16_farm_check.csv", index=False, float_format="%.5g")
    log.info("fp16: farm ratios %s vs fp32 %s", df["s_fp16"].round(3).tolist(), df["s_fp32"].round(3).tolist())
    return df
