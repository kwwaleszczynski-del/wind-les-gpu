"""Command line: verification, precursor, direction sweep, grid study, comparison, figures, web page.

GPU steps default to cuda:0 and fall back to the CPU (slow; only sensible for the tests' tiny grids).
Kelmarsh data are read from ./data (override with --data-dir or KELMARSH_DATA_DIR, e.g. the data folder of
wind-power-curve-kelmarsh); only the compare step needs them, plus the [compare] extra.

Usage:
    python -m wind_les verify                          # analytical checks, disk vs momentum theory, MLUPS
    python -m wind_les prepare --cells-per-d 8         # disk calibration + spun-up precursor
    python -m wind_les empty --cells-per-d 8           # the same inflow without turbines (reference)
    python -m wind_les sweep --cells-per-d 8 --step 5  # one LES per wind direction (resumable)
    python -m wind_les prepare --realisation 1         # + empty, sweep: a second inflow realisation
    python -m wind_les single --cells-per-d 8          # one turbine: wake deficit vs distance
    python -m wind_les grid                            # 240 / 300 deg at D/8, D/12, D/16
    python -m wind_les fp16                            # float16 populations: accuracy and speed
    python -m wind_les compare                         # sector losses: SCADA, PyWake, LES
    python -m wind_les figures && python -m wind_les web
    python -m wind_les all                             # everything above, about 3 h on a 6 GB laptop GPU
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import warp as wp

from wind_les import abl, study

log = logging.getLogger("wind_les")


def _device(name: str) -> str:
    if name.startswith("cuda") and not wp.is_cuda_available():
        log.warning("no CUDA device, falling back to the CPU")
        return "cpu"
    return name


def cmd_verify(a):
    study.step_verify(a.results, a.device)


def _setup(a) -> abl.Setup:
    return abl.Setup(cells_per_d=a.cells_per_d, realisation=a.realisation)


def cmd_prepare(a):
    study.step_prepare(_setup(a), a.results, a.device)


def cmd_empty(a):
    study.step_empty(_setup(a), a.results, a.device)


def cmd_sweep(a):
    dirs = a.directions if a.directions else np.arange(0.0, 360.0, a.step)
    study.step_sweep(_setup(a), a.results, a.device, dirs, force=a.force)


def cmd_single(a):
    setup = abl.Setup(cells_per_d=a.cells_per_d)
    df = study.step_single(setup, a.results, a.device)
    df.to_csv(a.results / f"single_wake_D{a.cells_per_d}.csv", index=False, float_format="%.5f")


def cmd_grid(a):
    study.step_grid(a.results, a.device, a.directions or study.GRID_DIRECTIONS)


def cmd_fp16(a):
    study.step_fp16(a.results, a.device)


def cmd_compare(a):
    from wind_les import report
    report.compare_all(a.results, a.data_dir)


def cmd_figures(a):
    from wind_les import plots
    plots.all_figures(a.results, a.docs / "img")


def cmd_web(a):
    from wind_les import web
    web.build(a.results, a.docs)


def cmd_all(a):
    cmd_verify(a)
    for n in (8,):
        a.cells_per_d = n
        cmd_prepare(a)
        cmd_empty(a)
        cmd_sweep(a)
        cmd_single(a)
    cmd_grid(a)
    cmd_fp16(a)
    cmd_compare(a)
    cmd_figures(a)
    cmd_web(a)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wind_les", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("verify", cmd_verify), ("prepare", cmd_prepare), ("empty", cmd_empty), ("sweep", cmd_sweep),
                     ("single", cmd_single), ("grid", cmd_grid), ("fp16", cmd_fp16), ("compare", cmd_compare),
                     ("figures", cmd_figures), ("web", cmd_web), ("all", cmd_all)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--results", type=Path, default=Path("results"))
        p.add_argument("--docs", type=Path, default=Path("docs"))
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("KELMARSH_DATA_DIR", "data")))
        p.add_argument("--cells-per-d", type=int, default=8, help="grid cells per rotor diameter")
        p.add_argument("--step", type=float, default=study.DIRECTION_STEP, help="direction step of the sweep [deg]")
        p.add_argument("--directions", type=float, nargs="*", help="explicit wind directions [deg]")
        p.add_argument("--force", action="store_true", help="re-run directions that are already cached")
        p.add_argument("--realisation", type=int, default=0,
                       help="inflow turbulence realisation (precursor seed) for prepare / empty / sweep")
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.device = _device(a.device)
    a.func(a)
