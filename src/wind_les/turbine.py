"""Senvion MM92 as an actuator disk: power curve, thrust, momentum theory and the disk calibration run.

Power: the manufacturer's catalogue curve at 1.225 kg/m3 (Open Energy Platform turbine library, as distributed
with windpowerlib), the same curve as in wind-wake-shadow. Thrust: no MM92 Ct curve is public; the generic DTU
turbine of PyWake gives Ct = 0.8 in partial load up to 8 m/s, which is the operating range of the SCADA records
compared here, so the disks use a constant Ct = 0.8.

An actuator disk prescribes C_T' = C_T / (1 - a)^2 on the disk-averaged velocity u_d. Momentum theory links the
two: a = C_T' / (4 + C_T'), C_T = 4 a (1 - a). The filtered (smeared) disk of the LES deviates a little from
this; ``isolated_disk`` measures the deviation in uniform inflow and ``calibrate`` picks the C_T' that gives
C_T = 0.8 at a given resolution, plus the ratio r = u_d / U_inf used to turn disk velocities into the free-stream
equivalent speed for the power curve.

Usage:
    power_kw(8.0)                                    # 991 kW
    momentum(1.528)                                  # a = 0.276, ct = 0.800
    cal = calibrate(cells_per_d=8, device="cuda:0")  # {'ct_prime': ..., 'r': ..., 'ct': 0.80, ...}
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from wind_les.solver import LBM

DIAMETER_M = 92.0
RATED_KW = 2050.0
CUT_IN, CUT_OUT = 3.0, 25.0
CT_PARTIAL = 0.8
SIGMA_CELLS = 0.75
MM92_POWER_KW = pd.Series(
    [0.0, 22.0, 93.1, 207.2, 388.2, 642.7, 991.2, 1355.7, 1780.0, 2012.6, 2048.0] + [2055.0] * 13,
    index=np.arange(2.0, 26.0), name="power_kw").rename_axis("ws")
log = logging.getLogger(__name__)


def power_kw(ws):
    ws = np.asarray(ws, dtype=float)
    p = np.interp(ws, MM92_POWER_KW.index, MM92_POWER_KW.values)
    return np.where((ws >= CUT_IN) & (ws <= CUT_OUT), p, 0.0)


def momentum(ct_prime: float) -> dict:
    """Induction and thrust coefficient of an ideal disk with local thrust coefficient C_T'."""
    a = ct_prime / (4.0 + ct_prime)
    return {"a": a, "ct": 4.0 * a * (1.0 - a)}


def ct_prime_from_ct(ct: float) -> float:
    a = 0.5 * (1.0 - math.sqrt(1.0 - ct))
    return ct / (1.0 - a) ** 2


def isolated_disk(cells_per_d: int, ct_prime: float, device: str, sigma: float = SIGMA_CELLS,
                  u_inf: float = 0.05, up_d: float = 5.0, down_d: float = 9.0, side_d: float = 10.0,
                  flow_throughs: float = 2.0) -> dict:
    """One disk in uniform inflow (free-slip lateral walls far away): induction, thrust, wake centreline."""
    n = cells_per_d
    nx, ny, nz = int((up_d + down_d) * n), int(side_d * n), int(side_d * n)
    sim = LBM(nx, ny, nz, device=device, periodic_x=False, nu=1e-4, smagorinsky=0.16,
              sponge_cells=int(1.5 * n), sponge_nu=0.05)
    u = np.zeros((3, nz, ny, nx), np.float32)
    u[0] = u_inf
    sim.init_equilibrium(u)
    sim.set_inlet_profile(np.full(nz, u_inf))
    steps = int(flow_throughs * nx / u_inf)
    centre = [up_d * n, ny / 2 - 0.5, nz / 2 - 0.5]
    info = sim.set_turbines([centre], n / 2, ct_prime, sigma=sigma, u_init=u_inf, hist_len=steps)
    sim.start_plane_stats(nz // 2)
    sim.step(steps // 2)
    sim.start_plane_stats(nz // 2)
    sim.step(steps - steps // 2)
    ud = sim.disk_velocity_history()[steps // 2:, 0]
    m = float(info["M"][0])
    thrust = 0.5 * ct_prime * m ** 2 * float((ud ** 2).mean()) * math.pi * (n / 2) ** 2
    plane = sim.plane_stats()
    jc = ny // 2
    line = plane["u"][jc - 1:jc + 1].mean(axis=0) / u_inf      # the disk centre sits between two nodes
    x_d = (np.arange(nx) - up_d * n) / n
    return {
        "cells_per_d": n, "ct_prime": ct_prime, "M": m, "r": float(ud.mean() / u_inf),
        "a": 1.0 - m * float(ud.mean()) / u_inf, "ct": thrust / (0.5 * u_inf ** 2 * math.pi * (n / 2) ** 2),
        "theory": momentum(ct_prime), "x_d": x_d, "centreline": line, "cells": sim.cells, "steps": steps,
    }


def calibrate(cells_per_d: int, device: str, ct_target: float = CT_PARTIAL, sigma: float = SIGMA_CELLS,
              iters: int = 3) -> dict:
    """C_T' that gives the target C_T for an isolated disk at this resolution (secant iterations)."""
    x0 = ct_prime_from_ct(ct_target)
    r0 = isolated_disk(cells_per_d, x0, device, sigma)
    x1 = x0 * ct_target / r0["ct"]
    r1 = isolated_disk(cells_per_d, x1, device, sigma)
    for _ in range(iters - 2):
        if abs(r1["ct"] - r0["ct"]) < 1e-9:
            break
        x2 = x1 + (ct_target - r1["ct"]) * (x1 - x0) / (r1["ct"] - r0["ct"])
        x0, r0 = x1, r1
        x1, r1 = x2, isolated_disk(cells_per_d, x2, device, sigma)
    log.info("disk calibration D/%d: C_T' %.4f -> C_T %.4f, u_d/U %.4f", cells_per_d, x1, r1["ct"], r1["r"])
    return {"cells_per_d": cells_per_d, "ct_prime": x1, "ct": r1["ct"], "r": r1["r"], "a": r1["a"],
            "M": r1["M"], "sigma": sigma, "ct_prime_momentum": ct_prime_from_ct(ct_target)}
