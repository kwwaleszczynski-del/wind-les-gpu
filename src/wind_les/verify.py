"""Checks of the method against analytical solutions, before any farm result is trusted.

    taylor_green   decay of a 2D Taylor-Green vortex: measured / set viscosity (second-order convergence)
    poiseuille     body-force driven channel between no-slip walls: profile error (forcing + bounce-back)
    benchmark      million lattice updates per second and memory for a range of grid sizes

The actuator disk against momentum theory is ``turbine.isolated_disk``; the inflow log law is
``abl.profile_stats``.

Usage:
    taylor_green(32, device="cpu")          # {'n': 32, 'nu_ratio': 1.025, ...}
    poiseuille(16, nu=0.1, device="cpu")    # {'max_rel_err': 0.011, ...}
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from wind_les.solver import LBM, benchmark


def taylor_green(n: int, device: str, nu: float = 0.02, u0: float = 0.02, decay: float = 0.5,
                 storage: str = "fp32") -> dict:
    """Run until the kinetic energy has dropped by exp(-decay); compare the decay rate with 4 nu k^2."""
    k = 2 * np.pi / n
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)
    u = np.zeros((3, 1, n, n))
    u[0, 0] = u0 * np.sin(k * xx) * np.cos(k * yy)
    u[1, 0] = -u0 * np.cos(k * xx) * np.sin(k * yy)
    rho = 1 + 3 * (-u0 ** 2 / 4 * (np.cos(2 * k * xx) + np.cos(2 * k * yy)))[None]
    sim = LBM(n, n, 1, device=device, nu=nu, storage=storage)
    sim.init_equilibrium(u, rho)
    steps = int(round(decay / (4 * nu * k * k)))
    sim.step(steps)
    _, um = sim.macroscopic()
    e0, e1 = (u[:2] ** 2).sum(), (um[:2] ** 2).sum()
    nu_m = -np.log(e1 / e0) / (4 * k * k * steps)
    return {"n": n, "nu": nu, "storage": storage, "steps": steps, "nu_measured": float(nu_m),
            "nu_ratio": float(nu_m / nu)}


def poiseuille(nz: int, nu: float, device: str, force: float = 1e-6) -> dict:
    """Steady channel flow; walls half a cell outside the first and last node."""
    sim = LBM(4, 4, nz, device=device, nu=nu, bottom="no_slip", top="no_slip", force=(force, 0.0, 0.0))
    sim.init_equilibrium(np.zeros((3, nz, 4, 4)))
    steps = int(8 * nz ** 2 / nu)
    sim.step(steps)
    rho, u = sim.macroscopic()
    z = np.arange(nz) + 0.5
    exact = force / (2 * nu) * z * (nz - z)
    um = u[0, :, 0, 0] + 0.5 * force / rho[:, 0, 0]
    return {"nz": nz, "nu": nu, "steps": steps, "max_rel_err": float(np.abs(um - exact).max() / exact.max()),
            "profile": um, "exact": exact}


def benchmark_sizes(device: str, sizes, steps: int = 200, storage: str = "fp32") -> list[dict]:
    """MLUPS of a turbulent-channel-like configuration (wall model, Smagorinsky) for (nx, ny, nz) sizes."""
    rows = []
    for nx, ny, nz in sizes:
        sim = LBM(nx, ny, nz, device=device, periodic_x=True, bottom="wall_model", nu=1e-5, smagorinsky=0.16,
                  z0=0.01, force=(1e-7, 0.0, 0.0), storage=storage)
        u = np.zeros((3, nz, ny, nx), np.float32)
        u[0] = 0.05
        sim.init_equilibrium(u)
        t0 = time.perf_counter()
        mlups = benchmark(sim, steps)
        rows.append({"storage": storage, "nx": nx, "ny": ny, "nz": nz, "cells": sim.cells, "mlups": mlups,
                     "population_mb": sim.bytes_allocated() / 2 ** 20, "seconds": time.perf_counter() - t0})
        del sim
        wp.synchronize_device(device)
    return rows
