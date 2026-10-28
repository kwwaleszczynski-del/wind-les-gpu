"""Kelmarsh in the LES: domain geometry, one run per wind direction, and what each run keeps.

The farm domain is the same for every direction: the turbines are rotated into the wind-aligned frame around
the farm centre, which sits far enough from the inflow plane and the sides for any direction. The precursor
(abl.py) runs next to it and hands over its outflow plane at every step. Every direction starts from the same
spun-up precursor state, so all directions see the same inflow turbulence (common random numbers): the
direction-to-direction differences carry little sampling noise.

A run is a spin-up (the wakes cross the domain twice) followed by the averaging window. Kept: the filtered disk
velocity of every turbine at every step, hub-plane mean and variance maps, optional hub-plane snapshots, the
inflow hub speed from the precursor, and timings.

Usage:
    geo = geometry(setup)
    out = run_direction(setup, geo, cal, 240.0, pre_state, u_ref_lb=0.049, device="cuda:0")
    single = run_single(setup, geo, cal, pre_state, u_ref_lb=0.049, device="cuda:0")
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from wind_les import abl, layout
from wind_les.solver import LBM

SPINUP_S = 600.0
AVERAGE_S = 1800.0
BLOCK_S = 600.0          # SCADA-like 10-minute blocks inside the averaging window
SPONGE_D = 2.0
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Geometry:
    nx: int
    ny: int
    x0: float            # farm centre, lattice coordinates
    y0: float
    radius_m: float


def geometry(setup: abl.Setup, up_d: float = 5.0, side_d: float = 5.0, down_d: float = 10.0) -> Geometry:
    """Fixed domain that holds the farm for any wind direction with the given margins (in rotor diameters)."""
    x, y, _ = layout.kelmarsh()
    d = setup.diameter_m
    r = float(np.max(np.hypot(x, y))) + d / 2
    nx = int(math.ceil((up_d * d + 2 * r + down_d * d) / setup.dx))
    ny = int(math.ceil((2 * side_d * d + 2 * r) / setup.dx))
    return Geometry(nx, ny, (up_d * d + r) / setup.dx, (ny - 1) / 2, r)


def turbine_centres(setup: abl.Setup, geo: Geometry, wd: float, x=None, y=None, hub=None) -> np.ndarray:
    """Lattice coordinates (n, 3) of the rotor centres for wind from ``wd``."""
    if x is None:
        x, y, hub = layout.kelmarsh()
    xr, yr = layout.to_wind_frame(x, y, wd)
    return np.stack([geo.x0 + xr / setup.dx, geo.y0 + yr / setup.dx,
                     np.array([setup.k_at(h) for h in hub])], axis=1)


def farm_domain(setup: abl.Setup, geo: Geometry, device: str) -> LBM:
    return LBM(geo.nx, geo.ny, setup.nz, device=device, periodic_x=False, bottom="wall_model", top="free_slip",
               nu=setup.nu_lb, smagorinsky=setup.smagorinsky, z0=setup.z0_lb, force=(setup.force_lb, 0.0, 0.0),
               sponge_cells=int(SPONGE_D * setup.cells_per_d), sponge_nu=0.05, storage=setup.storage)


def _inflow_hub_speed(pre: LBM, setup: abl.Setup) -> float:
    k = setup.k_at(setup.hub_m)
    k0 = int(math.floor(k))
    lo, hi = pre.plane_velocity(k0)[0].mean(), pre.plane_velocity(k0 + 1)[0].mean()
    return float(lo + (k - k0) * (hi - lo))


def run(setup: abl.Setup, geo: Geometry, cal: dict, centres: np.ndarray, pre_state: np.ndarray, u_ref_lb: float,
        device: str, spinup_s: float = SPINUP_S, average_s: float = AVERAGE_S, frames_s: float = 0.0,
        frame_every_s: float = 4.0, band: tuple[int, int] | None = None) -> dict:
    """Precursor + farm domain with actuator disks at ``centres``; returns the kept statistics.

    ``band`` = (k0, k1) also keeps the mean streamwise velocity on those planes for every 10-minute block.
    """
    pre = abl.precursor_domain(setup, geo.ny, device)
    pre.load(pre_state)
    main = farm_domain(setup, geo, device)
    main.tile_state_from(pre)
    dt = setup.dt(u_ref_lb)
    n_spin, n_avg = int(round(spinup_s / dt)), int(round(average_s / dt))
    main.set_turbines(centres, setup.cells_per_d / 2, cal["ct_prime"], sigma=cal["sigma"],
                      u_init=u_ref_lb * cal["r"], hist_len=n_spin + n_avg)
    k_hub = int(round(setup.k_at(setup.hub_m)))
    frame_every = max(1, int(round(frame_every_s / dt)))
    n_frames = int(frames_s / frame_every_s)
    frames, inflow, band_sums = [], [], []
    n_block = max(1, int(round(BLOCK_S / dt)))
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for s in range(n_spin + n_avg):
        if s == n_spin:
            main.start_plane_stats(k_hub)
            if band is not None:
                main.start_band_stats(*band)
        if band is not None and s > n_spin and (s - n_spin) % n_block == 0:
            band_sums.append(main.band_sums())
        pre.step(1)
        main.step(1, inlet=pre)
        if s >= n_spin and (s - n_spin) % 200 == 0:
            inflow.append(_inflow_hub_speed(pre, setup))
        if s >= n_spin and len(frames) < n_frames and (s - n_spin) % frame_every == 0:
            frames.append(main.plane_velocity(k_hub)[0].astype(np.float16))
    if band is not None:
        band_sums.append(main.band_sums())
    wp.synchronize_device(device)
    wall = time.perf_counter() - t0
    ud = main.disk_velocity_history()
    if not np.all(np.isfinite(ud)):
        raise FloatingPointError("farm run diverged")
    stats = main.plane_stats()
    cells = pre.cells + main.cells
    extra = {}
    if band is not None:
        sums = [np.zeros_like(band_sums[0][0])] + [b[0] for b in band_sums]
        counts = [0] + [b[1] for b in band_sums]
        extra = {"band_k0": band[0], "band_blocks": np.array([(sums[b + 1] - sums[b]) / (counts[b + 1] - counts[b])
                                                             for b in range(len(band_sums))], np.float32)}
    return {**extra,
        "centres": centres, "ud": ud.astype(np.float32), "n_spin": n_spin, "n_avg": n_avg, "dt": dt,
        "u_ref_lb": u_ref_lb, "inflow_hub_lb": float(np.mean(inflow)), "k_hub": k_hub,
        "z_plane_m": (k_hub + 0.5) * setup.dx,
        "plane_u": stats["u"].astype(np.float32), "plane_v": stats["v"].astype(np.float32),
        "plane_su": np.sqrt(np.maximum(stats["uu"], 0)).astype(np.float32),
        "frames": np.array(frames) if frames else np.zeros((0, geo.ny, geo.nx), np.float16),
        "frame_dt_s": frame_every * dt, "wall_s": wall, "cells": cells,
        "mlups": cells * (n_spin + n_avg) / wall / 1e6, "bytes": pre.bytes_allocated() + main.bytes_allocated(),
    }


def run_direction(setup: abl.Setup, geo: Geometry, cal: dict, wd: float, pre_state: np.ndarray, u_ref_lb: float,
                  device: str, **kw) -> dict:
    out = run(setup, geo, cal, turbine_centres(setup, geo, wd), pre_state, u_ref_lb, device, **kw)
    out["wd"] = wd
    log.info("wd %5.1f: %.0f s wall, %.0f MLUPS", wd, out["wall_s"], out["mlups"])
    return out


def run_single(setup: abl.Setup, geo: Geometry, cal: dict, pre_state: np.ndarray, u_ref_lb: float, device: str,
               x_d: float = 7.0, **kw) -> dict:
    """One turbine at 78.5 m hub height, ``x_d`` rotor diameters behind the inflow plane, mid-domain."""
    centre = np.array([[x_d * setup.cells_per_d, geo.y0, setup.k_at(setup.hub_m)]])
    out = run(setup, geo, cal, centre, pre_state, u_ref_lb, device, **kw)
    out["wd"] = np.nan
    return out


def run_empty(setup: abl.Setup, geo: Geometry, cal: dict, pre_state: np.ndarray, u_ref_lb: float, device: str,
              **kw) -> dict:
    """The same run without turbines: the undisturbed mean flow over the rotor heights, per 10-minute block.

    Every direction starts from the same precursor state and the solver is deterministic, so this is the
    flow each turbine would have seen without the farm, including the slowly evolving streaks of the inflow.
    """
    reach = setup.cells_per_d / 2 + 4 * cal["sigma"]
    k1 = int(math.ceil(max(setup.k_at(h) for h in layout.STATIC["hub_height_m"]) + reach)) + 2
    out = run(setup, geo, cal, np.zeros((0, 3)), pre_state, u_ref_lb, device, band=(0, min(k1, setup.nz)), **kw)
    out["wd"] = np.nan
    return out


def _erf(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorised."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    y = 1.0 - poly * np.exp(-x * x)
    return s * y


def disk_weights(centre, radius: float, sigma: float, shape) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    """The actuator disk's smoothing weights (as in the solver) on its bounding box of a (nz, ny, nx) grid."""
    xc, yc, zc = centre
    nz, ny, nx = shape
    reach = radius + 4 * sigma
    sl = (slice(max(0, int(np.floor(zc - reach))), min(nz, int(np.ceil(zc + reach)) + 1)),
          slice(max(0, int(np.floor(yc - reach))), min(ny, int(np.ceil(yc + reach)) + 1)),
          slice(max(0, int(np.floor(xc - 4 * sigma))), min(nx, int(np.ceil(xc + 4 * sigma)) + 1)))
    k, j, i = np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing="ij")
    dx = i - xc
    r = np.hypot(j - yc, k - zc)
    g = np.exp(-0.5 * dx ** 2 / sigma ** 2) / (np.sqrt(2 * np.pi) * sigma)
    w = g * 0.5 * (1 - _erf((r - radius) / (np.sqrt(2) * sigma)))
    w[(np.abs(dx) > 4 * sigma) | (r > reach)] = 0.0
    return sl, w


def local_reference(empty: dict, centres, setup: abl.Setup, sigma: float) -> np.ndarray:
    """Disk-averaged undisturbed velocity at each turbine position, per block: shape (blocks, n)."""
    blocks = empty["band_blocks"]
    k0 = int(empty["band_k0"])
    out = np.zeros((blocks.shape[0], len(centres)))
    for n, c in enumerate(centres):
        (sz, sy, sx), w = disk_weights((c[0], c[1], c[2] - k0), setup.cells_per_d / 2, sigma, blocks.shape[1:])
        out[:, n] = (blocks[:, sz, sy, sx] * w).sum(axis=(1, 2, 3)) / w.sum()
    return out


def height_factor(empty: dict, setup: abl.Setup, hub_m, sigma: float) -> np.ndarray:
    """Undisturbed disk-averaged speed at each hub height relative to the 78.5 m one (farm-wide mean profile)."""
    prof = empty["band_blocks"].mean(axis=(0, 2, 3))
    k0 = int(empty["band_k0"])
    flat = {"band_blocks": np.broadcast_to(prof[None, :, None, None], (1, len(prof), 9 * setup.cells_per_d,
                                                                       9 * setup.cells_per_d)), "band_k0": k0}
    mid = 4.5 * setup.cells_per_d
    cs = [[mid, mid, setup.k_at(h)] for h in [setup.hub_m, *hub_m]]
    ref = local_reference(flat, cs, setup, sigma)[0]
    return ref[1:] / ref[0]


def block_means(ud: np.ndarray, n_spin: int, dt: float, block_s: float = BLOCK_S) -> np.ndarray:
    """Mean disk velocity per turbine in consecutive blocks of the averaging window, shape (blocks, n).

    The last block takes the remainder, so a window a few time steps off a whole number of blocks (rounding of
    the time step) keeps its last block, as the band snapshots of the run without turbines do.
    """
    avg = ud[n_spin:]
    n = max(1, int(round(block_s / dt)))
    k = max(1, int(round(len(avg) / n)))
    edges = [b * n for b in range(k)] + [len(avg)]
    return np.stack([avg[edges[b]:edges[b + 1]].mean(axis=0) for b in range(k)])


def equivalent_speed_ratio(out: dict, cal: dict, block_s: float = BLOCK_S, empty: dict | None = None,
                           setup: abl.Setup | None = None) -> np.ndarray:
    """Free-stream equivalent speed of each turbine relative to the inflow hub speed, per 10-minute block.

    With the run without turbines (``empty``), each turbine is referred to its own undisturbed inflow (times
    the shear factor of its hub height); without it, to the farm-wide inflow hub speed.
    """
    ud = block_means(out["ud"], int(out["n_spin"]), float(out["dt"]), block_s)
    if empty is None:
        return ud / cal["r"] / out["inflow_hub_lb"]
    centres = np.asarray(out["centres"])
    hub = (centres[:, 2] + 0.5) * setup.dx
    ref = local_reference(empty, centres, setup, cal["sigma"])[: ud.shape[0]]
    return ud / cal["r"] / ref * height_factor(empty, setup, hub, cal["sigma"])
