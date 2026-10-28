"""Neutral atmospheric boundary layer in lattice units: scaling, log law, and a periodic precursor.

The surface layer is a pressure-driven open channel: constant body force F in x, a rough wall (log-law wall
stress in the first cell, roughness z0) and a free-slip lid at height H. In equilibrium the wall stress balances
the driving force, u*^2 = F H, and the mean profile follows the log law near the ground:

    U(z) = u* / kappa ln(z / z0)

The LES is effectively Reynolds-number independent (eddy viscosity >> molecular), so the hub-height speed in
lattice units only has to be small enough for low Mach number. Physical time follows from the measured hub
speed: dt = dx u_hub,lattice / u_hub,physical.

The precursor is periodic in x with a lateral shift at the wrap, so streaks do not lock in place (Munters,
Meneveau & Meyers 2016). Its outflow plane feeds the inflow of the farm domain at every step.

Usage:
    setup = Setup(cells_per_d=8)
    pre = make_precursor(setup, ny=160, device="cuda:0")
    hist = spin_up(pre, setup, steps=100_000)
    stats = profile_stats(pre, setup, steps=20_000, every=50)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from wind_les.solver import KAPPA, LBM

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Setup:
    cells_per_d: int = 8
    diameter_m: float = 92.0
    height_m: float = 460.0          # free-slip lid, about 6 hub heights
    hub_m: float = 78.5
    z0_m: float = 0.20               # farmland with hedgerows and woods; LES hub TI close to the SCADA 12.7 %
    u_hub_lb: float = 0.05           # target hub speed in lattice units (Mach ~ 0.09)
    u_hub_ms: float = 8.0            # physical hub speed the lattice speed is mapped to
    smagorinsky: float = 0.16
    nu_lb: float = 1e-5              # molecular viscosity, negligible next to the eddy viscosity
    precursor_d: float = 12.0        # precursor length in rotor diameters
    storage: str = "fp32"            # population storage: fp32 or fp16

    @property
    def dx(self) -> float:
        return self.diameter_m / self.cells_per_d

    @property
    def nz(self) -> int:
        return int(round(self.height_m / self.dx))

    @property
    def z0_lb(self) -> float:
        return self.z0_m / self.dx

    @property
    def ustar_lb(self) -> float:
        return self.u_hub_lb * KAPPA / math.log(self.hub_m / self.z0_m)

    @property
    def force_lb(self) -> float:
        return self.ustar_lb ** 2 / self.nz

    def z_m(self) -> np.ndarray:
        """Cell-centre heights; the wall sits half a cell below the first node."""
        return (np.arange(self.nz) + 0.5) * self.dx

    def k_at(self, z_m: float) -> float:
        """Fractional lattice index of a height."""
        return z_m / self.dx - 0.5

    def dt(self, u_hub_lb: float | None = None) -> float:
        """Physical seconds per step for the measured (or target) lattice hub speed."""
        return self.dx * (u_hub_lb or self.u_hub_lb) / self.u_hub_ms

    def log_law(self, z_m) -> np.ndarray:
        """Log-law speed in lattice units at the target friction velocity."""
        return self.ustar_lb / KAPPA * np.log(np.asarray(z_m) / self.z0_m)


def precursor_domain(setup: Setup, ny: int, device: str) -> LBM:
    """The periodic channel itself, without an initial state."""
    nx = int(round(setup.precursor_d * setup.cells_per_d))
    return LBM(nx, ny, setup.nz, device=device, periodic_x=True, shift_y=ny // 3, bottom="wall_model",
               top="free_slip", nu=setup.nu_lb, smagorinsky=setup.smagorinsky, z0=setup.z0_lb,
               force=(setup.force_lb, 0.0, 0.0), storage=setup.storage)


def make_precursor(setup: Setup, ny: int, device: str, seed: int = 1) -> LBM:
    """Periodic channel with a log-law start and random plus streak perturbations to trigger turbulence."""
    sim = precursor_domain(setup, ny, device)
    nz, ny, nx = sim.shape
    rng = np.random.default_rng(seed)
    u_log = setup.log_law(setup.z_m()).astype(np.float32)
    u = np.zeros((3, nz, ny, nx), np.float32)
    u[0] = u_log[:, None, None]
    zz = (np.arange(nz) + 0.5)[:, None, None] / nz
    yy = np.arange(ny)[None, :, None] / ny
    xx = np.arange(nx)[None, None, :] / nx
    envelope = np.exp(-((zz - 0.15) / 0.15) ** 2)
    streaks = 0.1 * setup.u_hub_lb * np.sin(2 * np.pi * 4 * yy) * np.cos(2 * np.pi * xx) * envelope
    u[0] += streaks.astype(np.float32)
    u += (0.05 * setup.u_hub_lb * envelope * rng.standard_normal((3, nz, ny, nx))).astype(np.float32)
    sim.init_equilibrium(u)
    return sim


def hub_speed(sim: LBM, setup: Setup, z_m: float | None = None) -> float:
    """Plane-averaged streamwise speed at hub height (linear between cell centres)."""
    _, u = sim.macroscopic()
    prof = u[0].mean(axis=(1, 2))
    return float(np.interp(setup.k_at(z_m or setup.hub_m), np.arange(setup.nz), prof))


def spin_up(sim: LBM, setup: Setup, steps: int, every: int = 2000) -> np.ndarray:
    """Run the precursor, logging the plane-averaged hub speed; returns (step, speed) rows."""
    rows = []
    done = 0
    while done < steps:
        n = min(every, steps - done)
        sim.step(n)
        done += n
        u = hub_speed(sim, setup)
        if not np.isfinite(u):
            raise FloatingPointError(f"precursor diverged after {done} steps")
        rows.append((done, u))
        if len(rows) % 10 == 0:
            log.info("precursor step %d: hub speed %.4f (lattice)", done, u)
    return np.array(rows)


def profile_stats(sim: LBM, setup: Setup, steps: int, every: int = 50) -> dict:
    """Plane- and time-averaged profiles of mean speed, variances and shear stress, plus hub-height TI."""
    n = 0
    s = {k: np.zeros(setup.nz) for k in ("u", "v", "w", "uu", "vv", "ww", "uw")}
    for _ in range(steps // every):
        sim.step(every)
        _, u = sim.macroscopic()
        for name, val in (("u", u[0]), ("v", u[1]), ("w", u[2]), ("uu", u[0] ** 2), ("vv", u[1] ** 2),
                          ("ww", u[2] ** 2), ("uw", u[0] * u[2])):
            s[name] += val.mean(axis=(1, 2))
        n += 1
    m = {k: v / n for k, v in s.items()}
    out = {"z_m": setup.z_m(), "u": m["u"], "su": np.sqrt(np.maximum(m["uu"] - m["u"] ** 2, 0)),
           "sv": np.sqrt(np.maximum(m["vv"] - m["v"] ** 2, 0)), "sw": np.sqrt(np.maximum(m["ww"] - m["w"] ** 2, 0)),
           "uw": m["uw"] - m["u"] * m["w"]}
    k_hub = setup.k_at(setup.hub_m)
    out["u_hub"] = float(np.interp(k_hub, np.arange(setup.nz), out["u"]))
    out["ti_hub"] = float(np.interp(k_hub, np.arange(setup.nz), out["su"])) / out["u_hub"]
    out["ustar_target"] = setup.ustar_lb
    return out
