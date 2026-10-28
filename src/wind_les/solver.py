"""Lattice Boltzmann LES on the GPU (or CPU) with NVIDIA Warp: D3Q19, regularized BGK, Smagorinsky, actuator disks.

One fused kernel per time step pulls the 19 populations from the neighbours (streaming), computes density and
velocity, adds body forces with Guo's scheme and relaxes towards equilibrium:

    f*_q = f_eq,q + (1 - 1/tau) f1_q + (1 - 1/(2 tau)) S_q

where f1 is the non-equilibrium part projected on second-order Hermite polynomials (regularization, Latt &
Chopard 2006) and S the Guo forcing term. The relaxation time carries the Smagorinsky eddy viscosity computed
locally from the non-equilibrium momentum flux Pi (Hou et al. 1996):

    tau = (tau0 + sqrt(tau0^2 + 18 sqrt(2) (Cs Delta)^2 |Pi| / rho)) / 2

Boundaries
    x   periodic (optionally with a lateral shift at the wrap, Munters et al. 2016, for precursors) or open:
        the first and last planes are ghost layers filled after each step from an inflow source and by
        zero-gradient copy at the outflow
    y   periodic
    z   free slip (specular reflection), no slip (half-way bounce-back), or free slip plus a log-law wall
        stress applied as a force in the first cell (Moeng 1984 type wall model, local velocity)

Actuator disks (thrust only, rotor normal to x) spread their force with a Gaussian of width sigma along x and
an error-function edge in the rotor plane. The thrust uses the disk-averaged velocity of the previous step:

    F = -1/2 rho C_T' M^2 u_d^2 A,   M = (1 + C_T' Delta / (4 sqrt(3 pi) R))^-1

with the filtered-disk correction M of Shapiro, Gayme & Meneveau (2019) and Delta = sqrt(12) sigma.

Populations are stored as f - w (see lattice.py) in two arrays (A-B pattern), as float32 or float16 (half the
memory and memory traffic; all arithmetic stays float32). Lattice units throughout: dx = dt = 1, rho ~ 1.

Usage:
    sim = LBM(64, 32, 32, device="cuda:0", periodic_x=True, nu=0.01)
    sim.init_equilibrium(u0)                # u0 shape (3, nz, ny, nx)
    sim.step(100)
    rho, u = sim.macroscopic()
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import warp as wp

from wind_les import lattice

wp.config.quiet = True
wp.set_module_options({"max_unroll": 32, "enable_backward": False, "fast_math": True})

FREE_SLIP, NO_SLIP, WALL_MODEL = 0, 1, 2
STORAGE = {"fp32": wp.float32, "fp16": wp.float16}
BOUNDARY = {"free_slip": FREE_SLIP, "no_slip": NO_SLIP, "wall_model": WALL_MODEL}
KAPPA = 0.4

vec19 = wp.types.vector(length=19, dtype=wp.float32)
vec19i = wp.types.vector(length=19, dtype=wp.int32)
CX = wp.constant(vec19i(*[int(c) for c in lattice.C[:, 0]]))
CY = wp.constant(vec19i(*[int(c) for c in lattice.C[:, 1]]))
CZ = wp.constant(vec19i(*[int(c) for c in lattice.C[:, 2]]))
OPP = wp.constant(vec19i(*[int(c) for c in lattice.OPPOSITE]))
MIRZ = wp.constant(vec19i(*[int(c) for c in lattice.MIRROR_Z]))
WQ = wp.constant(vec19(*[float(w) for w in lattice.W]))
WALL = wp.constant(WALL_MODEL)
NOSLIP = wp.constant(NO_SLIP)
KAP = wp.constant(KAPPA)


@wp.struct
class Params:
    nx: int
    ny: int
    nz: int
    periodic_x: int
    shift_y: int
    bottom: int
    top: int
    tau0: float
    cs2: float
    z0: float
    force: wp.vec3
    sponge_start: int
    sponge_nu: float
    n_turb: int
    sigma: float


@wp.func
def disk_weight(x: float, y: float, z: float, c: wp.vec3, radius: float, sigma: float) -> float:
    dx = x - c[0]
    if wp.abs(dx) > 4.0 * sigma:
        return 0.0
    r = wp.sqrt((y - c[1]) * (y - c[1]) + (z - c[2]) * (z - c[2]))
    if r > radius + 4.0 * sigma:
        return 0.0
    g = wp.exp(-0.5 * dx * dx / (sigma * sigma)) / (2.5066282746 * sigma)
    return g * 0.5 * (1.0 - wp.erf((r - radius) / (1.4142135624 * sigma)))


@wp.kernel
def stream_collide(f_in: wp.array4d(dtype=Any), f_out: wp.array4d(dtype=Any), p: Params,
                   tpos: wp.array(dtype=wp.vec3), trad: wp.array(dtype=float), tcoef: wp.array(dtype=float),
                   ud: wp.array(dtype=float), acc: wp.array(dtype=float)):
    k, j, i = wp.tid()
    if p.periodic_x == 0 and (i == 0 or i == p.nx - 1):
        return

    # streaming (pull) with boundary handling
    f = vec19()
    for q in range(19):
        xs = i - CX[q]
        ys = j - CY[q]
        zs = k - CZ[q]
        qq = q
        if xs < 0:
            xs = xs + p.nx
            ys = ys - p.shift_y
        elif xs >= p.nx:
            xs = xs - p.nx
            ys = ys + p.shift_y
        if ys < 0:
            ys = ys + p.ny
        elif ys >= p.ny:
            ys = ys - p.ny
        if zs < 0 or zs >= p.nz:
            bc = p.bottom
            if zs >= p.nz:
                bc = p.top
            if bc == NOSLIP:
                xs = i
                ys = j
                qq = OPP[q]
            else:
                qq = MIRZ[q]
            zs = k
        f[q] = wp.float32(f_in[qq, zs, ys, xs]) + WQ[q]

    rho = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    for q in range(19):
        rho += f[q]
        mx += f[q] * float(CX[q])
        my += f[q] * float(CY[q])
        mz += f[q] * float(CZ[q])
    inv_rho = 1.0 / rho
    ux = mx * inv_rho
    uy = my * inv_rho
    uz = mz * inv_rho

    # body forces: driving pressure gradient, wall stress, actuator disks
    fx = p.force[0]
    fy = p.force[1]
    fz = p.force[2]
    if p.bottom == WALL and k == 0:
        uh = wp.sqrt(ux * ux + uy * uy)
        cd = KAP / wp.log(0.5 / p.z0)
        fx -= rho * cd * cd * uh * ux
        fy -= rho * cd * cd * uh * uy
    disk = int(-1)
    wdisk = float(0.0)
    for n in range(p.n_turb):
        w = disk_weight(float(i), float(j), float(k), tpos[n], trad[n], p.sigma)
        if w > 0.0:
            fx -= tcoef[n] * ud[n] * wp.abs(ud[n]) * w
            disk = n
            wdisk = w
    ux += 0.5 * fx * inv_rho
    uy += 0.5 * fy * inv_rho
    uz += 0.5 * fz * inv_rho
    if disk >= 0:
        wp.atomic_add(acc, disk, wdisk * ux)

    # non-equilibrium momentum flux
    usq = ux * ux + uy * uy + uz * uz
    pxx = float(0.0)
    pyy = float(0.0)
    pzz = float(0.0)
    pxy = float(0.0)
    pxz = float(0.0)
    pyz = float(0.0)
    for q in range(19):
        cx = float(CX[q])
        cy = float(CY[q])
        cz = float(CZ[q])
        cu = cx * ux + cy * uy + cz * uz
        fneq = f[q] - WQ[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        pxx += cx * cx * fneq
        pyy += cy * cy * fneq
        pzz += cz * cz * fneq
        pxy += cx * cy * fneq
        pxz += cx * cz * fneq
        pyz += cy * cz * fneq

    # Smagorinsky relaxation time (Mason-Thomson damping near a modelled wall), sponge layer
    tau = p.tau0
    if p.cs2 > 0.0:
        l2 = p.cs2
        if p.bottom == WALL:
            kz = KAP * (float(k) + 0.5)
            l2 = 1.0 / (1.0 / p.cs2 + 1.0 / (kz * kz))
        # strain-rate part of Pi: add back the second-order moment of the Guo forcing
        sxx = pxx + fx * ux
        syy = pyy + fy * uy
        szz = pzz + fz * uz
        sxy = pxy + 0.5 * (fx * uy + fy * ux)
        sxz = pxz + 0.5 * (fx * uz + fz * ux)
        syz = pyz + 0.5 * (fy * uz + fz * uy)
        pi2 = sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)
        tau = 0.5 * (p.tau0 + wp.sqrt(p.tau0 * p.tau0 + 25.455844122 * l2 * wp.sqrt(pi2) * inv_rho))
    if p.sponge_start > 0 and i > p.sponge_start:
        s = float(i - p.sponge_start) / float(p.nx - 1 - p.sponge_start)
        tau += 3.0 * p.sponge_nu * s * s
    omega = 1.0 / tau

    # regularized collision with Guo forcing; the first-order non-equilibrium moment is -F/2 (from the
    # half-force velocity shift) and has to be kept, or the fluid only receives (3 - omega)/2 of the force
    jx = -0.5 * fx
    jy = -0.5 * fy
    jz = -0.5 * fz
    for q in range(19):
        cx = float(CX[q])
        cy = float(CY[q])
        cz = float(CZ[q])
        cu = cx * ux + cy * uy + cz * uz
        feq = WQ[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        hpi = ((cx * cx - 1.0 / 3.0) * pxx + (cy * cy - 1.0 / 3.0) * pyy + (cz * cz - 1.0 / 3.0) * pzz
               + 2.0 * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz))
        f1 = WQ[q] * (3.0 * (cx * jx + cy * jy + cz * jz) + 4.5 * hpi)
        sq = WQ[q] * (3.0 * ((cx - ux) * fx + (cy - uy) * fy + (cz - uz) * fz)
                      + 9.0 * cu * (cx * fx + cy * fy + cz * fz))
        f_out[q, k, j, i] = f_out.dtype(feq + (1.0 - omega) * f1 + (1.0 - 0.5 * omega) * sq - WQ[q])


@wp.kernel
def init_equilibrium_kernel(f: wp.array4d(dtype=Any), rho: wp.array3d(dtype=float),
                            u: wp.array3d(dtype=wp.vec3)):
    k, j, i = wp.tid()
    v = u[k, j, i]
    r = rho[k, j, i]
    usq = wp.dot(v, v)
    for q in range(19):
        cu = float(CX[q]) * v[0] + float(CY[q]) * v[1] + float(CZ[q]) * v[2]
        f[q, k, j, i] = f.dtype(WQ[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq) - WQ[q])


@wp.kernel
def macroscopic_kernel(f: wp.array4d(dtype=Any), rho: wp.array3d(dtype=float),
                       u: wp.array3d(dtype=wp.vec3)):
    k, j, i = wp.tid()
    r = float(0.0)
    m = wp.vec3()
    for q in range(19):
        fq = wp.float32(f[q, k, j, i]) + WQ[q]
        r += fq
        m += fq * wp.vec3(float(CX[q]), float(CY[q]), float(CZ[q]))
    rho[k, j, i] = r
    u[k, j, i] = m / r


@wp.kernel
def plane_stats_kernel(f: wp.array4d(dtype=Any), k: int, acc: wp.array3d(dtype=float)):
    """Running sums of u, v, w, uu, vv, ww, uv on the horizontal plane k."""
    j, i = wp.tid()
    r = float(0.0)
    m = wp.vec3()
    for q in range(19):
        fq = wp.float32(f[q, k, j, i]) + WQ[q]
        r += fq
        m += fq * wp.vec3(float(CX[q]), float(CY[q]), float(CZ[q]))
    u = m / r
    acc[0, j, i] += u[0]
    acc[1, j, i] += u[1]
    acc[2, j, i] += u[2]
    acc[3, j, i] += u[0] * u[0]
    acc[4, j, i] += u[1] * u[1]
    acc[5, j, i] += u[2] * u[2]
    acc[6, j, i] += u[0] * u[1]


@wp.kernel
def band_stats_kernel(f: wp.array4d(dtype=Any), k0: int, acc: wp.array3d(dtype=float)):
    """Running sum of u on the horizontal planes k0 .. k0 + acc.shape[0] - 1."""
    b, j, i = wp.tid()
    k = k0 + b
    r = float(0.0)
    m = float(0.0)
    for q in range(19):
        fq = wp.float32(f[q, k, j, i]) + WQ[q]
        r += fq
        m += fq * float(CX[q])
    acc[b, j, i] += m / r


@wp.kernel
def plane_velocity_kernel(f: wp.array4d(dtype=Any), k: int, out: wp.array3d(dtype=float)):
    j, i = wp.tid()
    r = float(0.0)
    m = wp.vec3()
    for q in range(19):
        fq = wp.float32(f[q, k, j, i]) + WQ[q]
        r += fq
        m += fq * wp.vec3(float(CX[q]), float(CY[q]), float(CZ[q]))
    out[0, j, i] = m[0] / r
    out[1, j, i] = m[1] / r
    out[2, j, i] = m[2] / r


@wp.kernel
def copy_inlet_kernel(src: wp.array4d(dtype=Any), dst: wp.array4d(dtype=Any), i_src: int):
    q, k, j = wp.tid()
    dst[q, k, j, 0] = dst.dtype(wp.float32(src[q, k, j % src.shape[2], i_src]))


@wp.kernel
def inlet_profile_kernel(dst: wp.array4d(dtype=Any), ux: wp.array(dtype=float)):
    q, k, j = wp.tid()
    u = ux[k]
    cu = float(CX[q]) * u
    dst[q, k, j, 0] = dst.dtype(WQ[q] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u * u) - WQ[q])


@wp.kernel
def outlet_kernel(f: wp.array4d(dtype=Any), nx: int):
    q, k, j = wp.tid()
    f[q, k, j, nx - 1] = f[q, k, j, nx - 2]


@wp.kernel
def disk_weight_sum_kernel(p: Params, tpos: wp.array(dtype=wp.vec3), trad: wp.array(dtype=float),
                           wsum: wp.array(dtype=float)):
    k, j, i = wp.tid()
    for n in range(p.n_turb):
        w = disk_weight(float(i), float(j), float(k), tpos[n], trad[n], p.sigma)
        if w > 0.0:
            wp.atomic_add(wsum, n, w)


@wp.kernel
def turbine_update_kernel(acc: wp.array(dtype=float), wsum: wp.array(dtype=float), ud: wp.array(dtype=float),
                          hist: wp.array2d(dtype=float), counter: wp.array(dtype=int)):
    n = wp.tid()
    v = acc[n] / wsum[n]
    ud[n] = v
    c = counter[0]
    if c < hist.shape[0]:
        hist[c, n] = v
    acc[n] = 0.0


@wp.kernel
def advance_counter_kernel(counter: wp.array(dtype=int)):
    counter[0] = counter[0] + 1


def shapiro_correction(ct_prime: float, sigma: float, radius: float) -> float:
    """Filtered actuator disk correction M (Shapiro et al. 2019) for a Gaussian of standard deviation sigma."""
    delta = math.sqrt(12.0) * sigma
    return 1.0 / (1.0 + ct_prime * delta / (4.0 * math.sqrt(3.0 * math.pi) * radius))


class LBM:
    """One lattice Boltzmann domain of nx x ny x nz cells on a Warp device."""

    def __init__(self, nx: int, ny: int, nz: int, *, device: str = "cuda:0", periodic_x: bool = True,
                 shift_y: int = 0, bottom: str = "free_slip", top: str = "free_slip", nu: float = 1e-4,
                 smagorinsky: float = 0.0, z0: float = 0.0, force=(0.0, 0.0, 0.0), sponge_cells: int = 0,
                 sponge_nu: float = 0.0, storage: str = "fp32"):
        self.shape = (nz, ny, nx)
        self.device = device
        self.dtype = STORAGE[storage]
        self.f = wp.zeros((19, nz, ny, nx), dtype=self.dtype, device=device)
        self.g = wp.zeros_like(self.f)
        p = Params()
        p.nx, p.ny, p.nz = nx, ny, nz
        p.periodic_x = int(periodic_x)
        p.shift_y = int(shift_y)
        p.bottom, p.top = BOUNDARY[bottom], BOUNDARY[top]
        p.tau0 = 0.5 + 3.0 * nu
        p.cs2 = smagorinsky ** 2
        p.z0 = z0
        p.force = wp.vec3(*force)
        p.sponge_start = nx - 1 - sponge_cells if sponge_cells > 0 else 0
        p.sponge_nu = sponge_nu
        p.n_turb = 0
        p.sigma = 1.0
        self.p = p
        self.steps = 0
        self.inlet_profile = None
        self._turbine_arrays(np.zeros((0, 3)), np.zeros(0), np.zeros(0), hist_len=1)
        self.plane = None
        self.band = None

    @property
    def nx(self) -> int:
        return self.shape[2]

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape))

    def bytes_allocated(self) -> int:
        return 2 * 19 * wp.types.type_size_in_bytes(self.dtype) * self.cells

    # --- state ------------------------------------------------------------------------------------------------

    def init_equilibrium(self, u, rho=None) -> None:
        """Equilibrium populations for a velocity field (3, nz, ny, nx) and density (default 1)."""
        u = np.asarray(u, dtype=np.float32)
        rho = np.ones(self.shape, np.float32) if rho is None else np.asarray(rho, dtype=np.float32)
        u_wp = wp.array(np.ascontiguousarray(np.moveaxis(u, 0, -1)), dtype=wp.vec3, device=self.device)
        rho_wp = wp.array(rho, dtype=float, device=self.device)
        wp.launch(init_equilibrium_kernel, dim=self.shape, inputs=[self.f, rho_wp, u_wp], device=self.device)

    def set_force(self, force) -> None:
        self.p.force = wp.vec3(*[float(v) for v in force])

    def copy_state_from(self, other: "LBM") -> None:
        wp.copy(self.f, other.f)

    def tile_state_from(self, other: "LBM") -> None:
        """Fill this domain by periodic tiling of another one with the same nz (e.g. a precursor)."""
        src = other.f.numpy()
        nz, ny, nx = self.shape
        jj = np.arange(ny) % src.shape[2]
        ii = np.arange(nx) % src.shape[3]
        self.f = wp.array(src[:, :, jj][:, :, :, ii], dtype=self.dtype, device=self.device)
        self.g = wp.zeros_like(self.f)

    def save(self) -> np.ndarray:
        return self.f.numpy().copy()

    def load(self, f: np.ndarray) -> None:
        self.f = wp.array(f, dtype=self.dtype, device=self.device)
        self.g = wp.zeros_like(self.f)

    # --- actuator disks ---------------------------------------------------------------------------------------

    def _turbine_arrays(self, centers, radius, coef, hist_len: int) -> None:
        n = len(radius)
        d = self.device
        self.tpos = wp.array(np.asarray(centers, np.float32).reshape(-1, 3), dtype=wp.vec3, device=d)
        self.trad = wp.array(np.asarray(radius, np.float32), dtype=float, device=d)
        self.tcoef = wp.array(np.asarray(coef, np.float32), dtype=float, device=d)
        self.ud = wp.zeros(max(n, 1), dtype=float, device=d)
        self.acc = wp.zeros(max(n, 1), dtype=float, device=d)
        self.wsum = wp.zeros(max(n, 1), dtype=float, device=d)
        self.hist = wp.zeros((hist_len, max(n, 1)), dtype=float, device=d)
        self.counter = wp.zeros(1, dtype=int, device=d)
        self.p.n_turb = n

    def set_turbines(self, centers, radius, ct_prime, sigma: float = 0.75, correction: bool = True,
                     u_init=None, hist_len: int = 1) -> dict:
        """Actuator disks at lattice coordinates ``centers`` (n, 3) with radii and local thrust coefficients.

        Returns the disk weight sums and correction factors. ``u_init`` seeds the disk velocity of the first step.
        """
        radius = np.broadcast_to(np.asarray(radius, float), (len(centers),)).copy()
        ct_prime = np.broadcast_to(np.asarray(ct_prime, float), (len(centers),)).copy()
        m = np.array([shapiro_correction(c, sigma, r) if correction else 1.0 for c, r in zip(ct_prime, radius)])
        self.p.sigma = float(sigma)
        self._turbine_arrays(centers, radius, np.zeros(len(radius)), hist_len)
        wp.launch(disk_weight_sum_kernel, dim=self.shape, inputs=[self.p, self.tpos, self.trad, self.wsum],
                  device=self.device)
        wsum = self.wsum.numpy()[: len(radius)]
        area = np.pi * radius ** 2
        coef = 0.5 * ct_prime * m ** 2 * area / wsum
        self.tcoef = wp.array(coef.astype(np.float32), dtype=float, device=self.device)
        if u_init is not None:
            self.ud = wp.array(np.full(len(radius), u_init, np.float32), dtype=float, device=self.device)
        self.disk = {"wsum": wsum, "M": m, "area": area, "ct_prime": ct_prime, "coef": coef}
        return self.disk

    def disk_velocity_history(self) -> np.ndarray:
        """Filtered disk velocity of each turbine at every recorded step (rows up to the step counter)."""
        n = int(self.counter.numpy()[0])
        return self.hist.numpy()[: min(n, self.hist.shape[0]), : self.p.n_turb]

    def thrust(self) -> np.ndarray:
        """Current thrust of each disk (lattice units), from the last disk velocity."""
        ud = self.ud.numpy()[: self.p.n_turb]
        return self.disk["coef"] * self.disk["wsum"] * ud * np.abs(ud)

    # --- time stepping ----------------------------------------------------------------------------------------

    def step(self, n: int = 1, inlet: "LBM | None" = None, inlet_plane: int = -1) -> None:
        """Advance n steps. Open domains take their inflow plane from ``inlet`` (another domain, stepped first)
        or from ``self.inlet_profile`` (equilibrium with u(z))."""
        nz, ny, nx = self.shape
        d = self.device
        for _ in range(n):
            wp.launch(stream_collide, dim=self.shape,
                      inputs=[self.f, self.g, self.p, self.tpos, self.trad, self.tcoef, self.ud, self.acc], device=d)
            if not self.p.periodic_x:
                if inlet is not None:
                    i_src = inlet_plane % inlet.nx
                    wp.launch(copy_inlet_kernel, dim=(19, nz, ny), inputs=[inlet.f, self.g, i_src], device=d)
                elif self.inlet_profile is not None:
                    wp.launch(inlet_profile_kernel, dim=(19, nz, ny), inputs=[self.g, self.inlet_profile], device=d)
                wp.launch(outlet_kernel, dim=(19, nz, ny), inputs=[self.g, nx], device=d)
            if self.p.n_turb > 0:
                wp.launch(turbine_update_kernel, dim=self.p.n_turb,
                          inputs=[self.acc, self.wsum, self.ud, self.hist, self.counter], device=d)
                wp.launch(advance_counter_kernel, dim=1, inputs=[self.counter], device=d)
            if self.plane is not None:
                wp.launch(plane_stats_kernel, dim=(ny, nx), inputs=[self.g, self.plane_k, self.plane], device=d)
                self.plane_n += 1
            if self.band is not None:
                wp.launch(band_stats_kernel, dim=self.band.shape, inputs=[self.g, self.band_k0, self.band], device=d)
                self.band_n += 1
            self.f, self.g = self.g, self.f
            self.steps += 1

    def set_inlet_profile(self, ux_of_k) -> None:
        self.inlet_profile = wp.array(np.asarray(ux_of_k, np.float32), dtype=float, device=self.device)

    # --- diagnostics ------------------------------------------------------------------------------------------

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray]:
        """Density (nz, ny, nx) and velocity (3, nz, ny, nx), without the half-force correction."""
        rho = wp.zeros(self.shape, dtype=float, device=self.device)
        u = wp.zeros(self.shape, dtype=wp.vec3, device=self.device)
        wp.launch(macroscopic_kernel, dim=self.shape, inputs=[self.f, rho, u], device=self.device)
        return rho.numpy(), np.moveaxis(u.numpy(), -1, 0)

    def plane_velocity(self, k: int) -> np.ndarray:
        """Instantaneous velocity (3, ny, nx) on the horizontal plane k."""
        nz, ny, nx = self.shape
        out = wp.zeros((3, ny, nx), dtype=float, device=self.device)
        wp.launch(plane_velocity_kernel, dim=(ny, nx), inputs=[self.f, int(k), out], device=self.device)
        return out.numpy()

    def start_plane_stats(self, k: int) -> None:
        nz, ny, nx = self.shape
        self.plane_k = int(k)
        self.plane = wp.zeros((7, ny, nx), dtype=float, device=self.device)
        self.plane_n = 0

    def start_band_stats(self, k0: int, k1: int) -> None:
        """Accumulate the streamwise velocity on planes k0 .. k1 - 1 from now on."""
        nz, ny, nx = self.shape
        self.band_k0 = int(k0)
        self.band = wp.zeros((int(k1) - int(k0), ny, nx), dtype=float, device=self.device)
        self.band_n = 0

    def band_sums(self) -> tuple[np.ndarray, int]:
        """Running sum of u on the band planes and the number of steps in it."""
        return self.band.numpy().copy(), self.band_n      # copy: on the CPU numpy() is a view

    def plane_stats(self) -> dict:
        """Mean u, v, w and (co)variances on the statistics plane since ``start_plane_stats``."""
        s = self.plane.numpy() / max(self.plane_n, 1)
        mean = s[:3]
        return {"u": mean[0], "v": mean[1], "w": mean[2], "uu": s[3] - mean[0] ** 2, "vv": s[4] - mean[1] ** 2,
                "ww": s[5] - mean[2] ** 2, "uv": s[6] - mean[0] * mean[1], "n": self.plane_n}


def benchmark(sim: LBM, steps: int = 200, **kw) -> float:
    """Million lattice updates per second over ``steps`` steps (after one warm-up step)."""
    sim.step(1, **kw)
    wp.synchronize_device(sim.device)
    t0 = time.perf_counter()
    sim.step(steps, **kw)
    wp.synchronize_device(sim.device)
    return sim.cells * steps / (time.perf_counter() - t0) / 1e6
