"""D3Q19 lattice: velocities, weights, index maps, and NumPy reference versions of the moments.

The GPU kernels in ``solver.py`` hard-code the same tables; the NumPy functions here are small and slow and are
used by the tests and for building initial states on the host.

Populations are stored shifted by their weight (f - w), which keeps more significant digits for the small
deviations from rest that carry the flow (Lehmann et al. 2022, "Accuracy and performance of the lattice
Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats").

Usage:
    feq = equilibrium(rho, u)          # rho (...), u (3, ...) -> (19, ...)
    rho, u = moments(f)                # inverse, without forcing
"""
from __future__ import annotations

import numpy as np

Q = 19
CS2 = 1.0 / 3.0
C = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
], dtype=np.int32)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPPOSITE = np.array([int(np.flatnonzero((C == -c).all(axis=1))[0]) for c in C], dtype=np.int32)
MIRROR_Z = np.array([int(np.flatnonzero((C == c * [1, 1, -1]).all(axis=1))[0]) for c in C], dtype=np.int32)


def equilibrium(rho, u) -> np.ndarray:
    """Second-order equilibrium; ``rho`` of any shape, ``u`` with a leading axis of 3."""
    rho, u = np.asarray(rho, dtype=float), np.asarray(u, dtype=float)
    cu = np.tensordot(C.astype(float), u, axes=(1, 0))
    usq = (u ** 2).sum(axis=0)
    w = W.reshape((Q,) + (1,) * rho.ndim)
    return w * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usq)


def moments(f) -> tuple[np.ndarray, np.ndarray]:
    """Density and velocity of unshifted populations (19, ...)."""
    f = np.asarray(f, dtype=float)
    rho = f.sum(axis=0)
    u = np.tensordot(C.T.astype(float), f, axes=(1, 0)) / rho
    return rho, u
