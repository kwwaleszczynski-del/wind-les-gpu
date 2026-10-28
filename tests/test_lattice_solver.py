"""Lattice tables and the solver on tiny grids (CPU, so the tests run without a GPU)."""
import numpy as np
import pytest

from wind_les import lattice
from wind_les.solver import LBM, shapiro_correction
from wind_les.verify import poiseuille, taylor_green

DEVICE = "cpu"


def test_lattice_isotropy():
    c, w = lattice.C.astype(float), lattice.W
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w @ c, 0.0)
    assert np.allclose(np.einsum("q,qa,qb->ab", w, c, c), lattice.CS2 * np.eye(3))
    assert np.allclose(np.einsum("q,qa,qb,qc->abc", w, c, c, c), 0.0)


def test_index_maps():
    c = lattice.C
    assert (c[lattice.OPPOSITE] == -c).all()
    assert (c[lattice.MIRROR_Z] == c * [1, 1, -1]).all()


def test_equilibrium_moments():
    rho = np.array([1.02, 0.97])
    u = np.array([[0.05, -0.02], [0.01, 0.03], [-0.02, 0.0]])
    r, v = lattice.moments(lattice.equilibrium(rho, u))
    assert np.allclose(r, rho) and np.allclose(v, u)


def test_taylor_green_second_order():
    e16 = abs(taylor_green(16, DEVICE)["nu_ratio"] - 1)
    e32 = abs(taylor_green(32, DEVICE)["nu_ratio"] - 1)
    assert e32 < 0.03
    assert e16 / e32 > 3.0          # second order: 4 in the limit


def test_taylor_green_half_precision_storage():
    """float16 populations (shifted by w) change the measured viscosity by less than 0.5 %."""
    half = taylor_green(32, DEVICE, storage="fp16")["nu_ratio"]
    assert half == pytest.approx(taylor_green(32, DEVICE)["nu_ratio"], abs=0.005)


def test_poiseuille_profile():
    assert poiseuille(8, 0.1, DEVICE)["max_rel_err"] < 0.02


def test_uniform_flow_stays_uniform_with_free_slip_and_shift():
    sim = LBM(12, 10, 6, device=DEVICE, periodic_x=True, shift_y=3, nu=1e-3, smagorinsky=0.16)
    u = np.zeros((3, 6, 10, 12))
    u[0] = 0.05
    u[1] = 0.01
    sim.init_equilibrium(u)
    sim.step(50)
    rho, v = sim.macroscopic()
    assert np.allclose(rho, 1.0, atol=1e-5)
    assert np.allclose(v[0], 0.05, atol=1e-5) and np.allclose(v[1], 0.01, atol=1e-5) and np.allclose(v[2], 0, atol=1e-5)


def test_mass_and_momentum_with_disk_force():
    """Guo forcing adds exactly the applied force to the momentum; the disk thrust is the only force here."""
    sim = LBM(24, 12, 12, device=DEVICE, periodic_x=True, nu=1e-3, smagorinsky=0.16)
    u = np.zeros((3, 12, 12, 24), np.float32)
    u[0] = 0.05
    sim.init_equilibrium(u)
    info = sim.set_turbines([[8.0, 5.5, 5.5]], 3.0, 1.5, sigma=0.75, u_init=0.05, hist_len=100)
    rho0, v0 = sim.macroscopic()
    m0 = (rho0 * v0[0]).sum()
    sim.step(40)
    rho1, v1 = sim.macroscopic()
    ud = np.concatenate([[0.05], sim.disk_velocity_history()[:-1, 0]])   # thrust uses the previous step's u_d
    thrust = (info["coef"][0] * info["wsum"][0] * ud ** 2).sum()
    assert rho1.sum() == pytest.approx(rho0.sum(), rel=1e-5)
    assert (rho1 * v1[0]).sum() - m0 == pytest.approx(-thrust, rel=2e-3)


def test_shapiro_correction_limits():
    assert shapiro_correction(1.5, 1e-6, 4.0) == pytest.approx(1.0)
    assert shapiro_correction(1.5, 0.75, 4.0) < shapiro_correction(1.5, 0.75, 8.0) < 1.0
