"""Boundary layer set-up, actuator disk, layout and farm geometry."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from wind_les import abl, farm, layout, turbine

DEVICE = "cpu"


def test_setup_scaling():
    s = abl.Setup(cells_per_d=8)
    assert s.dx == pytest.approx(11.5)
    assert s.log_law(s.hub_m) == pytest.approx(s.u_hub_lb)
    assert s.force_lb * s.nz == pytest.approx(s.ustar_lb ** 2)
    assert s.dt() == pytest.approx(11.5 * 0.05 / 8.0)
    assert s.k_at(0.5 * s.dx) == pytest.approx(0.0)


def test_precursor_runs_and_stays_finite():
    s = abl.Setup(cells_per_d=4, height_m=184.0, precursor_d=4.0)
    pre = abl.make_precursor(s, ny=8, device=DEVICE)
    hist = abl.spin_up(pre, s, steps=60, every=30)
    assert np.isfinite(hist).all()
    assert 0.6 * s.u_hub_lb < hist[-1, 1] < 1.4 * s.u_hub_lb


def test_momentum_theory():
    m = turbine.momentum(turbine.ct_prime_from_ct(0.8))
    assert m["ct"] == pytest.approx(0.8)
    assert m["a"] == pytest.approx(0.5 * (1 - math.sqrt(0.2)))


def test_power_curve():
    assert turbine.power_kw(8.0) == pytest.approx(991.2)
    assert turbine.power_kw(2.5) == 0.0 and turbine.power_kw(26.0) == 0.0


def test_isolated_disk_close_to_momentum_theory():
    """Coarse disk (4 cells per D) on the CPU: induction within 25 % of momentum theory."""
    r = turbine.isolated_disk(4, 1.5, DEVICE, side_d=6.0, up_d=3.0, down_d=5.0, flow_throughs=1.5)
    assert r["a"] == pytest.approx(r["theory"]["a"], rel=0.25)
    assert r["ct"] == pytest.approx(r["theory"]["ct"], rel=0.25)


def test_wind_frame_rotation():
    # wind from the west blows towards the east: east is downwind, north is to the left
    xr, yr = layout.to_wind_frame([100.0, 0.0], [0.0, 100.0], 270.0)
    assert np.allclose(xr, [100.0, 0.0]) and np.allclose(yr, [0.0, 100.0])
    x, y, _ = layout.kelmarsh()
    for wd in (0.0, 37.0, 240.0):
        back = layout.from_wind_frame(*layout.to_wind_frame(x, y, wd), wd)
        assert np.allclose(back, (x, y))


def test_free_stream_has_a_reference_everywhere():
    x, y, _ = layout.kelmarsh()
    free = layout.free_stream(x, y, np.arange(360))
    assert free.any(axis=1).all()
    assert (~free).any(axis=1).any()


@pytest.mark.skipif(not (Path(os.environ.get("KELMARSH_DATA_DIR", "data")) / "raw" / "Kelmarsh_WT_static.csv").exists(),
                    reason="Kelmarsh static file not available")
def test_layout_matches_dataset():
    st = layout.read_static(Path(os.environ.get("KELMARSH_DATA_DIR", "data")))
    assert np.allclose(st[["lat", "lon", "hub_height_m"]].to_numpy(), layout.STATIC.to_numpy())


def test_geometry_holds_farm_for_all_directions():
    s = abl.Setup(cells_per_d=8)
    geo = farm.geometry(s)
    for wd in np.arange(0, 360, 15):
        c = farm.turbine_centres(s, geo, wd)
        assert (c[:, 0] >= 5 * 8 - 1e-6).all() and (c[:, 0] <= geo.nx - 10 * 8).all()
        assert (c[:, 1] - 4 >= 5 * 8 - 1).all() and (c[:, 1] + 4 <= geo.ny - 5 * 8 + 1).all()
        assert np.allclose(c[:, 2], [s.k_at(h) for h in layout.STATIC["hub_height_m"]])


def test_block_means_and_ratios():
    ud = np.vstack([np.full((10, 2), 9.0), np.tile([[1.0, 2.0]], (12, 1))])
    b = farm.block_means(ud, n_spin=10, dt=1.0, block_s=4.0)
    assert b.shape == (3, 2) and np.allclose(b, [[1.0, 2.0]] * 3)
    # an averaging window one step short of three blocks (time-step rounding) still gives three blocks
    assert farm.block_means(ud[:-1], n_spin=10, dt=1.0, block_s=4.0).shape == (3, 2)
    out = {"ud": ud, "n_spin": 10, "dt": 1.0, "inflow_hub_lb": 2.0}
    assert np.allclose(farm.equivalent_speed_ratio(out, {"r": 0.5}, block_s=4.0), [[1.0, 2.0]] * 3)


def test_numpy_disk_weights_match_the_solver():
    from wind_les.solver import LBM
    sim = LBM(24, 20, 16, device=DEVICE)
    centre = [9.3, 9.7, 6.2]
    info = sim.set_turbines([centre], 4.0, 1.5, sigma=0.75)
    _, w = farm.disk_weights(centre, 4.0, 0.75, sim.shape)
    assert w.sum() == pytest.approx(info["wsum"][0], rel=1e-5)


def test_local_reference_of_a_uniform_field_and_height_factor():
    s = abl.Setup(cells_per_d=8)
    prof = np.log((np.arange(20) + 0.5) * s.dx / s.z0_m)
    blocks = np.broadcast_to(prof[None, :, None, None], (2, 20, 90, 90)).astype(np.float32)
    empty = {"band_blocks": blocks, "band_k0": 0}
    centres = np.array([[40.0, 40.0, s.k_at(78.5)], [50.0, 45.0, s.k_at(68.5)]])
    ref = farm.local_reference(empty, centres, s, 0.75)
    assert ref.shape == (2, 2) and np.allclose(ref[0], ref[1])
    h = farm.height_factor(empty, s, [78.5, 68.5], 0.75)
    assert h[0] == pytest.approx(1.0) and h[1] == pytest.approx(ref[0, 1] / ref[0, 0], rel=1e-6)


def test_farm_run_and_empty_reference_on_a_coarse_grid(monkeypatch):
    """Whole chain on 4 cells per D (CPU): empty run, one direction, ratios against the local reference."""
    monkeypatch.setattr(farm, "BLOCK_S", 2.0)
    s = abl.Setup(cells_per_d=4, height_m=184.0, precursor_d=4.0)
    geo = farm.geometry(s)
    pre = abl.make_precursor(s, geo.ny, DEVICE)
    pre.step(20)
    state = pre.save()
    cal = {"ct_prime": 1.4, "r": 0.8, "sigma": 0.75}
    kw = {"spinup_s": 2.0, "average_s": 6.0}
    empty = farm.run_empty(s, geo, cal, state, s.u_hub_lb, DEVICE, **kw)
    out = farm.run_direction(s, geo, cal, 240.0, state, s.u_hub_lb, DEVICE, **kw)
    assert empty["band_blocks"].shape[0] >= 3 and np.isfinite(empty["band_blocks"]).all()
    ratio = farm.equivalent_speed_ratio(out, cal, block_s=2.0, empty=empty, setup=s)
    assert ratio.shape == (3, 6) and np.isfinite(ratio).all()
