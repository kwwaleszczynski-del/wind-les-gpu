"""Kelmarsh layout and the rotation into the wind-aligned simulation frame.

Turbine positions and hub heights come from the static file of the open Kelmarsh dataset (C. Plumley and
R. Takeuchi, Cubico Sustainable Investments, https://doi.org/10.5281/zenodo.16807551, CC-BY-4.0); the six
rows are copied here so the solver runs without the data download (``read_static`` checks them against the file).

Directions follow the meteorological convention: degrees clockwise from north, where the wind comes from. In the
simulation frame x points downwind and y to the left of the flow (right-handed with z up).

Usage:
    x, y, hub = kelmarsh()                         # metres east / north of the farm centre, hub heights
    xr, yr = to_wind_frame(x, y, wd=240.0)         # downwind / crosswind coordinates
    free = free_stream(x, y, np.arange(360))        # IEC 61400-12-1 undisturbed sectors (as in wind-wake-shadow)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8
DIAMETER_M = 92.0
STATIC = pd.DataFrame({
    "turbine": [1, 2, 3, 4, 5, 6],
    "lat": [52.400604, 52.402551, 52.403834, 52.398781, 52.402308, 52.400687],
    "lon": [-0.947133, -0.949527, -0.94419, -0.94115, -0.940537, -0.936093],
    "hub_height_m": [78.5, 78.5, 68.5, 78.5, 78.5, 68.5],
}).set_index("turbine")


def to_local(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Metres east and north of the mean position (equirectangular, sub-metre accurate over a few km)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    x = np.deg2rad(lon - lon.mean()) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat.mean()))
    y = np.deg2rad(lat - lat.mean()) * EARTH_RADIUS_M
    return x, y


def kelmarsh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = to_local(STATIC["lat"], STATIC["lon"])
    return x, y, STATIC["hub_height_m"].to_numpy()


def read_static(data_dir: Path) -> pd.DataFrame:
    """Positions and hub heights from the dataset's static CSV (for checking the copy above)."""
    df = pd.read_csv(Path(data_dir) / "raw" / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
    return pd.DataFrame({
        "turbine": df["Title"].str.extract(r"(\d+)$", expand=False).astype(int),
        "lat": df["Latitude"], "lon": df["Longitude"], "hub_height_m": df["Hub Height (m)"].astype(float),
    }).set_index("turbine").sort_index()


def to_wind_frame(x, y, wd: float) -> tuple[np.ndarray, np.ndarray]:
    """Downwind (x') and crosswind (y', to the left looking downwind) coordinates for wind from ``wd``."""
    th = np.deg2rad(wd)
    down = np.array([-np.sin(th), -np.cos(th)])
    left = np.array([np.cos(th), -np.sin(th)])
    pts = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1)
    return pts @ down, pts @ left


def from_wind_frame(xr, yr, wd: float) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(wd)
    down = np.array([-np.sin(th), -np.cos(th)])
    left = np.array([np.cos(th), -np.sin(th)])
    xr, yr = np.asarray(xr, float), np.asarray(yr, float)
    return xr * down[0] + yr * left[0], xr * down[1] + yr * left[1]


def bearing_deg(x0, y0, x1, y1):
    return np.rad2deg(np.arctan2(np.asarray(x1) - x0, np.asarray(y1) - y0)) % 360.0


def angle_diff(a, b):
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


def free_stream(x, y, wd, diameter_m: float = DIAMETER_M, max_d: float = 20.0) -> np.ndarray:
    """(len(wd), n) True where no neighbour within 20 D lies in the IEC 61400-12-1 Annex A disturbed sector."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    wd = np.atleast_1d(np.asarray(wd, dtype=float))
    dist = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
    bearing = bearing_deg(x[:, None], y[:, None], x[None, :], y[None, :])
    width = 1.3 * np.rad2deg(np.arctan(2.5 * diameter_m / np.where(dist > 0, dist, np.inf) + 0.15)) + 10.0
    near = (dist > 0) & (dist <= max_d * diameter_m)
    inside = np.abs(angle_diff(wd[:, None, None], bearing[None])) < width[None] / 2.0
    return ~(inside & near[None]).any(axis=2)
