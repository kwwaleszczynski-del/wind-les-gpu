"""Figures for the README and the web page (matplotlib, PNG). Each one is skipped if its inputs are missing.

Colours: one fixed categorical order for the sources in every figure (SCADA, LES, then the PyWake models),
a single blue ramp for magnitudes (velocity maps, resolutions).

Usage:
    all_figures(Path("results"), Path("docs/img"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wind_les import abl, farm, layout, study  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SOURCES = {  # fixed categorical order
    "scada": ("SCADA", "#2a78d6"),
    "les": ("LES (this work)", "#eb6834"),
    "pywake_gaussian": ("PyWake Gaussian", "#1baf7a"),
    "pywake_jensen": ("PyWake Jensen", "#eda100"),
    "pywake_turbopark": ("PyWake TurbOPark", "#e87ba4"),
}
RES_COLOURS = {8: "#9ec5f4", 12: "#2a78d6", 16: "#0d3f80"}   # ordinal blue ramp, coarse -> fine
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2.0,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", path)


def verification(results: Path, out: Path) -> None:
    tgv = pd.read_csv(results / "verify_taylor_green.csv")
    disk = pd.read_csv(results / "verify_disk.csv")
    n = 8
    prof = pd.read_csv(results / f"precursor_profile_D{n}.csv")
    setup = abl.Setup(cells_per_d=n)
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))

    a = ax[0, 0]
    err = np.abs(tgv["nu_ratio"] - 1)
    a.loglog(tgv["n"], err, "o-", color=SOURCES["les"][1], ms=7, label="measured")
    a.loglog(tgv["n"], err.iloc[0] * (tgv["n"] / tgv["n"].iloc[0]) ** -2.0, "--", color=MUTED, lw=1.2,
             label="second order")
    a.set(xlabel="cells per wavelength", ylabel="|viscosity error|", title="Taylor-Green vortex decay")
    a.set_xticks(tgv["n"], [str(v) for v in tgv["n"]])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.legend()

    a = ax[0, 1]
    ctp = np.linspace(0.3, 2.7, 100)
    a.plot(ctp, ctp / (4 + ctp), color=MUTED, lw=1.2, label="momentum theory a")
    a.plot(ctp, 16 * ctp / (4 + ctp) ** 2, color=INK2, lw=1.2, ls="--", label="momentum theory $C_T$")
    for res, g in disk.groupby("cells_per_d"):
        c = RES_COLOURS.get(int(res), INK2)
        a.plot(g["ct_prime"], g["a"], "o", color=c, ms=7, label=f"LES a, D/{res}")
        a.plot(g["ct_prime"], g["ct"], "s", color=c, ms=7, mfc=SURFACE, mew=1.8, label=f"LES $C_T$, D/{res}")
    a.set(xlabel="local thrust coefficient $C_T'$", ylabel="induction a, thrust coefficient $C_T$",
          title="Actuator disk in uniform inflow", ylim=(0, 1.3))
    a.legend(fontsize=8, ncol=2, loc="upper left")

    a = ax[1, 0]
    us = setup.ustar_lb
    z = prof["z_m"]
    a.semilogy(prof["u"] / us, z, "o", color=SOURCES["les"][1], ms=6, label=f"LES precursor, D/{n}")
    zz = np.geomspace(z.min(), 200, 50)
    a.semilogy(np.log(zz / setup.z0_m) / 0.4, zz, color=MUTED, lw=1.2, label=f"log law, $z_0$ = {setup.z0_m} m")
    a.axhline(setup.hub_m, color=GRID, lw=1.0, ls=":")
    a.text(0.98, setup.hub_m * 1.05, "hub", color=INK2, fontsize=8, ha="right", transform=a.get_yaxis_transform())
    a.set(xlabel="$U / u_*$", ylabel="height [m]", title="Inflow: mean profile", ylim=(4, 220))
    a.legend(loc="upper left")

    a = ax[1, 1]
    for col, label, c in (("su", r"$\sigma_u$", "#2a78d6"), ("sv", r"$\sigma_v$", "#eb6834"),
                          ("sw", r"$\sigma_w$", "#1baf7a")):
        a.plot(prof[col] / us, z, color=c, label=label)
    a.axhline(setup.hub_m, color=GRID, lw=1.0, ls=":")
    info = json.loads((results / f"precursor_D{n}.json").read_text())
    a.set(xlabel="standard deviation / $u_*$", ylabel="height [m]", ylim=(0, 300),
          title=f"Inflow turbulence (hub TI {100 * info['ti_hub']:.1f} %)")
    a.legend()
    fig.tight_layout()
    _save(fig, out / "verification.png")


def _turbine_marks(ax, centres, setup, origin, wd_label=True):
    for i, (xc, yc, _) in enumerate(centres):
        x, y = (xc - origin[0]) * setup.dx / 1000, (yc - origin[1]) * setup.dx / 1000
        r = setup.diameter_m / 2000
        ax.plot([x, x], [y - r, y + r], color=INK, lw=2.2, solid_capstyle="round")
        ax.text(x - 0.03, y + r + 0.02, f"T{i + 1}", fontsize=7.5, color=INK, ha="right")


def hub_plane(results: Path, out: Path, n: int = 8, directions=(240.0, 300.0)) -> None:
    setup = abl.Setup(cells_per_d=n)
    geo = farm.geometry(setup)
    runs = {wd: study.load_run(study.run_path(results, n, wd)) for wd in directions
            if study.run_path(results, n, wd).exists()}
    if not runs:
        return
    fig, ax = plt.subplots(len(runs), 2, figsize=(11, 4.3 * len(runs)), squeeze=False)
    for row, (wd, r) in enumerate(runs.items()):
        uref = r["inflow_hub_lb"]
        ext = [-geo.x0 * setup.dx / 1000, (geo.nx - geo.x0) * setup.dx / 1000,
               -geo.y0 * setup.dx / 1000, (geo.ny - geo.y0) * setup.dx / 1000]
        fields = [("mean, 30 min", r["plane_u"] / uref)]
        if len(r["frames"]):
            fields.append(("instantaneous", r["frames"][-1].astype(np.float32) / uref))
        for col, (name, f) in enumerate(fields):
            a = ax[row, col]
            im = a.imshow(f, origin="lower", extent=ext, cmap="Blues_r", vmin=0.3, vmax=1.2, interpolation="bilinear")
            _turbine_marks(a, r["centres"], setup, (geo.x0, geo.y0))
            a.set(title=f"wind from {wd:.0f} deg: {name}", xlabel="downwind [km]", ylabel="crosswind [km]")
            a.grid(False)
        cb = fig.colorbar(im, ax=ax[row].tolist(), shrink=0.85, pad=0.01)
        cb.set_label(f"U / U_hub at z = {r['z_plane_m']:.0f} m")
    _save(fig, out / "hub_plane.png")


def wake_deficit(results: Path, out: Path) -> None:
    py_path = results / "single_wake_pywake.csv"
    les = sorted(results.glob("single_wake_D*.csv"))
    if not les or not py_path.exists():
        return
    py = pd.read_csv(py_path)
    info = json.loads((results / "precursor_D8.json").read_text())
    py = py[np.isclose(py["ti"], round(info["ti_hub"], 4))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in ("pywake_gaussian", "pywake_jensen", "pywake_turbopark"):
        label, c = SOURCES[name]
        ax.plot(py["x_d"], py[name.replace("pywake_", "")], color=c, label=label)
    for path in les:
        d = pd.read_csv(path)
        d = d[d["x_d"] >= 2]
        ax.plot(d["x_d"], d["deficit"], color=SOURCES["les"][1], lw=2.6,
                label=f"{SOURCES['les'][0]}, D/{d['cells_per_d'].iloc[0]}")
    ax.set(xlabel="distance behind the rotor [D]", ylabel="centreline deficit 1 - U/U_inf",
           title=f"Single-wake deficit at hub height, inflow TI {100 * info['ti_hub']:.1f} %", xlim=(2, 15),
           ylim=(0, 0.65))
    ax.legend()
    _save(fig, out / "wake_deficit.png")


def sector_losses(results: Path, out: Path) -> None:
    path = results / "sector_losses.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    names = [k for k in SOURCES if k in df.columns]
    x = np.arange(len(df))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, name in enumerate(names):
        label, c = SOURCES[name]
        pos = x - 0.4 + (i + 0.5) * w
        ax.bar(pos, df[name], width=w - 0.02, color=c, label=label, edgecolor=SURFACE, linewidth=0.6)
        if name == "scada":
            ax.errorbar(pos, df["scada"], yerr=[df["scada"] - df["scada_lo"], df["scada_hi"] - df["scada"]],
                        fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5)
        if name == "les" and "les_se" in df:
            ax.errorbar(pos, df["les"], yerr=2 * df["les_se"], fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5)
    labels = [f"{s} deg" if s != "all" else "all" for s in df["sector_deg"].astype(str)]
    ax.set_xticks(x, labels)
    ax.axvline(len(df) - 1.5, color=MUTED, lw=0.8)
    ax.set(ylabel="wake loss [%]", title="Wake loss per 30-degree sector, same SCADA records for every source")
    ax.legend(ncol=5, loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, out / "sector_losses.png")


def efficiency_direction(results: Path, out: Path) -> None:
    path = results / "efficiency_by_direction.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    df = df[df["records"] >= 100]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for name in [k for k in SOURCES if k in df.columns]:
        label, c = SOURCES[name]
        lw = 2.6 if name in ("scada", "les") else 1.4
        ax.plot(df.index, df[name], color=c, lw=lw, label=label)
    ax.set(xlabel="wind direction [deg]", ylabel="farm efficiency", xlim=(0, 360),
           title="Farm efficiency by direction (5-degree bins, bins with >= 100 records)")
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(ncol=5, loc="lower left")
    _save(fig, out / "efficiency_direction.png")


def performance(results: Path, out: Path) -> None:
    bench = results / "benchmark.csv"
    grid = results / "grid_convergence.csv"
    if not bench.exists():
        return
    series = [("float32", pd.read_csv(bench), "o-", SOURCES["les"][1])]
    storage = results / "benchmark_storage.csv"
    if storage.exists():
        s = pd.read_csv(storage)
        series.append(("float16", s[s["storage"] == "fp16"], "s--", INK2))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    a = ax[0]
    for name, b, style, colour in series:
        a.plot(b["cells"] / 1e6, b["mlups"], style, color=colour, ms=7, lw=1.6, label=name)
        last = b.iloc[-1]   # population memory of the largest grid of each series
        a.annotate(f"{last['population_mb'] / 1024:.1f} GB", (last["cells"] / 1e6, last["mlups"]),
                   textcoords="offset points", xytext=(0, -15), ha="center", fontsize=7.5, color=INK2)
    a.set_xscale("log")
    ticks = [0.3, 1, 3, 10, 30]
    a.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a.minorticks_off()
    a.set(xlabel="grid size [million cells]", ylabel="MLUPS", ylim=(0, 1600),
          title="Bare kernel throughput (label: population memory)")
    a.legend(fontsize=8, loc="lower right")
    a = ax[1]
    if grid.exists():
        g = pd.read_csv(grid).groupby("cells_per_d").agg(wall_s=("wall_s", "mean"), gpu_mb=("gpu_mb", "max"),
                                                          cells=("cells", "first"))
        a.bar([f"D/{i}" for i in g.index], g["wall_s"] / 60, color=[RES_COLOURS.get(int(i), INK2) for i in g.index],
              width=0.55)
        for k, (i, r) in enumerate(g.iterrows()):
            a.text(k, r["wall_s"] / 60, f"{r['cells'] / 1e6:.1f} M cells\n{r['gpu_mb'] / 1024:.1f} GB on GPU",
                   ha="center", va="bottom", fontsize=7.5, color=INK2)
        a.set(ylabel="minutes per wind direction", title="Cost of one direction (10 min spin-up + 30 min average)",
              ylim=(0, (g["wall_s"] / 60).max() * 1.3))
        a.grid(axis="x", visible=False)
        a.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out / "performance.png")


def grid_convergence(results: Path, out: Path) -> None:
    path = results / "grid_convergence.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    x, y, _ = layout.kelmarsh()
    dirs = sorted(df["wd"].unique())
    fig, ax = plt.subplots(1, len(dirs), figsize=(5.2 * len(dirs), 3.8), squeeze=False, sharey=True)
    for a, wd in zip(ax[0], dirs):
        free = layout.free_stream(x, y, [wd])[0]
        for _, r in df[df["wd"] == wd].iterrows():
            s = np.array([r[f"s{i}"] for i in range(1, 7)])
            ratio = (s / s[free].mean()) ** 3
            n = int(r["cells_per_d"])
            a.plot(np.arange(1, 7), ratio, "o-", color=RES_COLOURS.get(n, INK2), ms=7, lw=1.4,
                   label=f"D/{n}, TI {100 * r['ti_hub']:.1f} %")
        a.set_xticks(np.arange(1, 7), [f"T{i}" + ("" if f else "*") for i, f in zip(range(1, 7), free)])
        a.set(title=f"wind from {wd:.0f} deg (* = in a wake)", ylabel="(U_i / U_free)^3")
        a.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out / "grid_convergence.png")


def all_figures(results: Path, out: Path) -> None:
    for fn in (verification, hub_plane, wake_deficit, sector_losses, efficiency_direction, performance,
               grid_convergence):
        try:
            fn(results, out)
        except FileNotFoundError as exc:
            log.warning("%s skipped: %s", fn.__name__, exc)
