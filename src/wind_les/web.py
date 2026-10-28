"""Static web page (docs/index.html, GitHub Pages from main:/docs): hub-height flow animation and the results.

The animation frames are the instantaneous hub-plane speed of the LES, quantised to 8 bits (U/U_hub from
0.2 to 1.4) and stored as one binary file per direction; the page draws them on a canvas with a colour table
taken from the matplotlib map used in the figures. No external scripts.

Usage:
    build(Path("results"), Path("docs"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from wind_les import abl, farm, study

VMIN, VMAX = 0.2, 1.4
log = logging.getLogger(__name__)


def export_frames(results: Path, docs: Path, n: int = 8, directions=study.FRAME_DIRECTIONS) -> list[dict]:
    setup = abl.Setup(cells_per_d=n)
    geo = farm.geometry(setup)
    out = []
    for wd in directions:
        path = study.run_path(results, n, wd)
        if not path.exists():
            continue
        r = study.load_run(path)
        frames = r["frames"].astype(np.float32) / r["inflow_hub_lb"]
        if not len(frames):
            continue
        q = np.clip(np.round((frames - VMIN) / (VMAX - VMIN) * 255), 0, 255).astype(np.uint8)
        name = f"frames_{int(wd):03d}.bin"
        (docs / "data").mkdir(parents=True, exist_ok=True)
        (docs / "data" / name).write_bytes(q.tobytes())
        mean = np.clip(np.round((r["plane_u"] / r["inflow_hub_lb"] - VMIN) / (VMAX - VMIN) * 255), 0, 255)
        (docs / "data" / f"mean_{int(wd):03d}.bin").write_bytes(mean.astype(np.uint8).tobytes())
        out.append({"wd": wd, "file": f"data/{name}", "mean": f"data/mean_{int(wd):03d}.bin", "frames": len(q),
                    "nx": geo.nx, "ny": geo.ny, "dt_s": float(r["frame_dt_s"]), "dx_m": setup.dx,
                    "z_m": float(r["z_plane_m"]),
                    "turbines": [[float(c[0]), float(c[1])] for c in r["centres"]], "radius": n / 2})
    return out


def colour_table(name: str = "Blues_r") -> list[str]:
    cmap = matplotlib.colormaps[name]
    return [matplotlib.colors.to_hex(cmap(i / 255)) for i in range(256)]


def build(results: Path, docs: Path) -> Path:
    docs.mkdir(parents=True, exist_ok=True)
    anim = export_frames(results, docs)
    summary = json.loads((results / "run.json").read_text()) if (results / "run.json").exists() else {}
    sectors = pd.read_csv(results / "sector_losses.csv") if (results / "sector_losses.csv").exists() else None
    data = {
        "anim": anim, "lut": colour_table(), "vmin": VMIN, "vmax": VMAX, "summary": summary,
        "sectors": [] if sectors is None else json.loads(sectors.to_json(orient="records")),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = docs / "index.html"
    path.write_text(html, encoding="utf-8")
    log.info("web page %s (%d animations)", path, len(anim))
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kelmarsh GPU LES</title>
<link rel="icon" href="data:,">
<meta name="description" content="Home-made lattice Boltzmann LES of the Kelmarsh wind farm on a laptop GPU, compared with SCADA and PyWake.">
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --panel: #f3f2ef; --ink: #0b0b0b; --ink2: #52514e; --muted: #8a8984; --rule: #e4e3df;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #242422; --ink: #f0efec; --ink2: #c3c2b7; --muted: #8f8e88; --rule: #34332f;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --panel: #242422; --ink: #f0efec; --ink2: #c3c2b7; --muted: #8f8e88; --rule: #34332f;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1060px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 6px; }
h2 { font-size: 1.15rem; margin: 36px 0 8px; }
p { color: var(--ink2); margin: 6px 0 10px; max-width: 74ch; }
a { color: var(--accent); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0 8px; }
.kpi { background: var(--panel); border-radius: 8px; padding: 10px 12px; }
.kpi b { display: block; font-size: 1.35rem; font-variant-numeric: tabular-nums; }
.kpi span { color: var(--ink2); font-size: .82rem; }
.viewer { background: var(--panel); border-radius: 10px; padding: 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; }
.controls button, .controls select { font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.controls input[type=range] { flex: 1 1 160px; accent-color: var(--accent); }
.controls .t { color: var(--ink2); font-variant-numeric: tabular-nums; min-width: 7ch; }
canvas { width: 100%; height: auto; display: block; border-radius: 6px; background: #000; }
.scale { display: flex; align-items: center; gap: 8px; color: var(--ink2); font-size: .8rem; margin-top: 8px; }
.scale canvas { height: 10px; width: 220px; border-radius: 2px; }
figure { margin: 14px 0; }
figure img { width: 100%; height: auto; border-radius: 6px; }
figcaption { color: var(--ink2); font-size: .85rem; }
.chart { position: relative; }
.chart svg { width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .85rem; color: var(--ink2); margin: 6px 0; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px; padding: 6px 9px; font-size: .8rem; box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; white-space: nowrap; }
table { border-collapse: collapse; font-size: .85rem; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
details { margin-top: 8px; color: var(--ink2); }
.wrap { overflow-x: auto; }
footer { margin-top: 40px; color: var(--muted); font-size: .8rem; }
</style>
</head>
<body>
<main>
<h1>Kelmarsh wind farm in a laptop-GPU LES</h1>
<p>A home-made lattice Boltzmann large-eddy simulation (D3Q19, regularized collision, Smagorinsky) written in
NVIDIA Warp and run on a 6 GB Quadro RTX 3000. The six Senvion MM92 turbines are actuator disks in a neutral
boundary layer fed by a periodic precursor. Wake losses per wind sector are compared with the farm's open SCADA
data and with three PyWake engineering models, on the same records.</p>
<div class="kpis" id="kpis"></div>

<h2>Hub-height flow</h2>
<p>Instantaneous streamwise speed at hub height, 4 s between frames. The inflow turbulence comes from the
precursor; every direction sees the same inflow realisation.</p>
<div class="viewer">
  <div class="controls">
    <select id="dir" aria-label="wind direction"></select>
    <button id="play" type="button">Pause</button>
    <button id="mode" type="button">Show 30-min mean</button>
    <input id="frame" type="range" min="0" max="0" value="0" aria-label="frame">
    <span class="t" id="time"></span>
  </div>
  <canvas id="cv" width="640" height="520"></canvas>
  <div class="scale"><span id="vmin"></span><canvas id="bar" width="256" height="1"></canvas><span id="vmax"></span>
    <span>U / U<sub>hub</sub>, downwind to the right</span></div>
</div>

<h2>Wake loss per sector</h2>
<p>Same SCADA records for every source: all six turbines in partial load, free-stream reference speed 5-10 m/s,
2016-2021. Loss = 1 - farm power / (6 x free-stream turbine power). Whiskers: SCADA 95 % day-bootstrap interval,
LES &plusmn;2 standard errors from its three 10-minute blocks.</p>
<div class="legend" id="legend"></div>
<div class="chart" id="chart"><div class="tip" id="tip"></div></div>
<details><summary>Table</summary><div class="wrap" id="table"></div></details>

<h2>Verification and cost</h2>
<figure><img src="img/verification.png" alt="Taylor-Green viscosity convergence, actuator disk against momentum theory, inflow log-law profile and turbulence profiles"><figcaption>Analytical checks, actuator disk against momentum theory, inflow profiles.</figcaption></figure>
<figure><img src="img/wake_deficit.png" alt="Centreline wake deficit behind one turbine against distance, LES and PyWake models"><figcaption>Single-wake deficit behind one turbine.</figcaption></figure>
<figure><img src="img/efficiency_direction.png" alt="Farm efficiency against wind direction for SCADA, LES and PyWake"><figcaption>Farm efficiency by wind direction.</figcaption></figure>
<figure><img src="img/grid_convergence.png" alt="Turbine power ratios at three grid resolutions"><figcaption>Grid convergence.</figcaption></figure>
<figure><img src="img/performance.png" alt="Throughput in million lattice updates per second and run time per direction"><figcaption>Throughput and cost.</figcaption></figure>
<footer>Data: Kelmarsh wind farm data, C. Plumley and R. Takeuchi, Cubico Sustainable Investments,
doi:10.5281/zenodo.16807551, CC-BY-4.0. Engineering models: PyWake (DTU). Code: MIT.</footer>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 1) => (v == null || isNaN(v)) ? "-" : Number(v).toFixed(d);

// KPI row
(function () {
  const s = D.summary, w = s.wake_loss_pct || {}, c = s.les_cost || {};
  const items = [
    [fmt(w.scada, 2) + " %", "SCADA wake loss [" + fmt(w.scada_lo, 1) + "-" + fmt(w.scada_hi, 1) + "]"],
    [fmt(w.les, 2) + " %", "LES wake loss"],
    [fmt(w.pywake_gaussian, 2) + " %", "PyWake Gaussian"],
    [fmt(c.median_mlups, 0), "MLUPS (farm + precursor)"],
    [fmt(c.median_wall_s / 60, 1) + " min", "GPU time per direction"],
  ];
  document.getElementById("kpis").innerHTML = items.map(([b, t]) => `<div class="kpi"><b>${b}</b><span>${t}</span></div>`).join("");
})();

// animation
(function () {
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const sel = document.getElementById("dir"), slider = document.getElementById("frame");
  const play = document.getElementById("play"), mode = document.getElementById("mode"), tlabel = document.getElementById("time");
  const lut = D.lut.map(h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  const bar = document.getElementById("bar").getContext("2d"), bimg = bar.createImageData(256, 1);
  lut.forEach((c, i) => bimg.data.set([c[0], c[1], c[2], 255], 4 * i));
  bar.putImageData(bimg, 0, 0);
  document.getElementById("vmin").textContent = D.vmin; document.getElementById("vmax").textContent = D.vmax;
  if (!D.anim.length) { cv.replaceWith(Object.assign(document.createElement("p"), {textContent: "No animation data."})); return; }
  D.anim.forEach((a, i) => sel.add(new Option("wind from " + a.wd + " deg", i)));
  let cur = null, frames = null, mean = null, k = 0, playing = true, showMean = false, off = null, img = null;
  async function load(i) {
    cur = D.anim[i];
    const [f, m] = await Promise.all([fetch(cur.file), fetch(cur.mean)].map(p => p.then(r => r.arrayBuffer())));
    frames = new Uint8Array(f); mean = new Uint8Array(m);
    off = document.createElement("canvas"); off.width = cur.nx; off.height = cur.ny;
    img = off.getContext("2d").createImageData(cur.nx, cur.ny);
    const scale = Math.min(1060, cur.nx * 4) / cur.nx;
    cv.width = Math.round(cur.nx * scale); cv.height = Math.round(cur.ny * scale);
    slider.max = cur.frames - 1; k = 0; draw();
  }
  function draw() {
    if (!cur) return;
    const n = cur.nx * cur.ny, src = showMean ? mean : frames.subarray(k * n, (k + 1) * n);
    for (let j = 0; j < cur.ny; j++) for (let i = 0; i < cur.nx; i++) {
      const v = src[j * cur.nx + i], c = lut[v], p = 4 * ((cur.ny - 1 - j) * cur.nx + i);
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
    const s = cv.width / cur.nx;
    ctx.strokeStyle = "#0b0b0b"; ctx.lineWidth = Math.max(2, s * 0.6); ctx.lineCap = "round";
    ctx.fillStyle = "#0b0b0b"; ctx.font = `${Math.max(11, s * 3)}px system-ui, sans-serif`;
    cur.turbines.forEach(([x, y], t) => {
      const px = (x + 0.5) * s, py = cv.height - (y + 0.5) * s, r = cur.radius * s;
      ctx.beginPath(); ctx.moveTo(px, py - r); ctx.lineTo(px, py + r); ctx.stroke();
      ctx.fillText("T" + (t + 1), px + 4, py - r - 3);
    });
    slider.value = k;
    tlabel.textContent = showMean ? "mean" : "t = " + Math.round(k * cur.dt_s) + " s";
  }
  sel.onchange = () => load(+sel.value);
  slider.oninput = () => { k = +slider.value; draw(); };
  play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
  mode.onclick = () => { showMean = !showMean; mode.textContent = showMean ? "Show instantaneous" : "Show 30-min mean"; draw(); };
  setInterval(() => { if (playing && !showMean && frames) { k = (k + 1) % cur.frames; draw(); } }, 120);
  load(0);
})();

// sector chart: grouped bars with hover tooltip and a table view
(function () {
  const rows = D.sectors; if (!rows.length) return;
  const src = [["scada", "SCADA", "--s1"], ["les", "LES (this work)", "--s2"], ["pywake_gaussian", "PyWake Gaussian", "--s3"],
               ["pywake_jensen", "PyWake Jensen", "--s4"], ["pywake_turbopark", "PyWake TurbOPark", "--s5"]].filter(s => s[0] in rows[0]);
  document.getElementById("legend").innerHTML = src.map(s => `<span><i style="background:var(${s[2]})"></i>${s[1]}</span>`).join("");
  const W = 1000, H = 340, m = {l: 44, r: 8, t: 10, b: 30};
  const top = (r, s) => s[0] === "scada" ? r.scada_hi : s[0] === "les" && r.les_se != null ? r.les + 2 * r.les_se : r[s[0]];
  const ymax = Math.ceil(Math.max(...rows.flatMap(r => src.map(s => top(r, s)))) / 5) * 5;
  const gw = (W - m.l - m.r) / rows.length, bw = gw * 0.8 / src.length;
  const y = v => m.t + (H - m.t - m.b) * (1 - v / ymax);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Wake loss per sector">`;
  for (let v = 0; v <= ymax; v += 5) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/>` +
    `<text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--ink2)">${v}%</text>`;
  rows.forEach((r, i) => {
    const x0 = m.l + i * gw + gw * 0.1;
    src.forEach((s, j) => {
      const v = r[s[0]], x = x0 + j * bw;
      g += `<rect data-i="${i}" x="${x + 1}" y="${y(Math.max(v, 0))}" width="${bw - 2}" height="${Math.abs(y(0) - y(v))}" rx="2" fill="var(${s[2]})"/>`;
      if (s[0] === "scada") g += `<line x1="${x + bw / 2}" x2="${x + bw / 2}" y1="${y(r.scada_lo)}" y2="${y(r.scada_hi)}" stroke="var(--ink)" stroke-width="1.4"/>`;
      if (s[0] === "les" && r.les_se != null) g += `<line x1="${x + bw / 2}" x2="${x + bw / 2}" y1="${y(v - 2 * r.les_se)}" y2="${y(v + 2 * r.les_se)}" stroke="var(--ink)" stroke-width="1.4"/>`;
    });
    g += `<rect data-i="${i}" x="${m.l + i * gw}" y="${m.t}" width="${gw}" height="${H - m.t - m.b}" fill="transparent"/>`;
    g += `<text x="${m.l + (i + 0.5) * gw}" y="${H - 10}" text-anchor="middle" font-size="11" fill="var(--ink2)">${r.sector_deg === "all" ? "all" : r.sector_deg + "°"}</text>`;
  });
  g += "</svg>";
  const box = document.getElementById("chart"), tip = document.getElementById("tip");
  box.insertAdjacentHTML("beforeend", g);
  box.addEventListener("mousemove", e => {
    const i = e.target.getAttribute && e.target.getAttribute("data-i");
    if (i == null) { tip.style.display = "none"; return; }
    const r = rows[+i], b = box.getBoundingClientRect();
    tip.innerHTML = `<b>${r.sector_deg === "all" ? "all sectors" : "sector " + r.sector_deg + "°"}</b> · ${r.records} records<br>` +
      src.map(s => `${s[1]}: ${fmt(r[s[0]], 1)} %` + (s[0] === "scada" ? ` [${fmt(r.scada_lo, 1)}-${fmt(r.scada_hi, 1)}]` : "") +
        (s[0] === "les" && r.les_se != null ? ` ± ${fmt(2 * r.les_se, 1)}` : "")).join("<br>");
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX - b.left + 12, b.width - tip.offsetWidth - 4) + "px";
    tip.style.top = (e.clientY - b.top + 12) + "px";
  });
  box.addEventListener("mouseleave", () => tip.style.display = "none");
  document.getElementById("table").innerHTML = "<table><tr><th>sector</th><th>records</th>" + src.map(s => `<th>${s[1]}</th>`).join("") + "</tr>" +
    rows.map(r => `<tr><td>${r.sector_deg}</td><td>${r.records}</td>` + src.map(s => `<td>${fmt(r[s[0]], 2)}` +
      (s[0] === "scada" ? ` [${fmt(r.scada_lo, 1)}-${fmt(r.scada_hi, 1)}]` : "") + "</td>").join("") + "</tr>").join("") + "</table>";
})();
</script>
</body>
</html>
"""
