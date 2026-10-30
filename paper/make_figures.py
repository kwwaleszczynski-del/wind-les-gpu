"""Figures for the paper: the repository figures drawn at 3/4 size (so the text stays readable at print width),
without the titles the captions repeat, saved as PDF. Run from the repository root after
`python -m wind_les compare`; the hub-plane figure also needs the cached runs in results/cache.

Usage:
    python paper/make_figures.py                 # paper/figures/*.pdf
    python paper/make_figures.py --png preview   # the same as PNG, to look at
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from wind_les import plots

SCALE = 0.75
SHORT_TITLES = {"Bare kernel throughput (label: population memory)": "Bare kernel throughput",
                "Cost of one direction (10 min spin-up + 30 min average)": "Cost of one wind direction"}


def _subplots(*args, figsize=None, **kwargs):
    if figsize is not None:
        figsize = (figsize[0] * SCALE, figsize[1] * SCALE)
    return _plain_subplots(*args, figsize=figsize, **kwargs)


def _tidy(fig, name: str) -> None:
    """Paper layout: the caption carries the title; shorter labels where the smaller figure gets crowded."""
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    for ax in fig.axes:
        title = ax.get_title()
        if len(fig.axes) == 1 or name in ("efficiency_direction", "sector_losses"):
            ax.set_title("")
        elif title in SHORT_TITLES:
            ax.set_title(SHORT_TITLES[title])
        if ax.get_ylabel() == "power / free-stream reference":
            ax.set_ylabel("power / reference")
    if name == "sector_losses":
        ax = fig.axes[0]
        ax.set_xticks(ax.get_xticks(), [t.get_text().replace(" deg", "°") for t in ax.get_xticklabels()])
        leg = ax.get_legend()
        ax.legend(leg.legend_handles, [t.get_text() for t in leg.get_texts()], ncol=3, loc="upper left")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    if name == "verification":
        fig.axes[1].legend(ncol=2, fontsize=7, loc="upper left")
        fig.axes[2].legend(loc="lower right", fontsize=8)
    if name in ("turbine_efficiency", "efficiency_direction"):
        leg = next(ax.get_legend() for ax in fig.axes if ax.get_legend() is not None)
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        leg.remove()
        fig.legend(handles, labels, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.0))
    for ax in fig.axes:          # the repository labels say GB for binary gigabytes; the paper says GiB
        for t in ax.texts:
            t.set_text(t.get_text().replace(" GB", " GiB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    parser.add_argument("--png", type=Path, help="write PNG previews to this folder instead of PDFs")
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def save(fig, path: Path) -> None:
        _tidy(fig, path.stem)
        target = (a.png / path.stem).with_suffix(".png") if a.png else (a.out / path.stem).with_suffix(".pdf")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110 if a.png else 300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logging.info("figure %s", target)

    global _plain_subplots
    _plain_subplots = plt.subplots
    plt.subplots = _subplots
    plots._save = save
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    plots.all_figures(a.results, a.out)


if __name__ == "__main__":
    main()
