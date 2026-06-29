#!/usr/bin/env python3
"""Fig. 4: REFiNE compute overhead vs physical step cadence (Orin Nano constants, not from logs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

OUT_PATH = "fig/eval_timing.pdf"

# Hardware-benchmark constants (µs → ms where noted).
MS_FAIRNESS = 0.4
MS_INF_LIO = 0.815
MS_INF_REFINE = 0.854
MS_TRAIN_EP = 25.8
MS_PHYSICAL_MED = 13100.0
MS_PHYSICAL_P99 = 15700.0  # caption reference only

_BAR_BBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.4)

ROWS = [
    ("fairness/penalty compute", MS_FAIRNESS, "neutral"),
    ("inference (LIO)", MS_INF_LIO, "LIO"),
    ("inference (REFiNE)", MS_INF_REFINE, "REFiNE"),
    ("training overhead/episode", MS_TRAIN_EP, "neutral"),
    ("physical step (median)", MS_PHYSICAL_MED, "physical"),
]


def _bar_kwargs(kind: str) -> dict:
    if kind == "LIO":
        st = sty.METHOD_STYLE["LIO"]
        return dict(
            facecolor=st["fill"],
            hatch=st["hatch"],
            edgecolor="black",
            linewidth=0.6,
        )
    if kind == "REFiNE":
        st = sty.METHOD_STYLE["REFiNE"]
        return dict(
            facecolor=st["fill"],
            hatch=st["hatch"],
            edgecolor="black",
            linewidth=0.6,
        )
    if kind == "physical":
        return dict(facecolor="#2C3E50", edgecolor="black", linewidth=0.6)
    return dict(facecolor="#D0D0D0", edgecolor="black", linewidth=0.6)


def main() -> None:
    fig, axes = sty.new_fig(1, 1, width=sty.COL_W, panel_h=2.2)
    ax = axes[0]

    y_pos = np.arange(len(ROWS) - 1, -1, -1, dtype=float)
    labels = [r[0] for r in ROWS]
    values = [r[1] for r in ROWS]

    for y, (label, val, kind) in zip(y_pos, ROWS):
        ax.barh(y, val, height=0.62, **_bar_kwargs(kind))
        if val < 10:
            txt = f"{val:.3f} ms" if val < 1 else f"{val:.2f} ms"
        else:
            txt = f"{val:.0f} ms"
        if kind == "physical":
            ax.text(
                val * 0.97,
                y,
                txt,
                va="center",
                ha="right",
                fontsize=6.5,
                color="black",
                bbox=_BAR_BBOX,
                zorder=6,
            )
        else:
            ax.text(
                val * 1.08,
                y,
                txt,
                va="center",
                ha="left",
                fontsize=6.5,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Time (ms, log scale)")
    ax.set_xlim(0.2, MS_PHYSICAL_MED * 2.5)
    sty.grid(ax, "x")

    fig.suptitle("Compute vs physical cadence (Orin Nano)", fontsize=8)

    x_arrow = 500.0
    y_arrow_lo, y_arrow_hi = 0.15, 2.85
    ax.annotate(
        "",
        xy=(x_arrow, y_arrow_lo),
        xytext=(x_arrow, y_arrow_hi),
        arrowprops=dict(arrowstyle="<->", color="#E15759", lw=0.85),
        zorder=4,
    )
    ax.text(
        900.0,
        y_arrow_hi + 0.22,
        "~4 orders of magnitude",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="#E15759",
        zorder=5,
        clip_on=False,
    )

    sty.save(fig, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    x_inf = max(MS_INF_LIO, MS_INF_REFINE)
    print(
        f"  inference REFiNE +{100*(MS_INF_REFINE-MS_INF_LIO)/MS_INF_LIO:.1f}% vs LIO; "
        f"log10(physical/inference)≈{np.log10(MS_PHYSICAL_MED/x_inf):.1f}"
    )


if __name__ == "__main__":
    main()
