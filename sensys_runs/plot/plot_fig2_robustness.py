#!/usr/bin/env python3
"""Fig. 2: EIA destabilizes energy via tail/spread/conv-rate, not mean shift; REFiNE+EIA stable."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

RESULTS_ROOT = "lio/results"
OUT_PATH = "fig/eval_robustness.pdf"
N_AGENTS = 4
OFFSET = 0.18

# Destabilization = conv-rate drop + across-seed spread + high-energy tail — NOT mean/median shift.

LEVELS = [
    (
        "none",
        "er_lio_4_2",
        "sweep_er_refine_4_2_s*",
        "LIO",
        "REFiNE",
    ),
    (
        "w(1.1,0.9)",
        "er_eia_4_2_w1.1-0.9",
        "sweep_er_refine_eia_4_2_w1.1-0.9_s*",
        "EIA",
        "REFiNE_EIA",
    ),
    (
        "w(1.5,1.5)",
        "er_eia_4_2_w1.5-1.5",
        "sweep_er_refine_eia_4_2_w1.5-1.5_s*",
        "EIA",
        "REFiNE_EIA",
    ),
    (
        "w(2.0,0.2)",
        "er_eia_4_2_w2.0-0.2",
        "sweep_er_refine_eia_4_2_s*",
        "EIA",
        "REFiNE_EIA",
    ),
]


def _cell_style(method_key: str) -> dict:
    st = sty.METHOD_STYLE[method_key]
    return dict(fill=st["fill"], line=st["line"], hatch=st["hatch"])


def _draw_cell(
    ax,
    x: float,
    en: np.ndarray,
    method_key: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    st = _cell_style(method_key)
    en = np.asarray(en, dtype=float)
    med = float(np.nanmedian(en))
    sd = float(np.nanstd(en, ddof=1)) if np.sum(~np.isnan(en)) > 1 else 0.0
    finite = en[~np.isnan(en)]

    if len(finite) and np.ptp(finite) >= 1.0:
        parts = ax.violinplot(
            [en],
            positions=[x],
            widths=0.32,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(st["fill"])
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(0.30)
            if st["hatch"]:
                body.set_hatch(st["hatch"])

    jitter = rng.uniform(-0.06, 0.06, size=len(en))
    ax.scatter(
        x + jitter,
        en,
        s=16,
        c="black",
        alpha=0.8,
        linewidths=0,
        zorder=4,
    )
    ax.scatter(
        [x],
        [med],
        s=40,
        marker="D",
        facecolors=st["fill"],
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    return med, sd


def main() -> None:
    rng = np.random.default_rng(0)
    fig, axes = sty.new_fig(1, 1, width=sty.FULL_W, panel_h=2.6)
    ax = axes[0]

    lio_none = sty.load_runs(RESULTS_ROOT, "er_lio_4_2", N_AGENTS)
    lio_none_med = lio_none["en_s"]["median"]
    lio_line = sty.METHOD_STYLE["LIO"]["line"]

    all_en: list[float] = []
    cell_meta: list[tuple[str, str, int, int, float, float]] = []

    print("=== eval_robustness — per (level, side) ===")
    for gi, (label, lio_glob, ref_glob, lio_m, ref_m) in enumerate(LEVELS):
        xc = float(gi)
        for side, glob_pat, mkey, x_off in (
            ("LIO", lio_glob, lio_m, -OFFSET),
            ("REFiNE", ref_glob, ref_m, OFFSET),
        ):
            r = sty.load_runs(RESULTS_ROOT, glob_pat, N_AGENTS)
            en = r["en"]
            all_en.extend(en[~np.isnan(en)].tolist())
            med, sd = _draw_cell(ax, xc + x_off, en, mkey, rng)
            n_conv = r["n_conv"]
            n = r["n"]
            cell_meta.append((label, side, n, n_conv, med, sd))
            print(
                f"  {label:12s} {side:6s}  n={n:2d}  n_conv={n_conv:2d}  "
                f"median={med:7.2f}  std={sd:6.2f}"
            )

    y_min = 0.9 * min(all_en)
    y_max = 1.05 * max(all_en)
    ax.set_ylim(y_min, y_max)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for label, side, n, n_conv, med, sd in cell_meta:
        gi = next(i for i, (lb, *_) in enumerate(LEVELS) if lb == label)
        x_off = -OFFSET if side == "LIO" else OFFSET
        cell_x = float(gi) + x_off
        conv_color = "#E15759" if n_conv < n else "#10B981"
        ax.text(
            cell_x,
            -0.12,
            f"{n_conv}/{n}",
            fontsize=6.5,
            fontweight="bold",
            color=conv_color,
            ha="center",
            va="top",
            transform=trans,
            clip_on=False,
        )
        ax.text(
            cell_x,
            -0.185,
            f"sd={sd:.0f}",
            fontsize=6,
            color="#333333",
            ha="center",
            va="top",
            transform=trans,
            clip_on=False,
        )

    ax.axhline(
        lio_none_med,
        linestyle="--",
        color=lio_line,
        linewidth=0.8,
        alpha=0.85,
        zorder=1,
    )

    ax.set_xticks(np.arange(len(LEVELS), dtype=float))
    ax.set_xticklabels([lv[0] for lv in LEVELS])
    ax.set_ylabel("Energy consumed")
    ax.set_title("Attack-weight robustness @ ER(4,2)", fontweight="bold")

    sty.grid(ax, "y")

    med_handle = mlines.Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=6,
        markerfacecolor="0.75",
        markeredgecolor="black",
        label="median",
    )
    dot_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=4,
        markerfacecolor="black",
        alpha=0.8,
        label="per-seed",
    )
    viol_patch = mpatches.Patch(
        facecolor="0.85",
        edgecolor="black",
        hatch="//",
        alpha=0.35,
        label="seed distribution",
    )
    lio_patch = mpatches.Patch(
        facecolor=sty.METHOD_STYLE["LIO"]["fill"],
        edgecolor="black",
        label="LIO",
    )
    eia_patch = mpatches.Patch(
        facecolor=sty.METHOD_STYLE["EIA"]["fill"],
        edgecolor="black",
        hatch=sty.METHOD_STYLE["EIA"]["hatch"],
        label="LIO+EIA",
    )
    ref_patch = mpatches.Patch(
        facecolor=sty.METHOD_STYLE["REFiNE"]["fill"],
        edgecolor="black",
        hatch=sty.METHOD_STYLE["REFiNE"]["hatch"],
        label="REFiNE",
    )
    ref_eia_patch = mpatches.Patch(
        facecolor=sty.METHOD_STYLE["REFiNE_EIA"]["fill"],
        edgecolor="black",
        hatch=sty.METHOD_STYLE["REFiNE_EIA"]["hatch"],
        label="REFiNE+EIA",
    )
    sty.glyph_legend(
        ax,
        [
            (med_handle, "median"),
            (dot_handle, "per-seed"),
            (viol_patch, "seed distribution"),
            (lio_patch, "LIO"),
            (eia_patch, "LIO+EIA"),
            (ref_patch, "REFiNE"),
            (ref_eia_patch, "REFiNE+EIA"),
        ],
        loc="upper left",
    )

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
