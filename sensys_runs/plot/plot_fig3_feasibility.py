#!/usr/bin/env python3
"""Fig. 3: Quorum feasibility window, ER(6,4) infeasible, IPD limitation (honest scope)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

RESULTS_ROOT = "lio/results"
OUT_PATH = "fig/eval_feasibility.pdf"
X_OFF = 0.12

ER_SIZES = [
    ("ER(3,1)", "er_lio_3_1", "sweep_er_refine_3_1_s*", 3),
    ("ER(4,2)", "er_lio_4_2", "sweep_er_refine_4_2_s*", 4),
    ("ER(6,4)", "er_lio_6_4", "sweep_er_refine_6_4_s*", 6),
]

IPD_GROUPS = [
    ("LIO", "ipd_lio", "LIO"),
    ("EIA", "ipd_eia", "EIA"),
    ("REFiNE+EIA", "sweep_ipd_refine_eia_B0.1_s*", "REFiNE_EIA"),
]


def _print_env(name: str, r: dict) -> None:
    print(f"  {name:16s}  env_med={r['env_s']['median']:7.2f}  lo={r['env_s']['lo']:7.2f}  hi={r['env_s']['hi']:7.2f}")


def _plot_feasibility(ax) -> None:
    x_centers = np.arange(len(ER_SIZES), dtype=float)

    for xi, (label, lio_g, ref_g, na) in enumerate(ER_SIZES):
        r_lio = sty.load_runs(RESULTS_ROOT, lio_g, na)
        r_ref = sty.load_runs(RESULTS_ROOT, ref_g, na)

        for x_off, r, mkey in (
            (-X_OFF, r_lio, "LIO"),
            (X_OFF, r_ref, "REFiNE"),
        ):
            x = float(xi) + x_off
            st = sty.METHOD_STYLE[mkey]
            med = r["env_s"]["median"]
            lo = r["env_s"]["lo"]
            hi = r["env_s"]["hi"]
            ax.vlines(
                x,
                lo,
                hi,
                colors=st["line"],
                linewidth=0.9,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                x,
                med,
                s=36,
                marker="o",
                facecolors=st["fill"],
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )

    y_all = []
    for _, lio_g, ref_g, na in ER_SIZES:
        for g in (lio_g, ref_g):
            r = sty.load_runs(RESULTS_ROOT, g, na)
            y_all.extend(r["env"][~np.isnan(r["env"])].tolist())

    y_min = min(y_all) - 5.0
    y_max = max(y_all) + 5.0
    ax.set_ylim(y_min, y_max)

    ax.axhspan(y_min, 0.0, facecolor="0.92", edgecolor="none", zorder=0)
    ax.axhline(0.0, color="0.35", linewidth=0.7, zorder=1)

    lio_h = mlines.Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor=sty.METHOD_STYLE["LIO"]["line"],
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="LIO",
    )
    ref_h = mlines.Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor=sty.METHOD_STYLE["REFiNE"]["line"],
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="REFiNE",
    )
    ax.legend(
        handles=[lio_h, ref_h],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([s[0] for s in ER_SIZES])
    ax.set_ylabel("Median env. reward")
    ax.set_title(
        "(a) Task feasibility vs quorum size",
        loc="left",
        fontweight="bold",
    )
    sty.grid(ax, "y")


def _plot_ipd(ax) -> None:
    for label, glob_pat, mkey in IPD_GROUPS:
        tr = sty.load_traj(RESULTS_ROOT, glob_pat, 2, "reward")
        st = sty.METHOD_STYLE[mkey]
        for ep, val in tr:
            ax.plot(ep, val, color=st["line"], lw=0.5, alpha=0.22)
        ep_m, med_m, _ = sty.median_band(tr)
        ax.plot(ep_m, med_m, color=st["line"], lw=1.8, label=label)

    handles, labels_ax = ax.get_legend_handles_labels()
    sty.glyph_legend(ax, list(zip(handles, labels_ax)), loc="lower left")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Env. reward")
    ax.set_title(
        "(b) IPD limitation (env-reward convergence)",
        loc="left",
        fontweight="bold",
    )
    sty.grid(ax, "y")


def main() -> None:
    print("=== eval_feasibility — env medians ===")
    for label, lio_g, ref_g, na in ER_SIZES:
        r_l = sty.load_runs(RESULTS_ROOT, lio_g, na)
        r_r = sty.load_runs(RESULTS_ROOT, ref_g, na)
        _print_env(f"{label} LIO", r_l)
        _print_env(f"{label} REFiNE", r_r)
    for label, glob_pat, _ in IPD_GROUPS:
        _print_env(label, sty.load_runs(RESULTS_ROOT, glob_pat, 2))

    fig, axes = sty.new_fig(1, 2, width=sty.FULL_W, panel_h=2.2)
    _plot_feasibility(axes[0])
    _plot_ipd(axes[1])

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
