#!/usr/bin/env python3
"""Fig. 1: REFiNE energy reduction and task-preserved convergence (2×2)."""

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
OUT_PATH = "fig/eval_energy.pdf"

LIO_42 = "er_lio_4_2"  # (4) expect en med ~101.6, env ~43.8
REF_42 = "sweep_er_refine_4_2_s*"  # (4) no-EIA; en med 92.3, env 45.0, 9/10 conv (seed5 diverges)
LIO_31 = "er_lio_3_1"  # (3)
REF_31 = "sweep_er_refine_3_1_s*"  # (3) expect 74.7 / 63.3, 10/10

# Deterministic REFiNE seeds overlap into one faint line — do NOT jitter.


def _print_group(name: str, r: dict) -> None:
    print(
        f"{name:14s}  n={r['n']:2d}  n_conv={r['n_conv']:2d}  "
        f"div={r['diverged']}  en_med={r['en_s']['median']:7.2f}  env_med={r['env_s']['median']:7.2f}"
    )


def _plot_convergence(
    ax,
    lio_glob: str,
    ref_glob: str,
    n_agents: int,
    metric: str,
    ylabel: str,
    title: str,
    *,
    annotate_gap: bool,
) -> None:
    lio_tr = sty.load_traj(RESULTS_ROOT, lio_glob, n_agents, metric)
    ref_tr = sty.load_traj(RESULTS_ROOT, ref_glob, n_agents, metric)

    lio_st = sty.METHOD_STYLE["LIO"]
    ref_st = sty.METHOD_STYLE["REFiNE"]

    for ep, val in lio_tr:
        ax.plot(ep, val, color=lio_st["line"], lw=0.5, alpha=0.22)
    for ep, val in ref_tr:
        ax.plot(ep, val, color=ref_st["line"], lw=0.5, alpha=0.22)

    ep_l, med_l, _ = sty.median_band(lio_tr)
    ep_r, med_r, _ = sty.median_band(ref_tr)

    ax.plot(ep_l, med_l, color=lio_st["line"], lw=1.8, label="LIO")
    ax.plot(ep_r, med_r, color=ref_st["line"], lw=1.8, label="REFiNE")

    ax.axhline(
        med_l[-1],
        linestyle="--",
        color=lio_st["line"],
        linewidth=0.8,
        alpha=0.85,
        zorder=2,
    )

    if annotate_gap:
        r_lio = sty.load_runs(RESULTS_ROOT, lio_glob, n_agents)
        r_ref = sty.load_runs(RESULTS_ROOT, ref_glob, n_agents)
        y_l = float(r_lio["en_s"]["median"])
        y_r = float(r_ref["en_s"]["median"])
        pct_text = sty.fmt_pct(y_r, y_l)

        x_last = float(ep_l[-1])
        xspan = float(ep_l[-1] - ep_l[0]) if len(ep_l) > 1 else 1.0
        x_off = x_last + 0.04 * xspan
        ax.plot(
            [x_off, x_off],
            [y_l, y_r],
            color="0.25",
            linewidth=0.7,
            clip_on=False,
            zorder=4,
        )
        cap = 0.01 * xspan
        for y in (y_l, y_r):
            ax.plot(
                [x_off - cap, x_off + cap],
                [y, y],
                color="0.25",
                linewidth=0.7,
                clip_on=False,
                zorder=4,
            )
        ax.text(
            x_off + 0.02 * xspan,
            0.5 * (y_l + y_r),
            pct_text,
            fontsize=7,
            va="center",
            ha="left",
            clip_on=False,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    sty.grid(ax, "y")


def main() -> None:
    groups = [
        ("LIO_42", LIO_42, 4),
        ("REF_42", REF_42, 4),
        ("LIO_31", LIO_31, 3),
        ("REF_31", REF_31, 3),
    ]
    print("=== eval_energy — per-group medians ===")
    for name, glob_pat, na in groups:
        _print_group(name, sty.load_runs(RESULTS_ROOT, glob_pat, na))

    fig, axes = sty.new_fig(nrows=2, ncols=2, width=sty.FULL_W, panel_h=2.0)
    ax_a, ax_b, ax_c, ax_d = axes

    _plot_convergence(
        ax_a,
        LIO_42,
        REF_42,
        4,
        "energy",
        "Energy consumed",
        "(a) Energy consumed @ ER(4,2)",
        annotate_gap=True,
    )
    _plot_convergence(
        ax_b,
        LIO_31,
        REF_31,
        3,
        "energy",
        "Energy consumed",
        "(b) Energy consumed @ ER(3,1)",
        annotate_gap=True,
    )
    _plot_convergence(
        ax_c,
        LIO_42,
        REF_42,
        4,
        "reward",
        "Env. reward",
        "(c) Reward convergence @ ER(4,2)",
        annotate_gap=False,
    )
    _plot_convergence(
        ax_d,
        LIO_31,
        REF_31,
        3,
        "reward",
        "Env. reward",
        "(d) Reward convergence @ ER(3,1)",
        annotate_gap=False,
    )

    seed_proxy = mlines.Line2D(
        [],
        [],
        color="0.55",
        linewidth=0.5,
        alpha=0.5,
        label="per-seed",
    )
    handles, labels = ax_a.get_legend_handles_labels()
    sty.glyph_legend(
        ax_a,
        [(handles[0], labels[0]), (handles[1], labels[1]), (seed_proxy, "per-seed")],
    )

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
