#!/usr/bin/env python3
"""No-EIA F_T ablation @ ER(4,2): energy, sigma_agent, task, per-seed pairing."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

RESULTS_ROOT = "lio/results"
OUT_PATH = "fig/eval_ablation.pdf"
N_AGENTS = 4

LIO_GLOB = "er_lio_4_2"
ABL_AMP = "probe_er_4_2_B0.2_beta0_s*"  # amp-only (beta=0, B=0.2)
ABL_B0 = "probe_er_4_2_B0_beta1.0_s*"  # energy-only (beta=1, B=0)
FULL = "sweep_er_refine_4_2_s*"  # FULL (beta=1, B=0.2)

CONDITIONS = [
    ("LIO", LIO_GLOB, "LIO"),
    ("amp-only", ABL_AMP, "REFiNE"),
    ("energy-only", ABL_B0, "REFiNE"),
    ("FULL", FULL, "REFiNE"),
]

YTICK_LABELS = ["LIO", r"$\mathcal{B}$", r"$\beta$", "FULL"]
TITLE_FS = 7.5
TICK_FS = 6.5


def _run_seed(path: str) -> int | None:
    m = re.search(r"_s(\d+)/log\.csv$", path) or re.search(r"nano_er(\d+)/", path)
    return int(m.group(1)) if m else None


def _load_all() -> list[dict]:
    return [sty.load_runs_metrics(RESULTS_ROOT, glob_pat, N_AGENTS) for _, glob_pat, _ in CONDITIONS]


def _print_table(runs: list[tuple[str, dict]]) -> None:
    print("\n=== eval_ablation — median [IQR] per condition ===")
    for label, r in runs:
        es, ss, ts = r["energy_s"], r["sigma_s"], r["task_s"]
        print(
            f"  {label:12s}  n={r['n']:2d}  "
            f"energy={es['median']:6.2f} [{es['q1']:6.2f},{es['q3']:6.2f}]  "
            f"sigma={ss['median']:6.2f} [{ss['q1']:6.2f},{ss['q3']:6.2f}]  "
            f"task={ts['median']:6.2f} [{ts['q1']:6.2f},{ts['q3']:6.2f}]"
        )


def _bar_style(method_key: str) -> tuple[str, str]:
    st = sty.METHOD_STYLE[method_key]
    return st["fill"], st["hatch"] if method_key == "REFiNE" else ""


def _label_bars(ax, container, vals: list[float]) -> None:
    labels = ax.bar_label(
        container,
        labels=[f"{v:.1f}" for v in vals],
        label_type="center",
        rotation=0,
        fontsize=6.5,
        fontweight="bold",
        color="black",
    )
    for t in labels:
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])


def _draw_iqr_bars(
    ax,
    stats: list[dict],
    fills: list[str],
    hatches: list[str],
    *,
    xlabel: str,
    title: str,
) -> None:
    height = 0.65
    ys = np.arange(len(stats))
    medians = [st["median"] for st in stats]
    xerr_left = [st["median"] - st["q1"] for st in stats]
    xerr_right = [st["q3"] - st["median"] for st in stats]
    container = ax.barh(
        ys,
        medians,
        height=height,
        color=fills,
        hatch=hatches,
        edgecolor="black",
        linewidth=0.6,
        xerr=[xerr_left, xerr_right],
        error_kw=dict(lw=0.8, capsize=2, color="0.15"),
        zorder=2,
    )
    _label_bars(ax, container, medians)
    xmax = 1.15 * max(st["q3"] for st in stats)
    ax.set_yticks(ys)
    ax.set_yticklabels(YTICK_LABELS, fontsize=TICK_FS)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=TITLE_FS)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    sty.grid(ax, "x")


def _bar_panel_stats(runs: list[tuple[str, dict]], key: str) -> tuple[list[str], list[str], list[dict]]:
    fills, hatches, stats = [], [], []
    for (_, r), (_, _, mk) in zip(runs, CONDITIONS):
        fill, hatch = _bar_style(mk)
        fills.append(fill)
        hatches.append(hatch)
        stats.append(r[f"{key}_s"])
    return fills, hatches, stats


def _panel_energy(ax, runs: list[tuple[str, dict]]) -> None:
    fills, hatches, stats = _bar_panel_stats(runs, "energy")
    _draw_iqr_bars(ax, stats, fills, hatches, xlabel="Energy", title="(A) Energy")


def _panel_sigma(ax, runs: list[tuple[str, dict]]) -> None:
    fills, hatches, stats = _bar_panel_stats(runs, "sigma")
    _draw_iqr_bars(
        ax,
        stats,
        fills,
        hatches,
        xlabel=r"$\sigma_{\mathrm{agent}}$",
        title="(B) Reward fairness",
    )


def _panel_task(ax, runs: list[tuple[str, dict]]) -> None:
    fills, hatches, stats = _bar_panel_stats(runs, "task")
    _draw_iqr_bars(
        ax,
        stats,
        fills,
        hatches,
        xlabel="Task return",
        title="(C) Task return",
    )


def _is_collapse(e_only: float, full: float) -> bool:
    return full > 1.5 * e_only or (full - e_only) > 50


def _paired_by_seed(r_energy: dict, r_full: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    e_map = {_run_seed(p): v for p, v in zip(r_energy["paths"], r_energy["energy"])}
    f_map = {_run_seed(p): v for p, v in zip(r_full["paths"], r_full["energy"])}
    seeds = sorted(set(e_map) & set(f_map) - {None})
    x = np.array([e_map[s] for s in seeds])
    y = np.array([f_map[s] for s in seeds])
    return x, y, seeds


def _panel_paired(ax, runs: list[tuple[str, dict]]) -> None:
    r_e = runs[2][1]
    r_f = runs[3][1]
    x, y, seeds = _paired_by_seed(r_e, r_f)

    collapse = [_is_collapse(xi, yi) for xi, yi in zip(x, y)]
    inert_idx = [i for i, c in enumerate(collapse) if not c]
    collapse_idx = [i for i, c in enumerate(collapse) if c]

    lo = min(float(np.min(x)), float(np.min(y))) * 0.92
    hi = max(float(np.max(x)), float(np.max(y))) * 1.05
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=0.8, zorder=1)
    ax.text(
        hi * 0.62,
        hi * 0.68,
        r"$y\,{=}\,x$",
        fontsize=6,
        color="0.45",
        ha="left",
        va="bottom",
        clip_on=True,
    )
    ax.scatter(
        x,
        y,
        c=sty.METHOD_STYLE["REFiNE"]["line"],
        s=22,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.85,
        zorder=2,
    )

    x_off = 0.04 * (hi - lo)
    if inert_idx:
        cx = float(np.mean(x[inert_idx]))
        cy = float(np.mean(y[inert_idx]))
        ax.text(
            cx + x_off,
            cy,
            "9 seeds",
            fontsize=6,
            ha="left",
            va="center",
            clip_on=True,
        )

    if collapse_idx:
        i = collapse_idx[0]
        ax.text(
            x[i] + x_off,
            y[i],
            "seed 5",
            fontsize=6,
            ha="left",
            va="center",
            clip_on=True,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("energy-only", fontsize=TICK_FS)
    ax.set_ylabel("FULL", fontsize=TICK_FS)
    ax.set_title("(D) Per-seed", loc="left", fontweight="bold", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_aspect("equal", adjustable="box")
    sty.grid(ax, "both")

    if collapse_idx:
        print("\n=== Panel (D) collapse seeds ===")
        for i in collapse_idx:
            print(
                f"  seed {seeds[i]}: energy-only={x[i]:.1f}, FULL={y[i]:.1f}, "
                f"ratio={y[i]/x[i]:.2f}, gap={y[i]-x[i]:.1f}"
            )


def main() -> None:
    loaded = _load_all()
    runs = [(label, data) for (label, _, _), data in zip(CONDITIONS, loaded)]
    _print_table(runs)

    fig, axes = sty.new_fig(2, 2, width=sty.COL_W, panel_h=1.6)
    ax_a, ax_b, ax_c, ax_d = axes

    _panel_energy(ax_a, runs)
    _panel_sigma(ax_b, runs)
    _panel_task(ax_c, runs)
    _panel_paired(ax_d, runs)

    e_only = runs[2][1]["energy_s"]["median"]
    full_e = runs[3][1]["energy_s"]["median"]
    e_only_s = runs[2][1]["sigma_s"]["median"]
    full_s = runs[3][1]["sigma_s"]["median"]
    print(
        f"\n=== F_T inertness check (energy-only vs FULL) ===\n"
        f"  median energy:      {e_only:.4f} vs {full_e:.4f}  "
        f"({'IDENTICAL' if abs(e_only - full_e) < 1e-6 else 'DIFFER'})"
    )
    print(
        f"  median sigma_agent: {e_only_s:.4f} vs {full_s:.4f}  "
        f"({'IDENTICAL' if abs(e_only_s - full_s) < 1e-6 else 'DIFFER'})"
    )

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
