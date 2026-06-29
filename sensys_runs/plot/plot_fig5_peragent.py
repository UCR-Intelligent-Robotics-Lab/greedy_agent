#!/usr/bin/env python3
"""Fig. 5: Per-agent energy + env reward (one representative seed/col, ER 4,2, mechanism view)."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

RESULTS_ROOT = "lio/results"
OUT_PATH = "fig/eval_peragent.pdf"
N_AGENTS = 4
TAIL = 5
A2_COLOR = "#E15759"

# col: (energy title, reward title, glob, A2* in attack columns only)
COLUMNS = [
    ("(a) LIO", "(d) LIO", "er_lio_4_2", False),
    ("(b) LIO+EIA", "(e) LIO+EIA", "er_eia_4_2_w2.0-0.2", True),
    ("(c) REFiNE+EIA", "(f) REFiNE+EIA", "sweep_er_refine_eia_4_2_s*", True),
]


def representative_run(
    root: str, dir_glob: str, n_agents: int = N_AGENTS, tail: int = TAIL
) -> str:
    paths = sorted(glob.glob(f"{root}/*/{dir_glob}/log.csv"))
    if not paths:
        raise FileNotFoundError(f"No match: {root}/*/{dir_glob}/log.csv")
    ens = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        t = df.tail(tail)
        ens.append(
            float(
                np.mean(
                    [t[f"A{i}_total_energy"].mean() for i in range(1, n_agents + 1)]
                )
            )
        )
    med = np.median(ens)
    idx = int(np.argmin(np.abs(np.array(ens) - med)))
    return paths[idx]


def agent_energy(
    csv: str, n_agents: int = N_AGENTS
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    return df["episode"].values, {
        i: df[f"A{i}_total_energy"].values for i in range(1, n_agents + 1)
    }


def agent_reward(
    csv: str, n_agents: int = N_AGENTS
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    return df["episode"].values, {
        i: df[f"A{i}_reward_env"].values for i in range(1, n_agents + 1)
    }


def _episode_markevery(ep: np.ndarray) -> int:
    if len(ep) < 2:
        return 1
    med_step = float(np.median(np.diff(ep.astype(float))))
    if med_step <= 0:
        return 1
    return max(1, int(round(500 / med_step)))


def _plot_agents(
    ax,
    ep: np.ndarray,
    vals: dict[int, np.ndarray],
    *,
    highlight_a2: bool,
    markevery: int,
) -> list[float]:
    y_used: list[float] = []
    for aid in range(1, N_AGENTS + 1):
        y = np.asarray(vals[aid], dtype=float)
        y_used.extend(y.tolist())
        if highlight_a2 and aid == 2:
            ax.plot(
                ep,
                y,
                color=A2_COLOR,
                lw=1.7,
                alpha=0.9,
                marker="*",
                markevery=markevery,
                markersize=7,
            )
        else:
            ax.plot(
                ep,
                y,
                color=sty.AGENT_COLORS[aid],
                lw=1.0,
                alpha=0.9,
                marker="o",
                markevery=markevery,
                markersize=2.5,
            )
    return y_used


def _legend_entries(highlight_a2: bool) -> list[tuple]:
    entries = []
    for aid in range(1, N_AGENTS + 1):
        if aid == 2 and highlight_a2:
            h = mlines.Line2D(
                [],
                [],
                color=A2_COLOR,
                lw=1.7,
                marker="*",
                markersize=5,
                markevery=1,
                label="A2*",
            )
        elif aid == 2:
            h = mlines.Line2D(
                [],
                [],
                color=sty.AGENT_COLORS[2],
                lw=1.0,
                marker="o",
                markersize=3,
                label="A2",
            )
        else:
            h = mlines.Line2D(
                [],
                [],
                color=sty.AGENT_COLORS[aid],
                lw=1.0,
                marker="o",
                markersize=3,
                label=f"A{aid}",
            )
        entries.append((h, h.get_label()))
    return entries


def main() -> None:
    fig, axes = sty.new_fig(2, 3, width=sty.FULL_W, panel_h=2.2)
    axes = axes.reshape(2, 3)

    energy_y: list[float] = []
    reward_y: list[float] = []

    print("=== eval_peragent — representative runs ===")
    for j, (title_en, title_rw, glob_pat, highlight_a2) in enumerate(COLUMNS):
        rep = representative_run(RESULTS_ROOT, glob_pat, N_AGENTS, TAIL)
        rundir = os.path.basename(os.path.dirname(rep))
        parent = os.path.basename(os.path.dirname(os.path.dirname(rep)))
        print(f"  {title_en:16s}  {parent}/{rundir}")

        ep_en, vals_en = agent_energy(rep, N_AGENTS)
        ep_rw, vals_rw = agent_reward(rep, N_AGENTS)
        markevery = _episode_markevery(ep_en)

        ax_en = axes[0, j]
        ax_rw = axes[1, j]
        energy_y.extend(
            _plot_agents(ax_en, ep_en, vals_en, highlight_a2=highlight_a2, markevery=markevery)
        )
        reward_y.extend(
            _plot_agents(ax_rw, ep_rw, vals_rw, highlight_a2=highlight_a2, markevery=markevery)
        )

        ax_en.set_title(title_en, loc="left", fontweight="bold")
        ax_rw.set_title(title_rw, loc="left", fontweight="bold")
        sty.grid(ax_en, "y")
        sty.grid(ax_rw, "y")
        sty.glyph_legend(ax_en, _legend_entries(highlight_a2), loc="upper right")

    e_lo, e_hi = float(np.min(energy_y)), float(np.max(energy_y))
    e_pad = 0.04 * (e_hi - e_lo) if e_hi > e_lo else 1.0
    for j in range(3):
        axes[0, j].set_ylim(e_lo - e_pad, e_hi + e_pad)

    r_lo, r_hi = float(np.min(reward_y)), float(np.max(reward_y))
    r_pad = 0.04 * (r_hi - r_lo) if r_hi > r_lo else 1.0
    for j in range(3):
        axes[1, j].set_ylim(r_lo - r_pad, r_hi + r_pad)

    axes[0, 0].set_ylabel("Energy consumed")
    axes[1, 0].set_ylabel("Env. reward")
    for j in range(3):
        axes[1, j].set_xlabel("Episode")

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
