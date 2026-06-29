#!/usr/bin/env python3
"""Background fig. 2.3: LIO energy–reward coupling on ER(4,2) — 10-seed nano baseline."""

from __future__ import annotations

import glob
import re
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
OUT_PATH = "fig/bg_energy_coupling.pdf"
DIR_NAME = "er_lio_4_2"
N_AGENTS = 4
TAIL = sty.TAIL
EXPECTED_SEEDS = 10


def _seed_key(path: str) -> int:
    m = re.search(r"nano_er(\d+)", path)
    return int(m.group(1)) if m else 0


def discover_runs() -> list[str]:
    pattern = f"{RESULTS_ROOT}/nano_er*/{DIR_NAME}/log.csv"
    paths = sorted(glob.glob(pattern), key=_seed_key)
    print(f"=== LIO ER(4,2) baseline search: {pattern!r} ===")
    if not paths:
        print("  No runs found.")
        return []
    seeds = [_seed_key(p) for p in paths]
    print(f"  Found {len(paths)}/{EXPECTED_SEEDS} seeds: {seeds}")
    missing = sorted(set(range(1, EXPECTED_SEEDS + 1)) - set(seeds))
    if missing:
        print(f"  Missing seeds: {missing}")
    return paths


def _read_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def collect_final_points(
    paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reward, energy, agent_id arrays (one point per seed × agent)."""
    rewards: list[float] = []
    energies: list[float] = []
    agents: list[int] = []
    for p in paths:
        df = _read_log(p)
        t = df.tail(TAIL)
        for i in range(1, N_AGENTS + 1):
            rewards.append(float(t[f"A{i}_reward_env"].mean()))
            energies.append(float(t[f"A{i}_total_energy"].mean()))
            agents.append(i)
    return np.array(rewards), np.array(energies), np.array(agents)


def _cluster_by_reward_sign(
    reward: np.ndarray, energy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal (reward > 0) vs support (reward <= 0)."""
    terminal = reward > 0
    support = ~terminal
    return terminal, support


def _centroid(reward: np.ndarray, energy: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return float(np.mean(reward[mask])), float(np.mean(energy[mask]))


def main() -> None:
    paths = discover_runs()
    if not paths:
        sys.exit(1)

    reward, energy, agents = collect_final_points(paths)
    terminal_mask, support_mask = _cluster_by_reward_sign(reward, energy)
    term_c = _centroid(reward, energy, terminal_mask)
    supp_c = _centroid(reward, energy, support_mask)
    en_lo, en_hi = float(np.min(energy)), float(np.max(energy))

    fig, axes = sty.new_fig(1, 1, width=sty.COL_W, panel_h=2.6)
    ax = axes[0]
    ax.set_xlim(-30, 120)
    ax.set_ylim(25, 230)

    for aid in range(1, N_AGENTS + 1):
        mask = agents == aid
        ax.scatter(
            reward[mask],
            energy[mask],
            c=sty.AGENT_COLORS[aid],
            s=36,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )

    arrow_kw = dict(arrowstyle="->", color="black", lw=0.8)
    ax.annotate(
        "Support role\n(low reward, high energy)",
        xy=supp_c,
        xytext=(25, 135),
        textcoords="data",
        fontsize=7.5,
        ha="center",
        va="center",
        arrowprops=arrow_kw,
        clip_on=True,
    )
    ax.annotate(
        "Terminal role\n(high reward, low energy)",
        xy=term_c,
        xytext=(45, 80),
        textcoords="data",
        fontsize=7.5,
        ha="center",
        va="center",
        arrowprops=arrow_kw,
        clip_on=True,
    )

    ax.set_xlabel("Final environment reward")
    ax.set_ylabel("Final energy consumed")
    sty.grid(ax, "both")

    legend_entries = [
        (
            mlines.Line2D(
                [],
                [],
                color=sty.AGENT_COLORS[i],
                marker="o",
                ls="",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.35,
                label=f"A{i}",
            ),
            f"A{i}",
        )
        for i in range(1, N_AGENTS + 1)
    ]
    sty.glyph_legend(ax, legend_entries, loc="lower left")

    fig.suptitle(
        "Role specialization couples low reward with high energy",
        x=0.5,
        ha="center",
        fontweight="bold",
        fontsize=7,
    )

    print("\n=== Energy–reward coupling (tail-5 final) ===")
    print(f"  points:              {len(reward)} ({len(paths)} seeds × {N_AGENTS} agents)")
    print(f"  terminal centroid:   reward={term_c[0]:.1f}, energy={term_c[1]:.1f}")
    print(f"  support centroid:    reward={supp_c[0]:.1f}, energy={supp_c[1]:.1f}")
    print(f"  energy range:        min={en_lo:.1f}, max={en_hi:.1f}")

    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
