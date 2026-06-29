#!/usr/bin/env python3
"""Background fig. 2.2: LIO reward inequality on ER(4,2) — 10-seed nano baseline."""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sensys_runs.plot.style as sty

RESULTS_ROOT = "lio/results"
OUT_PATH = "fig/bg_reward_split.pdf"
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
    for p in paths:
        print(f"    nano_er{_seed_key(p):2d}  {p}")
    missing = sorted(set(range(1, EXPECTED_SEEDS + 1)) - set(seeds))
    if missing:
        print(f"  Missing seeds: {missing}")
    return paths


def _read_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def per_checkpoint_series(
    paths: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Episode grid, per-seed sigma_agent, per-seed team_reward."""
    sigmas: list[np.ndarray] = []
    teams: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    for p in paths:
        df = _read_log(p)
        ep = df["episode"].values.astype(float)
        rewards = np.stack(
            [df[f"A{i}_reward_env"].values.astype(float) for i in range(1, N_AGENTS + 1)],
            axis=1,
        )
        sigmas.append(np.std(rewards, axis=1, ddof=0))
        teams.append(np.sum(rewards, axis=1))
        episodes.append(ep)
    L = min(len(s) for s in sigmas)
    ep = episodes[0][:L]
    return ep, [s[:L] for s in sigmas], [t[:L] for t in teams]


def final_rewards_by_seed(paths: list[str]) -> tuple[list[int], np.ndarray]:
    """Seed indices and (n_seeds, n_agents) tail-5 mean A{i}_reward_env."""
    seeds: list[int] = []
    rows: list[list[float]] = []
    for p in paths:
        seeds.append(_seed_key(p))
        df = _read_log(p)
        t = df.tail(TAIL)
        rows.append(
            [float(t[f"A{i}_reward_env"].mean()) for i in range(1, N_AGENTS + 1)]
        )
    return seeds, np.array(rows)


def _warn_role_split(M: np.ndarray, seed_ids: list[int], overall_mean: float) -> None:
    for i, seed in enumerate(seed_ids):
        row = M[i]
        n_high = int(np.sum(row > overall_mean))
        n_low = int(np.sum(row <= overall_mean))
        if n_high != 2 or n_low != 2:
            vals = ", ".join(f"A{k + 1}={row[k]:.1f}" for k in range(N_AGENTS))
            print(
                f"WARNING: seed {seed} lacks clean 2-high/2-low split "
                f"(high={n_high}, low={n_low}): {vals}"
            )


def _annotate_heatmap(ax, M: np.ndarray, cmap, norm) -> None:
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = M[i, j]
            r, g, b, _ = cmap(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "black" if lum > 0.55 else "white"
            ax.text(
                j,
                i,
                f"{round(val):d}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )


def convergence_stats(
    paths: list[str], ep: np.ndarray, sigmas: list[np.ndarray], teams: list[np.ndarray]
) -> None:
    team_final = []
    agent_final_all = []
    sigma_final = []
    for p in paths:
        df = _read_log(p)
        t = df.tail(TAIL)
        af = np.array(
            [float(t[f"A{i}_reward_env"].mean()) for i in range(1, N_AGENTS + 1)]
        )
        agent_final_all.extend(af.tolist())
        team_final.append(float(np.sum(af)))
        sigma_final.append(float(np.std(af, ddof=0)))

    team_med = float(np.median(team_final))
    agent_mean = float(np.mean(agent_final_all))
    sigma_med = float(np.median(sigma_final))
    cv = sigma_med / agent_mean if agent_mean else float("nan")
    sigma_lo, sigma_hi = float(np.min(sigma_final)), float(np.max(sigma_final))

    sorted_rewards = np.sort(agent_final_all)
    n = len(sorted_rewards)
    low_tier = float(np.median(sorted_rewards[: n // 2]))
    high_tier = float(np.median(sorted_rewards[n // 2 :]))

    # Also report trajectory medians at last checkpoint (should match final closely)
    sigma_traj_last = float(np.median([s[-1] for s in sigmas]))
    team_traj_last = float(np.median([t[-1] for t in teams]))

    print("\n=== Convergence stats (tail-5 final eval) ===")
    print(f"  seeds used:              {len(paths)}")
    print(f"  team reward (median):    {team_med:.2f}")
    print(f"  mean per-agent reward:   {agent_mean:.2f}")
    print(f"  sigma_agent (median):    {sigma_med:.2f}")
    print(f"  CV (sigma/mean):         {cv:.3f}")
    print(f"  sigma_agent per-seed:    min={sigma_lo:.2f}  max={sigma_hi:.2f}")
    print(f"  high-tier reward (~):    {high_tier:.1f}")
    print(f"  low-tier reward (~):     {low_tier:.1f}")
    print(f"  (last-checkpoint median team={team_traj_last:.2f}, sigma={sigma_traj_last:.2f})")


def main() -> None:
    paths = discover_runs()
    if not paths:
        sys.exit(1)

    ep, sigmas, teams = per_checkpoint_series(paths)
    lio = sty.METHOD_STYLE["LIO"]

    fig, axes = sty.new_fig(1, 2, width=sty.FULL_W, panel_h=2.4)
    ax_a, ax_b = axes

    # --- (b) inequality grows while team reward converges ---
    ax_a.set_title("(b) Reward inequality grows during training", loc="left", fontweight="bold")
    for s in sigmas:
        ax_a.plot(ep, s, color="#888888", lw=0.7, alpha=0.3)
    sigma_med = np.median(np.vstack(sigmas), axis=0)
    ax_a.plot(
        ep,
        sigma_med,
        color=lio["line"],
        lw=1.8,
        label="Median $\\sigma_{\\mathrm{agent}}$",
    )
    ax_a.set_xlabel("Episode")
    ax_a.set_ylabel("Inter-agent reward std")
    sty.grid(ax_a, "y")

    ax_r = ax_a.twinx()
    team_med = np.median(np.vstack(teams), axis=0)
    ax_r.plot(
        ep,
        team_med,
        color=lio["line"],
        lw=1.2,
        ls="--",
        alpha=0.85,
        label="Median team reward",
    )
    ax_r.set_ylabel("Team reward")

    h1 = mlines.Line2D([], [], color=lio["line"], lw=1.8, label="Median $\\sigma_{\\mathrm{agent}}$")
    h2 = mlines.Line2D([], [], color=lio["line"], lw=1.2, ls="--", label="Median team reward")
    sty.glyph_legend(ax_a, [(h1, h1.get_label()), (h2, h2.get_label())], loc="center right")

    # --- (c) seed-dependent role split (heatmap) ---
    ax_b.set_title("(c) Role assignment is a seed-dependent split", loc="left", fontweight="bold")
    seed_ids, M = final_rewards_by_seed(paths)
    overall_mean = float(np.mean(M))
    norm = TwoSlopeNorm(vcenter=overall_mean, vmin=float(M.min()), vmax=float(M.max()))
    cmap = plt.cm.RdBu_r
    im = ax_b.imshow(M, aspect="auto", cmap=cmap, norm=norm, origin="upper")
    _annotate_heatmap(ax_b, M, cmap, norm)
    ax_b.set_xticks(range(N_AGENTS), labels=[f"A{i}" for i in range(1, N_AGENTS + 1)])
    ax_b.set_yticks(range(len(seed_ids)), labels=seed_ids)
    ax_b.set_ylabel("Seed")
    _warn_role_split(M, seed_ids, overall_mean)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("Final reward")
    cbar.ax.tick_params(labelsize=6)

    convergence_stats(paths, ep, sigmas, teams)
    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
