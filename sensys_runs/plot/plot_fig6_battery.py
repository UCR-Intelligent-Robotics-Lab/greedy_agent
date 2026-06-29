#!/usr/bin/env python3
"""Fig. 6: Real-world per-agent battery depletion (ER 4,2) from test episode step logs."""

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
OUT_PATH = "fig/eval_battery.pdf"
N_AGENTS = 4
BATTERY_J = 25.0
# Original EMSOFT / plot_ER42_REFiNE.ipynb shutdown tag: remaining J <= 1e-6
SHUTDOWN_REMAINING_J = 1e-6
A2_COLOR = "#E15759"

# Expected outputs of run_trained_*_er.py (exp_num=1); override EXP_NUM if needed.
EXP_NUM = 1
CONDITIONS = [
    ("(a) LIO", "LIO", f"er{EXP_NUM}/test_er_lio_4_2/test_epsiode_log.csv", False),
    ("(b) EIA", "EIA", f"er{EXP_NUM}/test_er_lio_attack_4_2/test_epsiode_log.csv", True),
    ("(c) REFiNE", "REFiNE", f"er{EXP_NUM}/test_er_REFiNE_attack_4_2/test_epsiode_log.csv", True),
]

SEARCH_GLOBS = [
    f"{RESULTS_ROOT}/er*/test_er_lio_4_2/test_epsiode_log.csv",
    f"{RESULTS_ROOT}/er*/test_er_lio_attack_4_2/test_epsiode_log.csv",
    f"{RESULTS_ROOT}/er*/test_er_REFiNE_attack_4_2/test_epsiode_log.csv",
    f"{RESULTS_ROOT}/**/test_epsiode_log.csv",
    f"{RESULTS_ROOT}/**/*battery*",
    f"{RESULTS_ROOT}/**/real*",
]


def locate_logs() -> dict[str, str | None]:
    """Resolve per-condition CSV paths; print discoveries."""
    found: dict[str, str | None] = {}
    print("=== Battery log search ===")
    for pat in SEARCH_GLOBS:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            print(f"  glob {pat!r}:")
            for h in hits:
                print(f"    {h}")
    for _title, key, rel, _ in CONDITIONS:
        path = os.path.join(RESULTS_ROOT, rel)
        exists = os.path.isfile(path)
        print(f"  [{key}] expected: {path}  {'OK' if exists else 'MISSING'}")
        found[key] = path if exists else None
    return found


def load_battery_pct(csv_path: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    it = df["step"].values.astype(int)
    pct: dict[int, np.ndarray] = {}
    for aid in range(1, N_AGENTS + 1):
        tot = df[f"A{aid}_total_consumed_energy"].values.astype(float)
        remain = np.maximum(0.0, BATTERY_J - tot)
        pct[aid] = 100.0 * remain / BATTERY_J
    return it, pct


def first_depletion_iteration(it: np.ndarray, pct: dict[int, np.ndarray]) -> int:
    """First iteration where any agent hits the notebook shutdown cutoff."""
    cutoff_pct = 100.0 * SHUTDOWN_REMAINING_J / BATTERY_J
    firsts = []
    for aid in range(1, N_AGENTS + 1):
        hit = np.where(pct[aid] <= cutoff_pct)[0]
        if len(hit):
            firsts.append(int(it[hit[0]]))
    if not firsts:
        # fallback: first agent to reach global minimum remaining %
        mins = [int(it[int(np.argmin(pct[aid]))]) for aid in range(1, N_AGENTS + 1)]
        return int(min(mins))
    return int(min(firsts))


def _step_markevery(it: np.ndarray) -> int:
    if len(it) < 2:
        return 1
    med = float(np.median(np.diff(it.astype(float))))
    if med <= 0:
        return 1
    return max(1, int(round(5 / med)))


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


def _plot_panel(
    ax,
    it: np.ndarray,
    pct: dict[int, np.ndarray],
    *,
    title: str,
    highlight_a2: bool,
    depletion_it: int,
) -> None:
    markevery = _step_markevery(it)
    for aid in range(1, N_AGENTS + 1):
        y = pct[aid]
        if highlight_a2 and aid == 2:
            ax.plot(
                it,
                y,
                color=A2_COLOR,
                lw=1.7,
                marker="*",
                markevery=markevery,
                markersize=5,
            )
        else:
            ax.plot(
                it,
                y,
                color=sty.AGENT_COLORS[aid],
                lw=1.0,
                marker="o",
                markevery=markevery,
                markersize=2.5,
            )
    ax.axvline(depletion_it, color="#C44E52", ls="--", lw=0.9, zorder=0)
    ax.text(
        depletion_it,
        100.0,
        str(depletion_it),
        color="#C44E52",
        ha="center",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(0, 100)
    sty.grid(ax, "y")
    sty.glyph_legend(ax, _legend_entries(highlight_a2), loc="upper right")


def main() -> None:
    paths = locate_logs()
    missing = [k for k, p in paths.items() if p is None]
    if missing:
        print(
            "\nSTOP: real-world battery step logs not found on disk.\n"
            "Generate them (once) from lio/alg/ with conda env LIO_tecs, e.g.:\n"
            "  cd lio/alg && python run_trained_lio_er.py\n"
            "  cd lio/alg && python run_trained_lio_eia_er.py\n"
            "  cd lio/alg && python run_trained_REFiNE_eia_er.py\n"
            "Or point EXP_NUM / CONDITIONS at your copy of test_epsiode_log.csv.\n"
            f"Missing: {', '.join(missing)}"
        )
        sys.exit(1)

    fig, axes = sty.new_fig(1, 3, width=sty.FULL_W, panel_h=2.2)
    print("\n=== First depletion iteration (shutdown cutoff) ===")
    for ax, (title, key, _rel, highlight_a2) in zip(axes, CONDITIONS):
        csv_path = paths[key]
        assert csv_path is not None
        it, pct = load_battery_pct(csv_path)
        dep = first_depletion_iteration(it, pct)
        print(f"  {key:8s}  {dep}  ({csv_path})")
        _plot_panel(
            ax, it, pct, title=title, highlight_a2=highlight_a2, depletion_it=dep
        )

    axes[0].set_ylabel("Battery percentage (%)")
    for ax in axes:
        ax.set_xlabel("Iteration")
    sty.save(fig, OUT_PATH)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
