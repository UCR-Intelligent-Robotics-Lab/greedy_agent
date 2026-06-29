"""Shared ACM sigconf style, metrics, and helpers for SenSys figures (no plotting here)."""

from __future__ import annotations

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAIL = 5
COL_W = 3.33
FULL_W = 7.0

METHOD_STYLE = {
    "LIO": {"fill": "#C2C8CF", "line": "#4F5B66", "hatch": ""},
    "EIA": {"fill": "#F4A6A2", "line": "#E15759", "hatch": ".."},
    "REFiNE": {"fill": "#8FE3C6", "line": "#10B981", "hatch": "//"},
    "REFiNE_EIA": {"fill": "#8FE3C6", "line": "#10B981", "hatch": "\\"},
}

AGENT_COLORS = {1: "#1F77B4", 2: "#9747FF", 3: "#2CA02C", 4: "#403990"}

_RC = {
    "font.family": "serif",
    "font.serif": [
        "Linux Libertine O",
        "Libertinus Serif",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.titlepad": 4,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
}

mpl.rcParams.update(_RC)


def _per_run(csv: str, n_agents: int, tail: int = TAIL) -> tuple[float, float]:
    m = _per_run_metrics(csv, n_agents, tail)
    return m["energy"], m["task"]


def _per_run_metrics(csv: str, n_agents: int, tail: int = TAIL) -> dict[str, float]:
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    t = df.tail(tail)
    env_vals = np.array(
        [float(t[f"A{i}_reward_env"].mean()) for i in range(1, n_agents + 1)]
    )
    has_en = "A1_total_energy" in df.columns
    if has_en:
        en_vals = np.array(
            [float(t[f"A{i}_total_energy"].mean()) for i in range(1, n_agents + 1)]
        )
        energy = float(np.mean(en_vals))
    else:
        energy = float("nan")
    return dict(
        energy=energy,
        task=float(np.mean(env_vals)),
        sigma_agent=float(np.std(env_vals, ddof=0)),
    )


def iqr_stats(a: np.ndarray) -> dict:
    x = np.asarray(a, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return dict(median=float("nan"), q1=float("nan"), q3=float("nan"), iqr=0.0, n=0)
    q1, q3 = np.percentile(x, [25, 75])
    return dict(
        median=float(np.median(x)),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        n=int(x.size),
    )


def load_runs_metrics(
    results_root: str, dir_glob: str, n_agents: int, tail: int = TAIL
) -> dict:
    """Per-run energy, task, sigma_agent arrays plus median/IQR summaries."""
    pattern = f"{results_root}/*/{dir_glob}/log.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No match: {pattern}")

    energy: list[float] = []
    task: list[float] = []
    sigma: list[float] = []
    for p in paths:
        m = _per_run_metrics(p, n_agents, tail)
        energy.append(m["energy"])
        task.append(m["task"])
        sigma.append(m["sigma_agent"])

    en = np.array(energy, dtype=float)
    env = np.array(task, dtype=float)
    sig = np.array(sigma, dtype=float)
    div = _diverged(en, env)

    return dict(
        paths=paths,
        n=len(paths),
        n_conv=len(paths) - len(div),
        diverged=sorted(div),
        energy=en,
        task=env,
        sigma_agent=sig,
        energy_s=iqr_stats(en),
        task_s=iqr_stats(env),
        sigma_s=iqr_stats(sig),
    )


def _diverged(en: np.ndarray, env: np.ndarray) -> set[int]:
    v = en if np.any(~np.isnan(en)) else env
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    thr = max(3 * mad, 0.25 * abs(med), 1e-9)
    return {i for i, x in enumerate(v) if not np.isnan(x) and abs(x - med) > thr}


def load_runs(
    results_root: str, dir_glob: str, n_agents: int, tail: int = TAIL
) -> dict:
    pattern = f"{results_root}/*/{dir_glob}/log.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No match: {pattern}")

    en_list: list[float] = []
    env_list: list[float] = []
    for p in paths:
        e, v = _per_run(p, n_agents, tail)
        en_list.append(e)
        env_list.append(v)

    en = np.array(en_list, dtype=float)
    env = np.array(env_list, dtype=float)
    div = _diverged(en, env)
    conv = [i for i in range(len(paths)) if i not in div]

    def stats(a: np.ndarray) -> dict:
        sub = a[conv] if conv else a
        n_valid = int(np.sum(~np.isnan(a)))
        return dict(
            median=float(np.nanmedian(a)),
            median_conv=float(np.nanmedian(sub)),
            mean=float(np.nanmean(a)),
            std=float(np.nanstd(a, ddof=1)) if n_valid > 1 else 0.0,
            lo=float(np.nanmin(a)),
            hi=float(np.nanmax(a)),
        )

    return dict(
        n=len(paths),
        n_conv=len(conv),
        diverged=sorted(div),
        en=en,
        env=env,
        en_s=stats(en),
        env_s=stats(env),
    )


def load_traj(
    results_root: str, dir_glob: str, n_agents: int, metric: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if metric not in ("energy", "reward"):
        raise ValueError(f"metric must be 'energy' or 'reward', got {metric!r}")
    col = "total_energy" if metric == "energy" else "reward_env"
    pattern = f"{results_root}/*/{dir_glob}/log.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No match: {pattern}")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        ep = df["episode"].values
        val = np.mean(
            [df[f"A{i}_{col}"].values for i in range(1, n_agents + 1)],
            axis=0,
        )
        out.append((ep, val))

    L = min(len(e) for e, _ in out)
    return [(e[:L], v[:L]) for e, v in out]


def median_band(
    trajs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = min(len(v) for _, v in trajs)
    ep = trajs[0][0][:L]
    M = np.vstack([v[:L] for _, v in trajs])
    return ep, np.median(M, axis=0), M


def new_fig(
    nrows: int = 1,
    ncols: int = 1,
    width: float = COL_W,
    panel_h: float = 2.0,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, panel_h * nrows), layout="constrained"
    )
    return fig, np.atleast_1d(axes).ravel()


def grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, lw=0.4, alpha=0.4, ls=":")


def fmt_pct(v: float, base: float) -> str:
    d = (v - base) / base * 100
    return f"{d:+.0f}%"


def glyph_legend(
    ax: plt.Axes,
    handles_labels: list[tuple],
    loc: str = "upper right",
) -> None:
    ax.legend(
        [h for h, _ in handles_labels],
        [l for _, l in handles_labels],
        loc=loc,
        frameon=False,
        handletextpad=0.4,
        borderpad=0.2,
    )


def save(fig: plt.Figure, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path)
