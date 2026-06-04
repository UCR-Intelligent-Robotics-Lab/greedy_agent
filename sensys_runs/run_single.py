#!/usr/bin/env python3
"""Launch one LIO / REFiNE training run in an isolated process."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform as platform_mod
import re
import random
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIO_ALG = os.path.join(REPO_ROOT, "lio", "alg")

ER_METHODS = ("lio_er", "lio_eia_er", "refine_er", "refine_eia_er")
EIA_ER_METHODS = ("lio_eia_er", "refine_eia_er")

# method -> (config module basename, train module basename)
METHOD_MAP: dict[str, tuple[str, str]] = {
    "lio_er": ("config_room_lio", "train_lio_er"),
    "lio_eia_er": ("config_room_lio", "train_lio_eia_er"),
    "lio_ipd": ("config_ipd_lio", "train_lio_ipd"),
    "lio_eia_ipd": ("config_ipd_lio", "train_lio_eia_ipd"),
    "refine_er": ("config_room_REFiNE", "train_REFiNE_er"),
    "refine_eia_er": ("config_room_REFiNE", "train_REFiNE_eia_er"),
    "refine_ipd": ("config_ipd_REFiNE", "train_REFiNE_ipd"),
    "refine_eia_ipd": ("config_ipd_REFiNE", "train_REFiNE_eia_ipd"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single LIO / REFiNE training job.")
    p.add_argument(
        "--method",
        required=True,
        choices=list(METHOD_MAP),
    )
    p.add_argument("--exp_name", required=True)
    p.add_argument("--dir_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--platform", type=str, default="nano")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--n_eval", type=int, default=None)
    p.add_argument("--period", type=int, default=None)
    p.add_argument("--n_agents", type=int, default=None)
    p.add_argument("--min_at_lever", type=int, default=None)
    p.add_argument("--w_lever", type=float, default=None)
    p.add_argument("--w_door", type=float, default=None)
    p.add_argument("--fairness_mult", type=float, default=None)
    p.add_argument("--energy_weight", type=float, default=None)
    p.add_argument(
        "--fairness_clip",
        action="store_true",
        default=False,
        help="Clip per-agent fairness coef g_i to be non-negative (REFiNE only)",
    )
    p.add_argument("--use_gpu", action="store_true", default=False)
    return p.parse_args()


def _set_thread_env(threads: int) -> None:
    # Best-effort caps; OMP/BLAS dominate CPU thrash on small MLP runs (TF1.x session
    # thread counts are set inside the unedited train scripts).
    val = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[key] = val


def _git_field(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def _conda_env_name() -> str | None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not conda_env or conda_env == "base":
        m = re.search(r"/envs/([^/]+)/", sys.executable)
        if m:
            conda_env = m.group(1)
    return conda_env


def _write_provenance(
    log_path: str,
    args: argparse.Namespace,
    cfg: Any,
    tf_version: str,
    np_version: str,
) -> None:
    record: dict[str, Any] = {
        "platform": args.platform,
        "hostname": socket.gethostname(),
        "git_branch": _git_field(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git_field(["git", "rev-parse", "HEAD"]),
        "git_dirty": _git_dirty(),
        "conda_env": _conda_env_name(),
        "python": platform_mod.python_version(),
        "executable": sys.executable,
        "tensorflow": tf_version,
        "numpy": np_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "exp_name": args.exp_name,
        "dir_name": args.dir_name,
        "seed": args.seed,
        "n_episodes": int(cfg.alg.n_episodes),
        "period": int(cfg.alg.period),
        "n_eval": int(cfg.alg.n_eval),
        "use_gpu": bool(cfg.main.use_gpu),
        "threads": args.threads,
    }
    if args.method in ER_METHODS:
        record["n_agents"] = int(cfg.env.n_agents)
        record["min_at_lever"] = int(cfg.env.min_at_lever)
    if args.method in EIA_ER_METHODS:
        record["eia_w_lever"] = getattr(cfg.lio, "eia_w_lever", None)
        record["eia_w_door"] = getattr(cfg.lio, "eia_w_door", None)
    if args.method.startswith("refine_"):
        record["fairness_mult"] = getattr(cfg.lio, "Fairness_multiplier", None)
        if args.method in ("refine_er", "refine_eia_er"):
            record["energy_weight"] = getattr(cfg.lio, "energy_weight", None)
        record["fairness_clip"] = bool(getattr(cfg.lio, "fairness_clip", False))

    try:
        with open(os.path.join(log_path, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"WARNING: could not write provenance.json: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    _set_thread_env(args.threads)

    for path in (REPO_ROOT, LIO_ALG):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.chdir(LIO_ALG)

    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    import tensorflow as tf

    try:
        tf.set_random_seed(args.seed)
    except AttributeError:
        tf.compat.v1.set_random_seed(args.seed)

    tf_version = getattr(tf, "__version__", str(tf))
    np_version = np.__version__

    config_mod_name, train_mod_name = METHOD_MAP[args.method]
    config_mod = importlib.import_module(f"lio.alg.{config_mod_name}")
    train_mod = importlib.import_module(f"lio.alg.{train_mod_name}")
    cfg = config_mod.get_config()
    train = train_mod.train

    cfg.main.exp_name = args.exp_name
    cfg.main.dir_name = args.dir_name
    cfg.main.seed = args.seed
    cfg.main.use_gpu = args.use_gpu

    if args.n_episodes is not None:
        cfg.alg.n_episodes = args.n_episodes
    if args.n_eval is not None:
        cfg.alg.n_eval = args.n_eval
    if args.period is not None:
        cfg.alg.period = args.period

    if args.method in ER_METHODS:
        if args.n_agents is not None:
            cfg.env.n_agents = args.n_agents
        if args.min_at_lever is not None:
            cfg.env.min_at_lever = args.min_at_lever
            cfg.lio.min_at_lever = args.min_at_lever

    if args.method in EIA_ER_METHODS:
        if args.w_lever is not None:
            cfg.lio.eia_w_lever = args.w_lever
        if args.w_door is not None:
            cfg.lio.eia_w_door = args.w_door

    if args.method.startswith("refine_"):
        if args.fairness_mult is not None:
            cfg.lio.Fairness_multiplier = args.fairness_mult
        if args.energy_weight is not None and args.method in (
            "refine_er",
            "refine_eia_er",
        ):
            cfg.lio.energy_weight = args.energy_weight
        if args.fairness_clip:
            cfg.lio.fairness_clip = True

    log_path = os.path.join(REPO_ROOT, "lio", "results", args.exp_name, args.dir_name)
    os.makedirs(log_path, exist_ok=True)
    _write_provenance(log_path, args, cfg, tf_version, np_version)

    extra = []
    if args.method in ER_METHODS:
        extra.append(f"n_agents={cfg.env.n_agents} min_at_lever={cfg.env.min_at_lever}")
    if args.method in EIA_ER_METHODS:
        wl = getattr(cfg.lio, "eia_w_lever", None)
        wd = getattr(cfg.lio, "eia_w_door", None)
        if wl is not None:
            extra.append(f"w_lever={wl} w_door={wd}")
    if args.method.startswith("refine_"):
        extra.append(f"Fairness_multiplier={cfg.lio.Fairness_multiplier}")
        if args.method in ("refine_er", "refine_eia_er"):
            extra.append(f"energy_weight={cfg.lio.energy_weight}")
        if getattr(cfg.lio, "fairness_clip", False):
            extra.append("fairness_clip=True")

    print(
        f"RUN {args.method} exp={args.exp_name} dir={args.dir_name} seed={args.seed} "
        f"platform={args.platform} threads={args.threads} "
        f"n_episodes={cfg.alg.n_episodes} period={cfg.alg.period} n_eval={cfg.alg.n_eval} "
        f"{' '.join(extra)} use_gpu={cfg.main.use_gpu}"
    )

    try:
        train(cfg)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(f"RUN_OK {args.dir_name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
