#!/usr/bin/env python3
"""
Kinematics-grounded per-step physical duration model for ER(4,2) TurtleBot3.

This module is standalone (NumPy only). It does not import ROS or TensorFlow.
"""

from __future__ import annotations

import math
import os
import traceback
from typing import Literal

import numpy as np

# === SOURCED CONSTANTS (do not modify without updating source) ===
# Source: TurtleBot3 Burger official datasheet
# https://emanual.robotis.com/docs/en/platform/turtlebot3/features/
V_LINEAR_MAX_MPS = 0.22  # max translational velocity (ROBOTIS TurtleBot3 Burger datasheet)
V_ANGULAR_MAX_RADPS = 2.84  # max rotational velocity = 162.72 deg/s (same datasheet)

# Source: paper Sec. IV, "185x75 cm^2 table surface for real-world TurtleBot3 deployment"
PLATFORM_LENGTH_M = 1.85  # 185 cm long edge (paper Sec. IV)
PLATFORM_WIDTH_M = 0.75  # 75 cm short edge (paper Sec. IV)

# Practical operating fraction (robots rarely run at saturation).
# Use 0.7 of nominal as the realistic operating point; also report results at 1.0
# (datasheet max) as a lower bound on duration.
SPEED_DERATING = 0.7  # analysis-only derating; not a manufacturer spec

# Assumed manipulation times (not from datasheet; conservative fixed delays)
T_LEVER_ACTUATE = 1.5  # s, mechanical pull-and-release at lever (engineering assumption)
T_DOOR_ACTUATE = 1.0  # s, door-open attempt at door (engineering assumption)
TILE_SIZE_M = 0.25  # m, one ER discrete grid step for `move` (ER convention assumption)

ActionName = Literal["lever", "door", "move"]

# Paper Fig. 2(B) and platform dimensions; exact coordinates approximated within the documented geometry.
LAYOUT = {
    "lever": (0.15, 0.375),  # short edge of 1.85 x 0.75 m platform
    "door": (1.70, 0.375),  # opposite short edge
    "agent_start": [
        (0.92, 0.20),  # A1
        (0.92, 0.40),  # A2
        (0.92, 0.55),  # A3
        (0.92, 0.65),  # A4
    ],
}


def t_rotate(theta_rad: float, derating: float = SPEED_DERATING) -> float:
    """Time to rotate in place by |theta_rad| radians."""
    return abs(theta_rad) / (V_ANGULAR_MAX_RADPS * derating)


def t_translate(distance_m: float, derating: float = SPEED_DERATING) -> float:
    """Time to translate by distance_m meters in a straight line."""
    return abs(distance_m) / (V_LINEAR_MAX_MPS * derating)


def _shortest_turn(delta_rad: float) -> float:
    """Wrap angle difference to [-pi, pi]."""
    return (delta_rad + math.pi) % (2.0 * math.pi) - math.pi


def t_navigate(
    p_from: tuple[float, float],
    heading_from: float,
    p_to: tuple[float, float],
    derating: float = SPEED_DERATING,
) -> tuple[float, float]:
    """Rotate to face target, then translate straight to p_to.

    Returns (total_time_s, new_heading_at_target).
    """
    x1, y1 = p_from
    x2, y2 = p_to
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-12:
        return 0.0, heading_from
    target_heading = math.atan2(dy, dx)
    turn = _shortest_turn(target_heading - heading_from)
    t_rot = t_rotate(turn, derating=derating)
    t_tr = t_translate(dist, derating=derating)
    return t_rot + t_tr, target_heading


def step_duration(
    agent_pose: tuple[float, float, float],
    action: ActionName,
    layout: dict,
    *,
    derating: float = SPEED_DERATING,
    t_lever_actuate: float = T_LEVER_ACTUATE,
    t_door_actuate: float = T_DOOR_ACTUATE,
    tile_size_m: float = TILE_SIZE_M,
) -> tuple[float, tuple[float, float, float]]:
    """Total physical duration (s) and new pose (x, y, heading) after one ER action."""
    x, y, h = agent_pose
    if action == "lever":
        lv = layout["lever"]
        t_nav, h_new = t_navigate((x, y), h, lv, derating=derating)
        total = t_nav + t_lever_actuate
        return total, (lv[0], lv[1], h_new)
    if action == "door":
        dr = layout["door"]
        t_nav, h_new = t_navigate((x, y), h, dr, derating=derating)
        total = t_nav + t_door_actuate
        return total, (dr[0], dr[1], h_new)
    if action == "move":
        t_m = t_translate(tile_size_m, derating=derating)
        xn = x + tile_size_m * math.cos(h)
        yn = y + tile_size_m * math.sin(h)
        return t_m, (xn, yn, h)
    raise ValueError("unknown action: %r" % (action,))


def _sample_action(rng: np.random.Generator, dist: Literal["balanced", "skewed"]) -> ActionName:
    if dist == "balanced":
        p = (0.5, 0.4, 0.1)
    else:
        p = (0.7, 0.2, 0.1)
    i = int(rng.choice(3, p=p))
    return ("lever", "door", "move")[i]


def monte_carlo_step_durations(
    n_episodes: int = 1000,
    t_max: int = 10,
    dist: Literal["balanced", "skewed"] = "balanced",
    *,
    derating: float = SPEED_DERATING,
    t_lever_actuate: float = T_LEVER_ACTUATE,
    t_door_actuate: float = T_DOOR_ACTUATE,
    rng_seed: int = 12345,
) -> list[float]:
    """Flat list of per-step team durations (max over 4 agents), synchronous semantics."""
    rng = np.random.default_rng(rng_seed)
    out: list[float] = []
    starts = [(sx, sy, 0.0) for sx, sy in LAYOUT["agent_start"]]
    for _ in range(n_episodes):
        poses = [tuple(p) for p in starts]  # type: ignore[arg-type]
        for _step in range(t_max):
            d_per: list[float] = []
            new_poses: list[tuple[float, float, float]] = []
            for pose in poses:
                act = _sample_action(rng, dist)
                dt, new_p = step_duration(
                    pose,
                    act,
                    LAYOUT,
                    derating=derating,
                    t_lever_actuate=t_lever_actuate,
                    t_door_actuate=t_door_actuate,
                )
                d_per.append(dt)
                new_poses.append(new_p)
            out.append(max(d_per))
            poses = new_poses
    return out


def _stats_seconds(arr: np.ndarray) -> tuple[int, float, float, float, float]:
    a = np.asarray(arr, dtype=np.float64)
    n = int(a.size)
    return (
        n,
        float(np.median(a)),
        float(np.mean(a)),
        float(np.percentile(a, 95)),
        float(np.percentile(a, 99)),
    )


def _write_assumptions(path: str) -> None:
    lines = [
        "# Kinematic step-timing model — audit trail",
        "",
        "Each item states what is assumed and why.",
        "",
        "## Manufacturer and paper-sourced inputs",
        "- **TurtleBot3 Burger** max linear speed 0.22 m/s and max angular speed 2.84 rad/s are taken from the official ROBOTIS e-manual (Burger specifications).",
        "- **Platform 1.85 m × 0.75 m** matches the paper Sec. IV description of a 185×75 cm table for real-world TurtleBot3 deployment.",
        "",
        "## Layout",
        "- **Lever, door, and start coordinates** are consistent with paper Fig. 2(B) topology (lever on one short edge, door on the other, agents along the long axis). Exact coordinates are **approximated** inside the documented rectangle so the model is auditable; they are not claimed to be measured from the physical lab.",
        "",
        "## Motion model",
        "- **Rotate then translate** to reach lever or door: shortest in-place turn to face the goal, then straight-line motion at the capped speed. This ignores obstacle avoidance and multi-segment paths.",
        "- **`move`**: exactly one **0.25 m** translation along the current heading with no rotation (standard discrete ER grid step size assumption).",
        "- **SPEED_DERATING = 0.7**: analysis uses 70% of datasheet limits as a realistic operating point; **derating = 1.0** is also reported as an optimistic lower bound on duration.",
        "",
        "## Manipulation",
        "- **Lever actuation 1.5 s** and **door actuation 1.0 s** are fixed delays for pull/release or door attempt. They are **not** from TurtleBot3 specs; they are conservative placeholders so navigation does not dominate unrealistically when the arm/gripper is slow.",
        "",
        "## Multi-agent step semantics",
        "- **Synchronous ER step**: team step time is the **maximum** over the four agents’ physical step durations for that step (the team waits for the slowest member). This matches discrete-step simultaneous-action ER semantics in the paper.",
        "",
        "## Action sampling (Monte Carlo)",
        "- **Balanced regime**: P(lever)=0.5, P(door)=0.4, P(move)=0.1 (post-REFiNE balanced behavior, illustrative).",
        "- **Skewed regime**: P(lever)=0.7, P(door)=0.2, P(move)=0.1 (LIO/EIA monopoly regime, illustrative).",
        "- **10 steps per episode × 1000 episodes** = 10,000 samples per regime (paper Fig. 1 caption: episodes typically within a small step budget).",
        "- **RNG**: `numpy.random.default_rng(12345)` for balanced regime and `default_rng(12346)` for skewed regime so the two mixes differ while remaining reproducible (`np.random.seed(12345)` is also set at program start per task).",
        "",
        "## Inference comparison",
        "- **REFiNE `run_actor` (prime=False)** median latency is read from `benchmark_results/inference_latency_raw.npz` key `refine_normal` (nanoseconds in file), converted to seconds. This is **Nano compute**, not robot motion.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _load_inference_median_s(npz_path: str) -> float | None:
    if not os.path.isfile(npz_path):
        return None
    try:
        d = np.load(npz_path)
        if "refine_normal" not in d.files:
            return None
        ns = np.asarray(d["refine_normal"], dtype=np.float64)
        return float(np.median(ns)) * 1e-9
    except Exception:
        return None


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(repo, "benchmark_results")
    npz_out = os.path.join(results_dir, "kinematic_step_durations.npz")
    summary_path = os.path.join(results_dir, "kinematic_summary.txt")
    assumptions_path = os.path.join(results_dir, "kinematic_assumptions.md")
    infer_npz = os.path.join(results_dir, "inference_latency_raw.npz")

    stages_ok: list[str] = []
    try:
        # Stage 0
        if not os.path.isdir(results_dir):
            raise FileNotFoundError("benchmark_results missing: %s" % results_dir)
        stages_ok.append("Stage 0: benchmark_results exists")

        np.random.seed(12345)

        balanced = np.asarray(
            monte_carlo_step_durations(dist="balanced", derating=SPEED_DERATING, rng_seed=12345),
            dtype=np.float64,
        )
        skewed = np.asarray(
            monte_carlo_step_durations(dist="skewed", derating=SPEED_DERATING, rng_seed=12346),
            dtype=np.float64,
        )

        np.savez(npz_out, balanced_seconds=balanced, skewed_seconds=skewed)
        stages_ok.append("Stage 5: saved %s" % npz_out)

        bal_full = np.asarray(
            monte_carlo_step_durations(dist="balanced", derating=1.0, rng_seed=12345),
            dtype=np.float64,
        )
        skw_full = np.asarray(
            monte_carlo_step_durations(dist="skewed", derating=1.0, rng_seed=12346),
            dtype=np.float64,
        )

        inf_med = _load_inference_median_s(infer_npz)

        nb, med_b, mean_b, p95_b, p99_b = _stats_seconds(balanced)
        ns, med_s, mean_s, p95_s, p99_s = _stats_seconds(skewed)
        med_b1 = float(np.median(bal_full))
        med_s1 = float(np.median(skw_full))

        lines = [
            "=== Q2 Physical Step Duration (Kinematics-Grounded Estimate) ===",
            "NOTE: This is a kinematics-grounded analysis using TurtleBot3 manufacturer",
            "specifications and the paper's documented platform geometry. It is NOT",
            "a real-world stopwatch measurement.",
            "Source constants:",
            "TurtleBot3 Burger max linear  = 0.22 m/s   (datasheet)",
            "TurtleBot3 Burger max angular = 2.84 rad/s (datasheet)",
            "Platform                       = 185 x 75 cm  (paper Sec. IV)",
            "Operating speed derating       = 0.7 of nominal",
            "Lever actuation                = 1.5 s  (assumed)",
            "Door actuation                 = 1.0 s  (assumed)",
            "Tile size (move action)        = 0.25 m  (assumed, ER discrete grid)",
            "Per-step team duration (max over 4 agents per synchronous step):",
            "Regime: balanced (post-REFiNE)",
            "n_steps = %d,  median = %.6f s, mean = %.6f s, p95 = %.6f s, p99 = %.6f s"
            % (nb, med_b, mean_b, p95_b, p99_b),
            "Regime: skewed (pre-REFiNE / EIA-monopoly)",
            "n_steps = %d,  median = %.6f s, mean = %.6f s, p95 = %.6f s, p99 = %.6f s"
            % (ns, med_s, mean_s, p95_s, p99_s),
            "ALSO REPORT at full datasheet speed (derating = 1.0) as a tighter",
            "lower bound on duration:",
            "balanced median = %.6f s" % med_b1,
            "skewed   median = %.6f s" % med_s1,
            "Comparison:",
        ]
        if inf_med is not None:
            inf_ms = inf_med * 1e3
            ratio_b = med_b / inf_med if inf_med > 0 else float("nan")
            log_o = math.log10(ratio_b) if ratio_b > 0 and math.isfinite(ratio_b) else float("nan")
            lines.append(
                "Inference (run_actor, REFiNE prime=False) median  = %.3f ms (from inference_latency_raw.npz, refine_normal)"
                % inf_ms
            )
            lines.append("Physical step median (balanced) = %.6f s" % med_b)
            lines.append(
                "Ratio = %.4g (physical / inference); inference is ~%.2f orders of magnitude below step duration (by time)"
                % (ratio_b, log_o)
            )
        else:
            lines.append(
                "Inference median unavailable (missing refine_normal in %s)." % infer_npz
            )

        # Stage 7 sensitivity (same rng seeds as main so action paths match per regime)
        b_half = np.asarray(
            monte_carlo_step_durations(
                dist="balanced",
                derating=SPEED_DERATING,
                t_lever_actuate=0.75,
                t_door_actuate=0.5,
                rng_seed=12345,
            ),
            dtype=np.float64,
        )
        s_half = np.asarray(
            monte_carlo_step_durations(
                dist="skewed",
                derating=SPEED_DERATING,
                t_lever_actuate=0.75,
                t_door_actuate=0.5,
                rng_seed=12346,
            ),
            dtype=np.float64,
        )
        b_dbl = np.asarray(
            monte_carlo_step_durations(
                dist="balanced",
                derating=SPEED_DERATING,
                t_lever_actuate=3.0,
                t_door_actuate=2.0,
                rng_seed=12345,
            ),
            dtype=np.float64,
        )
        s_dbl = np.asarray(
            monte_carlo_step_durations(
                dist="skewed",
                derating=SPEED_DERATING,
                t_lever_actuate=3.0,
                t_door_actuate=2.0,
                rng_seed=12346,
            ),
            dtype=np.float64,
        )

        lines.extend(
            [
                "",
                "=== Sensitivity (median team-step duration, seconds) ===",
                "Halved actuation (T_LEVER_ACTUATE=0.75 s, T_DOOR_ACTUATE=0.5 s), derating=0.7:",
                "  balanced median = %.6f s, skewed median = %.6f s"
                % (float(np.median(b_half)), float(np.median(s_half))),
                "Doubled actuation (T_LEVER_ACTUATE=3.0 s, T_DOOR_ACTUATE=2.0 s), derating=0.7:",
                "  balanced median = %.6f s, skewed median = %.6f s"
                % (float(np.median(b_dbl)), float(np.median(s_dbl))),
                "Datasheet speed (derating=1.0), default actuation:",
                "  balanced median = %.6f s, skewed median = %.6f s" % (med_b1, med_s1),
            ]
        )

        if inf_med is not None and inf_med > 0:
            aggressive = min(
                float(np.median(b_half)),
                float(np.median(s_half)),
                med_b1,
                med_s1,
            )
            lines.append(
                "Most aggressive (smallest) physical median among sensitivity cases: %.6f s"
                % aggressive
            )
            lines.append(
                "vs inference median %.6f s → factor %.4g (target: physical at least 100× inference)"
                % (inf_med, aggressive / inf_med)
            )

        summary_text = "\n".join(lines) + "\n"
        with open(summary_path, "w") as f:
            f.write(summary_text)

        _write_assumptions(assumptions_path)
        stages_ok.append("Stage 6–7: wrote %s" % summary_path)
        stages_ok.append("Wrote %s" % assumptions_path)

        print(summary_text)
        print("Files written under benchmark_results/:")
        for name in (
            "kinematic_step_durations.npz",
            "kinematic_summary.txt",
            "kinematic_assumptions.md",
        ):
            p = os.path.join(results_dir, name)
            print(" ", p, "(exists=%s)" % os.path.isfile(p))

        print("\nStages completed: %s" % "; ".join(stages_ok))

    except Exception:
        print(traceback.format_exc())
        print("Failed after: %s" % "; ".join(stages_ok))


if __name__ == "__main__":
    main()
