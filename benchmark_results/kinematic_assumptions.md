# Kinematic step-timing model — audit trail

Each item states what is assumed and why.

## Manufacturer and paper-sourced inputs
- **TurtleBot3 Burger** max linear speed 0.22 m/s and max angular speed 2.84 rad/s are taken from the official ROBOTIS e-manual (Burger specifications).
- **Platform 1.85 m × 0.75 m** matches the paper Sec. IV description of a 185×75 cm table for real-world TurtleBot3 deployment.

## Layout
- **Lever, door, and start coordinates** are consistent with paper Fig. 2(B) topology (lever on one short edge, door on the other, agents along the long axis). Exact coordinates are **approximated** inside the documented rectangle so the model is auditable; they are not claimed to be measured from the physical lab.

## Motion model
- **Rotate then translate** to reach lever or door: shortest in-place turn to face the goal, then straight-line motion at the capped speed. This ignores obstacle avoidance and multi-segment paths.
- **`move`**: exactly one **0.25 m** translation along the current heading with no rotation (standard discrete ER grid step size assumption).
- **SPEED_DERATING = 0.7**: analysis uses 70% of datasheet limits as a realistic operating point; **derating = 1.0** is also reported as an optimistic lower bound on duration.

## Manipulation
- **Lever actuation 1.5 s** and **door actuation 1.0 s** are fixed delays for pull/release or door attempt. They are **not** from TurtleBot3 specs; they are conservative placeholders so navigation does not dominate unrealistically when the arm/gripper is slow.

## Multi-agent step semantics
- **Synchronous ER step**: team step time is the **maximum** over the four agents’ physical step durations for that step (the team waits for the slowest member). This matches discrete-step simultaneous-action ER semantics in the paper.

## Action sampling (Monte Carlo)
- **Balanced regime**: P(lever)=0.5, P(door)=0.4, P(move)=0.1 (post-REFiNE balanced behavior, illustrative).
- **Skewed regime**: P(lever)=0.7, P(door)=0.2, P(move)=0.1 (LIO/EIA monopoly regime, illustrative).
- **10 steps per episode × 1000 episodes** = 10,000 samples per regime (paper Fig. 1 caption: episodes typically within a small step budget).
- **RNG**: `numpy.random.default_rng(12345)` for balanced regime and `default_rng(12346)` for skewed regime so the two mixes differ while remaining reproducible (`np.random.seed(12345)` is also set at program start per task).

## Inference comparison
- **REFiNE `run_actor` (prime=False)** median latency is read from `benchmark_results/inference_latency_raw.npz` key `refine_normal` (nanoseconds in file), converted to seconds. This is **Nano compute**, not robot motion.
