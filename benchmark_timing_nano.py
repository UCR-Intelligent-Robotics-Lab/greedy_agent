#!/usr/bin/env python
"""
Standalone timing benchmark: LIO vs REFiNE on Orin Nano (read-only observer).
Do not import modified lio/alg env utils or train_*.py beyond normal imports.
"""
from __future__ import print_function

import csv
import os
import sys
import time
import traceback

import numpy as np

# Repo root on sys.path
_REPO = os.path.abspath(os.path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()

from scipy import stats

from lio.alg import config_room_lio, config_room_REFiNE
from lio.env import room_symmetric


# --- Buffer + run_episode (mirrors train_lio_er / train_REFiNE_eia_er, ER only) ---


class Buffer(object):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_others = []
        self.r_given = []
        self.action_all = []
        self.energy_cost = []
        self.total_energy = 0

    def add(self, transition, energy):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        self.energy_cost.append(energy)
        self.total_energy += energy

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)


def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = 0
    while not done:
        list_actions = list(range(len(list_agents)))
        for agent in list_agents:
            action = agent.run_actor(
                list_obs[agent.agent_id], sess, epsilon, prime
            )
            list_actions[agent.agent_id] = action

        list_rewards = list(range(len(list_agents)))
        total_reward_given_to_each_agent = np.zeros((env.n_agents, env.n_agents))
        for idx, agent in enumerate(list_agents):
            if agent.can_give:
                reward = agent.give_reward(
                    list_obs[agent.agent_id], list_actions, sess
                )
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent[idx] += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards[agent.agent_id] = reward

        list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)

        for idx, buf in enumerate(list_buffers):
            energy_cost = list_agents[idx].calculate_energy_cost(
                list_obs[idx], list_actions[idx]
            )
            buf.add(
                [
                    list_obs[idx],
                    list_actions[idx],
                    env_rewards[idx],
                    list_obs_next[idx],
                    done,
                ],
                energy_cost,
            )
            buf.add_r_from_others(total_reward_given_to_each_agent)
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next
    return list_buffers, done


def compute_fairness_ft(list_buffers, n_agents, eps, gamma):
    """Jain-style fairness loop from train_REFiNE_eia_er.py (timestep loop + F_T)."""
    n_steps = len(list_buffers[0].obs)
    fair_ts = []
    for t in range(n_steps):
        R = np.array(
            [
                list_buffers[i].reward[t]
                + np.sum(list_buffers[i].r_from_others[t][:, i])
                for i in range(n_agents)
            ]
        )
        num = R.sum()
        den = n_agents * np.sum(R ** 2) + eps
        f_t = (num * num) / den
        fair_ts.append((gamma ** t) * f_t)
    sum_weights = sum(gamma ** t for t in range(n_steps))
    F_T = sum(fair_ts) / sum_weights
    return F_T


def apply_room_er42(cfg):
    cfg.env.n_agents = 4
    cfg.env.min_at_lever = 2
    cfg.nn.n_h1 = 64
    cfg.nn.n_h2 = 32
    return cfg


def build_agents_lio(config, env):
    from lio.alg.lio_agent_er import LIO

    return [
        LIO(
            config.lio,
            env.l_obs,
            env.l_action,
            config.nn,
            "agent_%d" % i,
            config.env.r_multiplier,
            env.n_agents,
            i,
            1.0,
        )
        for i in range(env.n_agents)
    ]


def build_agents_refine(config, env):
    from lio.alg.REFiNE_er import REFiNE

    return [
        REFiNE(
            config.lio,
            env.l_obs,
            env.l_action,
            config.nn,
            "agent_%d" % i,
            config.env.r_multiplier,
            env.n_agents,
            i,
            1.0,
        )
        for i in range(env.n_agents)
    ]


def wire_graph(list_agents, config):
    for agent in list_agents:
        if config.lio.decentralized:
            agent.create_opp_modeling_op()
        else:
            agent.receive_list_of_agents(list_agents)
        agent.create_policy_gradient_op()
        agent.create_update_op()
        if getattr(config.lio, "use_actor_critic", False):
            agent.create_critic_train_op()
    for agent in list_agents:
        agent.create_reward_train_op()


def stats_row(method, stage, samples_ns):
    s = np.asarray(samples_ns, dtype=np.float64)
    n = s.size
    if n == 0:
        return [method, stage, 0, np.nan, np.nan, np.nan, np.nan, np.nan]
    us = s / 1000.0
    return [
        method,
        stage,
        n,
        float(np.mean(us)),
        float(np.median(us)),
        float(np.percentile(us, 95)),
        float(np.percentile(us, 99)),
        float(np.std(us)),
    ]


def welch_p(a_ns, b_ns):
    a = np.asarray(a_ns, dtype=np.float64) / 1000.0
    b = np.asarray(b_ns, dtype=np.float64) / 1000.0
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def hardware_blurb():
    lines = []
    tegra = "/etc/nv_tegra_release"
    if os.path.isfile(tegra):
        try:
            with open(tegra) as f:
                lines.append(f.read().strip().split("\n")[0])
        except OSError:
            pass
    jp = "JetPack 6.x (L4T R36 family)" if lines else "unknown JetPack"
    cuda = "unknown CUDA"
    try:
        import subprocess

        p = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if p.stdout:
            for ln in p.stdout.splitlines():
                if "release" in ln.lower():
                    cuda = ln.strip()
                    break
    except (OSError, subprocess.TimeoutExpired):
        pass
    return jp, cuda, lines[0] if lines else ""


def benchmark_a_b_npz(path_npz, results_dir):
    """Benchmarks A and B; writes inference + give_reward arrays (best-effort per method)."""
    out = {}
    EPS_ACTOR = 0.05

    # ----- LIO -----
    try:
        tf.reset_default_graph()
        np.random.seed(12345)
        tf.compat.v1.set_random_seed(12345)
        cfg = apply_room_er42(config_room_lio.get_config())
        env = room_symmetric.Env(cfg.env)
        agents = build_agents_lio(cfg, env)
        wire_graph(agents, cfg)
        list_obs = env.reset()
        obs0 = np.asarray(list_obs[0], dtype=np.float32)

        config_proto = tf.ConfigProto(allow_soft_placement=True)
        config_proto.device_count["GPU"] = 0
        sess = tf.Session(config=config_proto)
        sess.run(tf.global_variables_initializer())

        ag0 = agents[0]
        for _ in range(2000):
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=False)
        lio_normal = np.empty(10000, dtype=np.int64)
        for i in range(10000):
            t0 = time.perf_counter_ns()
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=False)
            lio_normal[i] = time.perf_counter_ns() - t0

        for _ in range(2000):
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=True)
        lio_prime = np.empty(10000, dtype=np.int64)
        for i in range(10000):
            t0 = time.perf_counter_ns()
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=True)
            lio_prime[i] = time.perf_counter_ns() - t0

        l_action = env.l_action
        rng = np.random.RandomState(12345)

        def rand_actions():
            return [int(rng.randint(0, l_action)) for _ in range(env.n_agents)]

        action_all = np.array(rand_actions(), dtype=np.int32)
        for _ in range(1000):
            ag0.give_reward(obs0, action_all, sess)
        lio_give = np.empty(5000, dtype=np.int64)
        for i in range(5000):
            action_all = np.array(rand_actions(), dtype=np.int32)
            t0 = time.perf_counter_ns()
            ag0.give_reward(obs0, action_all, sess)
            lio_give[i] = time.perf_counter_ns() - t0

        sess.close()
        out["lio_normal"] = lio_normal
        out["lio_prime"] = lio_prime
        out["lio_give_reward"] = lio_give
    except Exception:
        print("Benchmark A/B LIO failed:\n", traceback.format_exc())

    # ----- REFiNE -----
    try:
        tf.reset_default_graph()
        np.random.seed(12345)
        tf.compat.v1.set_random_seed(12345)
        cfg = apply_room_er42(config_room_REFiNE.get_config())
        env = room_symmetric.Env(cfg.env)
        agents = build_agents_refine(cfg, env)
        wire_graph(agents, cfg)
        list_obs = env.reset()
        obs0 = np.asarray(list_obs[0], dtype=np.float32)

        config_proto = tf.ConfigProto(allow_soft_placement=True)
        config_proto.device_count["GPU"] = 0
        sess = tf.Session(config=config_proto)
        sess.run(tf.global_variables_initializer())

        ag0 = agents[0]
        for _ in range(2000):
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=False)
        refine_normal = np.empty(10000, dtype=np.int64)
        for i in range(10000):
            t0 = time.perf_counter_ns()
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=False)
            refine_normal[i] = time.perf_counter_ns() - t0

        for _ in range(2000):
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=True)
        refine_prime = np.empty(10000, dtype=np.int64)
        for i in range(10000):
            t0 = time.perf_counter_ns()
            ag0.run_actor(obs0, sess, epsilon=EPS_ACTOR, prime=True)
            refine_prime[i] = time.perf_counter_ns() - t0

        l_action = env.l_action
        rng = np.random.RandomState(12345)

        def rand_actions():
            return [int(rng.randint(0, l_action)) for _ in range(env.n_agents)]

        action_all = np.array(rand_actions(), dtype=np.int32)
        for _ in range(1000):
            ag0.give_reward(obs0, action_all, sess)
        refine_give = np.empty(5000, dtype=np.int64)
        for i in range(5000):
            action_all = np.array(rand_actions(), dtype=np.int32)
            t0 = time.perf_counter_ns()
            ag0.give_reward(obs0, action_all, sess)
            refine_give[i] = time.perf_counter_ns() - t0

        sess.close()
        out["refine_normal"] = refine_normal
        out["refine_prime"] = refine_prime
        out["refine_give_reward"] = refine_give
    except Exception:
        print("Benchmark A/B REFiNE failed:\n", traceback.format_exc())

    if out:
        np.savez(path_npz, **out)
    return out


def training_episode_times(method, results_dir):
    """Benchmark C: 30 episodes; returns list of dicts with ns timings."""
    tf.reset_default_graph()
    np.random.seed(12345)
    tf.compat.v1.set_random_seed(12345)
    rows = []
    EPS = 0.05

    if method == "LIO":
        cfg = apply_room_er42(config_room_lio.get_config())
    else:
        cfg = apply_room_er42(config_room_REFiNE.get_config())
    env = room_symmetric.Env(cfg.env)
    if method == "LIO":
        agents = build_agents_lio(cfg, env)
    else:
        agents = build_agents_refine(cfg, env)
    wire_graph(agents, cfg)

    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.device_count["GPU"] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    n_agents = env.n_agents
    gamma = cfg.lio.gamma

    for ep in range(30):
        t_ep0 = time.perf_counter_ns()
        t0 = time.perf_counter_ns()
        buf0, _ = run_episode(sess, env, agents, EPS, prime=False)
        rollout_ns = time.perf_counter_ns() - t0

        fairness_ns = 0
        F_T = 0.0
        if method == "REFiNE":
            eps_fair = cfg.lio.eps
            t0 = time.perf_counter_ns()
            F_T = compute_fairness_ft(buf0, n_agents, eps_fair, gamma)
            fairness_ns = time.perf_counter_ns() - t0
        policy_ns = 0
        for agent in agents:
            t0 = time.perf_counter_ns()
            if method == "LIO":
                agent.update(sess, buf0[agent.agent_id], EPS)
            else:
                buf = buf0[agent.agent_id]
                agent.update(
                    sess,
                    buf,
                    EPS,
                    buf.total_energy,
                    F_T,
                )
            policy_ns += time.perf_counter_ns() - t0

        buf1, _ = run_episode(sess, env, agents, EPS, prime=True)

        reward_ns = 0
        for agent in agents:
            if not agent.can_give:
                continue
            t0 = time.perf_counter_ns()
            agent.train_reward(sess, buf0, buf1, EPS)
            reward_ns += time.perf_counter_ns() - t0

        for agent in agents:
            if not cfg.lio.decentralized:
                agent.update_main(sess)

        total_episode_ns = time.perf_counter_ns() - t_ep0

        rows.append(
            {
                "method": method,
                "episode_idx": ep,
                "rollout_ns": rollout_ns,
                "fairness_compute_ns": fairness_ns,
                "policy_update_ns": policy_ns,
                "reward_update_ns": reward_ns,
                "total_episode_ns": total_episode_ns,
            }
        )

    sess.close()
    return rows


def write_episode_csv(rows, path):
    fieldnames = [
        "method",
        "episode_idx",
        "rollout_ns",
        "fairness_compute_ns",
        "policy_update_ns",
        "reward_update_ns",
        "total_episode_ns",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summaries(npz_path, episode_csv, out_summary, out_paired, paper_path):
    d = np.load(npz_path)
    rows_sum = []

    def add_from_npz(method, key, stage):
        if key not in d.files:
            return
        rows_sum.append(stats_row(method, stage, d[key]))

    add_from_npz("LIO", "lio_normal", "inference_normal")
    add_from_npz("REFiNE", "refine_normal", "inference_normal")
    add_from_npz("LIO", "lio_prime", "inference_prime")
    add_from_npz("REFiNE", "refine_prime", "inference_prime")
    add_from_npz("LIO", "lio_give_reward", "give_reward")
    add_from_npz("REFiNE", "refine_give_reward", "give_reward")

    ep_by_method = {}
    if os.path.isfile(episode_csv):
        with open(episode_csv, newline="") as f:
            for row in csv.DictReader(f):
                ep_by_method.setdefault(row["method"], []).append(row)

    for method in ("LIO", "REFiNE"):
        if method not in ep_by_method:
            continue
        er = ep_by_method[method]
        for col, stage in [
            ("rollout_ns", "rollout"),
            ("fairness_compute_ns", "fairness_compute"),
            ("policy_update_ns", "policy_update"),
            ("reward_update_ns", "reward_update"),
            ("total_episode_ns", "total_episode"),
        ]:
            rows_sum.append(
                stats_row(method, stage, [float(r[col]) for r in er])
            )

    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "stage",
                "n_samples",
                "mean_us",
                "median_us",
                "p95_us",
                "p99_us",
                "std_us",
            ]
        )
        w.writerows(rows_sum)

    def mean_us_for(stage, m):
        for row in rows_sum:
            if row[0] == m and row[1] == stage:
                return row[3]
        return float("nan")

    paired_stages = [
        ("inference_normal", "lio_normal", "refine_normal"),
        ("inference_prime", "lio_prime", "refine_prime"),
        ("give_reward", "lio_give_reward", "refine_give_reward"),
    ]
    if "LIO" in ep_by_method and "REFiNE" in ep_by_method:
        paired_stages.extend(
            [
                ("rollout", None, None),
                ("fairness_compute", None, None),
                ("policy_update", None, None),
                ("reward_update", None, None),
                ("total_episode", None, None),
            ]
        )

    paired_rows = []
    for stage, k1, k2 in paired_stages:
        if k1 is not None:
            lio_m = mean_us_for(stage, "LIO")
            ref_m = mean_us_for(stage, "REFiNE")
            a_ns = d[k1] if k1 in d.files else np.array([])
            b_ns = d[k2] if k2 in d.files else np.array([])
            pv = welch_p(a_ns, b_ns)
        else:
            lio_m = mean_us_for(stage, "LIO")
            ref_m = mean_us_for(stage, "REFiNE")
            col = {
                "rollout": "rollout_ns",
                "fairness_compute": "fairness_compute_ns",
                "policy_update": "policy_update_ns",
                "reward_update": "reward_update_ns",
                "total_episode": "total_episode_ns",
            }[stage]
            a_ns = np.array(
                [float(r[col]) for r in ep_by_method["LIO"]],
                dtype=np.float64,
            )
            b_ns = np.array(
                [float(r[col]) for r in ep_by_method["REFiNE"]],
                dtype=np.float64,
            )
            try:
                pv = welch_p(a_ns, b_ns)
            except Exception:
                pv = float("nan")

        diff = ref_m - lio_m
        pct = 100.0 * diff / lio_m if np.isfinite(lio_m) and lio_m != 0 else float("nan")
        paired_rows.append([stage, lio_m, ref_m, diff, pct, pv])

    with open(out_paired, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stage",
                "lio_mean_us",
                "refine_mean_us",
                "diff_us",
                "diff_pct",
                "welch_t_pvalue",
            ]
        )
        w.writerows(paired_rows)

    jp, cuda, tegra_line = hardware_blurb()
    lio_inf_m = mean_us_for("inference_normal", "LIO")
    ref_inf_m = mean_us_for("inference_normal", "REFiNE")
    lio_inf_med = next(
        (r[4] for r in rows_sum if r[0] == "LIO" and r[1] == "inference_normal"),
        float("nan"),
    )
    ref_inf_med = next(
        (r[4] for r in rows_sum if r[0] == "REFiNE" and r[1] == "inference_normal"),
        float("nan"),
    )
    lio_p99 = next(
        (r[6] for r in rows_sum if r[0] == "LIO" and r[1] == "inference_normal"),
        float("nan"),
    )
    ref_p99 = next(
        (r[6] for r in rows_sum if r[0] == "REFiNE" and r[1] == "inference_normal"),
        float("nan"),
    )
    p_inf = (
        welch_p(d["lio_normal"], d["refine_normal"])
        if "lio_normal" in d.files and "refine_normal" in d.files
        else float("nan")
    )
    lio_gr = mean_us_for("give_reward", "LIO")
    ref_gr = mean_us_for("give_reward", "REFiNE")
    p_gr = (
        welch_p(d["lio_give_reward"], d["refine_give_reward"])
        if "lio_give_reward" in d.files and "refine_give_reward" in d.files
        else float("nan")
    )

    def mean_ms(stage, m):
        v = mean_us_for(stage, m)
        return v / 1000.0 if np.isfinite(v) else float("nan")

    lines = []
    lines.append("=== Q2 Empirical Summary ===")
    lines.append("Hardware: Orin Nano, %s, %s" % (jp, cuda))
    if tegra_line:
        lines.append("Tegra: %s" % tegra_line)
    lines.append(
        "Network: 64-32 actor MLP, N=4 agents, ER(4,2) observations (l_obs=15)"
    )
    lines.append("Inference (run_actor, deployment path, prime=False):")
    lines.append(
        "LIO:    mean = %.3f us, median = %.3f us, p99 = %.3f us  (n=10000)"
        % (lio_inf_m, lio_inf_med, lio_p99)
    )
    lines.append(
        "REFiNE: mean = %.3f us, median = %.3f us, p99 = %.3f us  (n=10000)"
        % (ref_inf_m, ref_inf_med, ref_p99)
    )
    lines.append(
        "Paired difference (mean): %.3f us (%.3f%%), Welch t-test p = %.4g"
        % ((ref_inf_m - lio_inf_m), 100.0 * (ref_inf_m - lio_inf_m) / lio_inf_m, p_inf)
    )
    lines.append("Incentive computation (give_reward):")
    lines.append("LIO:    mean = %.3f us  (n=5000)" % lio_gr)
    lines.append("REFiNE: mean = %.3f us  (n=5000)" % ref_gr)
    lines.append(
        "Paired difference (mean): %.3f us, Welch p = %.4g"
        % ((ref_gr - lio_gr), p_gr)
    )
    lines.append("Training episode breakdown (n=30 episodes, mean values):")
    lines.append(
        "Stage              | LIO (ms)  | REFiNE (ms) | delta (ms)"
    )
    for st in [
        "rollout",
        "fairness_compute",
        "policy_update",
        "reward_update",
        "total_episode",
    ]:
        lm = mean_ms(st, "LIO")
        rm = mean_ms(st, "REFiNE")
        lines.append(
            "%-18s | %9.4f | %11.4f | %10.4f"
            % (st, lm, rm, rm - lm)
        )

    dep_path = os.path.join(os.path.dirname(paper_path), "deployment_cadence.txt")
    dep_line = "Deployment cadence: see benchmark_results/deployment_cadence.txt"
    if os.path.isfile(dep_path):
        with open(dep_path) as f:
            dep_line = f.read().strip().replace("\n", " ")
    lines.append("Deployment cadence: %s" % dep_line)

    block = "\n".join(lines)
    print(block)
    with open(paper_path, "w") as f:
        f.write(block + "\n")


def stage3_deployment_cadence(results_dir):
    import subprocess

    dep_txt = os.path.join(results_dir, "deployment_cadence.txt")
    log_csv = None
    try:
        out = subprocess.check_output(
            ["find", "results/", "-name", "log.csv"],
            cwd=_REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        paths = [p.strip() for p in out.splitlines() if p.strip()]
        if paths:
            log_csv = os.path.join(_REPO, paths[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = []

    if not log_csv or not os.path.isfile(log_csv):
        msg = (
            "No results/*/log.csv found under repo root. "
            "No per-step wall-clock timestamps available; time the next real-world run manually."
        )
        with open(dep_txt, "w") as f:
            f.write(msg + "\n")
        return

    rows = []
    with open(log_csv, newline="") as f:
        r = csv.reader(f)
        for row in r:
            rows.append(row)
    header = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []

    def looks_epoch(s):
        try:
            v = float(s)
            return v > 1e9
        except ValueError:
            return False

    def looks_iso(s):
        return len(s) > 10 and ("-" in s or "T" in s)

    ts_cols = []
    for i, h in enumerate(header):
        hl = h.lower()
        if "time" in hl or "timestamp" in hl or "epoch" in hl or "wall" in hl:
            ts_cols.append(i)
        elif body and i < len(body[0]):
            if looks_epoch(body[0][i]) or looks_iso(body[0][i]):
                ts_cols.append(i)

    lines = []
    lines.append("log_csv: %s" % log_csv)
    lines.append("header: %s" % ",".join(header))
    lines.append("first_5_rows:")
    for row in rows[: min(6, len(rows))]:
        lines.append(",".join(row))
    lines.append("last_5_rows:")
    for row in rows[-5:]:
        lines.append(",".join(row))

    if not ts_cols:
        lines.append(
            "conclusion: No wall-clock timestamp column (epoch or ISO datetime). "
            "Typical log.csv uses episode/step counters only; cannot compute "
            "median per-step deployment duration from this file."
        )
    else:
        lines.append("timestamp_column_indices: %s" % ts_cols)
        lines.append(
            "conclusion: Timestamp-like columns detected; inspect schema manually "
            "before deriving per-step cadence."
        )

    with open(dep_txt, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    results_dir = os.path.join(_REPO, "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    npz_path = os.path.join(results_dir, "inference_latency_raw.npz")
    ep_csv = os.path.join(results_dir, "episode_breakdown.csv")
    summary_csv = os.path.join(results_dir, "summary.csv")
    paired_csv = os.path.join(results_dir, "paired_diff.csv")
    paper_path = os.path.join(results_dir, "paper_summary.txt")

    out_ab = benchmark_a_b_npz(npz_path, results_dir)
    ok_ab_lio = "lio_normal" in out_ab
    ok_ab_ref = "refine_normal" in out_ab
    ok_a = bool(out_ab) and os.path.isfile(npz_path)

    all_ep_rows = []
    ok_c_lio = ok_c_ref = False
    for method in ("LIO", "REFiNE"):
        try:
            tf.reset_default_graph()
            rows = training_episode_times(method, results_dir)
            all_ep_rows.extend(rows)
            if method == "LIO":
                ok_c_lio = True
            else:
                ok_c_ref = True
        except Exception:
            print("Benchmark C %s failed:\n" % method, traceback.format_exc())

    if all_ep_rows:
        write_episode_csv(all_ep_rows, ep_csv)

    stage3_deployment_cadence(results_dir)

    if ok_a and os.path.isfile(npz_path):
        try:
            write_summaries(npz_path, ep_csv, summary_csv, paired_csv, paper_path)
        except Exception:
            print("Summary generation failed:\n", traceback.format_exc())

    print(
        "\nStages: A/B LIO=%s REFiNE=%s, C LIO=%s REFiNE=%s"
        % (ok_ab_lio, ok_ab_ref, ok_c_lio, ok_c_ref)
    )
    print("Written under:", results_dir)


if __name__ == "__main__":
    main()
