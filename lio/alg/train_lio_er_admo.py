"""Adaptive multi-objective trainer for Escape Room LIO.

This script mirrors `train_lio_er.py` while wrapping a selected agent
(the adversary) with an Adaptive Multi-Objective (ADMO) controller. The
controller blends incentive and policy manipulation losses using
Pareto-aware weights derived from gradient proxies, in line with the
design memo (Section~\ref{sec:MO}).

Source files such as `train_lio_er.py` remain untouched; the controller
is layered externally and reuses existing agent methods.
"""

from __future__ import division
from __future__ import print_function

import sys
import os

path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from lio.alg import config_room_lio
from lio.alg import evaluate
from lio.env import room_symmetric


@dataclass
class AdaptiveMOConfig:
    """Hyper-parameters for the Adaptive Multi-Objective controller."""

    mode: int = +1                     # +1 adversarial, -1 constructive
    c1: float = 0.1                    # lower bound on alpha_1
    c2: float = 0.1                    # lower bound on alpha_2
    beta_cost: float = 0.01            # incentive cost coefficient
    lambda_d: float = 0.1              # KL regularisation weight
    lambda_d_cooldown: float = 0.05    # fallback KL weight after spikes
    kl_threshold: float = 0.5          # trigger to reduce lambda_d
    ema_decay: float = 0.9             # smoothing factor for EMAs
    success_target: float = 0.6        # swap to constructive if exceeded
    enable_auto_mode: bool = False     # optional mode switching
    incentive_budget: float = math.inf # per-episode budget for incentives
    max_extra_policy_steps: int = 3
    max_extra_reward_steps: int = 3
    k_ref: int = 100                   # refresh period for reference policy
    max_grad_norm: float = 1.0         # clip gradient proxies


class MovingAverage:
    """Simple exponential moving average tracker."""

    def __init__(self, decay: float):
        self.decay = decay
        self.value = 0.0
        self.initialised = False

    def update(self, sample: float) -> float:
        if not self.initialised:
            self.value = sample
            self.initialised = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * sample
        return self.value


class AdaptiveMOController:
    """Pareto-aware controller layered on top of a single LIO agent."""

    def __init__(self, sess: tf.Session, agent, cfg: AdaptiveMOConfig,
                 gamma: float, n_agents: int, agent_id: int):
        self.sess = sess
        self.agent = agent
        self.cfg = cfg
        self.gamma = gamma
        self.n_agents = n_agents
        self.agent_id = agent_id

        self.alpha = np.array([0.5, 0.5], dtype=np.float32)
        self.ema_success = MovingAverage(cfg.ema_decay)
        self.ema_variance = MovingAverage(cfg.ema_decay)
        self.reference_params = self._snapshot_policy()
        self.current_budget = 0.0

        self.signals = {
            'returns_gap': 0.0,
            'team_welfare': 0.0,
            'inc_cost': 0.0,
            'variance': 0.0,
            'success': 0.0,
        }

    # ----------------------------- snapshots -----------------------------

    def _snapshot_policy(self) -> Dict[str, np.ndarray]:
        params = {}
        for var in self.agent.policy_params:
            params[var.name] = self.sess.run(var)
        return params

    def maybe_update_reference(self, iteration: int):
        if iteration % self.cfg.k_ref == 0 and iteration > 0:
            current = self._snapshot_policy()
            kl_proxy = self._estimate_parameter_kl(current)
            if kl_proxy > self.cfg.kl_threshold:
                self.cfg.lambda_d = self.cfg.lambda_d_cooldown
            self.reference_params = current

    def _estimate_parameter_kl(self, current_params: Dict[str, np.ndarray]) -> float:
        total = 0.0
        for name, ref_val in self.reference_params.items():
            cur_val = current_params[name]
            total += np.sum((cur_val - ref_val) ** 2)
        return 0.5 * total

    # ----------------------------- metrics ------------------------------

    def _discount_vector(self, length: int) -> np.ndarray:
        return np.power(self.gamma, np.arange(length, dtype=np.float32))

    def _episode_signals(self, list_buffers, mission_status: int):
        env_rewards = []
        inc_received = []
        inc_given = []
        for agent_idx, buf in enumerate(list_buffers):
            env = np.array(buf.reward, dtype=np.float32)
            inc_recv = []
            inc_give = []
            for matrix in buf.r_from_others:
                matrix = np.asarray(matrix, dtype=np.float32)
                inc_recv.append(np.sum(matrix[:, agent_idx]))
                inc_give.append(np.sum(matrix[agent_idx, :]))
            env_rewards.append(env)
            inc_received.append(np.array(inc_recv, dtype=np.float32))
            inc_given.append(np.array(inc_give, dtype=np.float32))

        total_returns = [env_rewards[i] + inc_received[i]
                         for i in range(self.n_agents)]

        max_len = max(len(r) for r in total_returns)
        discounts = self._discount_vector(max_len)

        adv_returns = total_returns[self.agent_id]
        others_returns = [total_returns[i] for i in range(self.n_agents)
                          if i != self.agent_id]
        mean_others = np.mean(np.stack(others_returns), axis=0)
        returns_gap = float(np.sum(discounts[:len(adv_returns)] *
                                   (adv_returns - mean_others)))

        welfare_series = np.sum(np.stack(total_returns), axis=0)
        team_welfare = float(np.sum(self._discount_vector(len(welfare_series)) *
                                    welfare_series))

        inc_cost = float(np.sum(discounts[:len(inc_given[self.agent_id])] *
                                np.abs(inc_given[self.agent_id])))

        variance = float(np.var(welfare_series))
        success_flag = int(team_welfare > 0.0 or mission_status)

        self.current_budget = inc_cost
        self.signals.update({
            'returns_gap': returns_gap,
            'team_welfare': team_welfare,
            'inc_cost': inc_cost,
            'variance': variance,
            'success': success_flag,
        })

        self.ema_success.update(success_flag)
        self.ema_variance.update(variance)

        if self.cfg.enable_auto_mode:
            if self.cfg.mode > 0 and self.ema_success.value > self.cfg.success_target:
                self.cfg.mode = -1
            elif self.cfg.mode < 0 and self.ema_success.value < 0.2:
                self.cfg.mode = +1

    # ---------------------------- Pareto weights ------------------------

    def solve_weights(self):
        returns_gap = self.signals['returns_gap']
        team_welfare = self.signals['team_welfare']
        inc_cost = self.signals['inc_cost']

        g_inc = np.array([returns_gap - self.cfg.beta_cost * inc_cost], dtype=np.float32)
        g_pol = np.array([-self.cfg.mode * team_welfare], dtype=np.float32)

        g_inc = np.clip(g_inc, -self.cfg.max_grad_norm, self.cfg.max_grad_norm)
        g_pol = np.clip(g_pol, -self.cfg.max_grad_norm, self.cfg.max_grad_norm)

        G = np.stack([g_inc, g_pol], axis=1)
        gram = G.T @ G
        e = np.ones(2, dtype=np.float32)
        c = np.array([self.cfg.c1, self.cfg.c2], dtype=np.float32)

        M = np.block([[gram, e[:, None]], [e[None, :], np.zeros((1, 1))]])
        rhs = np.concatenate([-gram @ c, [1 - np.dot(e, c)]])

        try:
            solution = np.linalg.solve(M + 1e-6 * np.eye(3), rhs)
            alpha_raw = solution[:2]
        except np.linalg.LinAlgError:
            alpha_raw = np.array([0.5, 0.5], dtype=np.float32)

        if not np.all(np.isfinite(alpha_raw)):
            alpha_raw = np.array([0.5, 0.5], dtype=np.float32)

        alpha_clipped = np.maximum(alpha_raw + c, c)
        denom = np.sum(alpha_clipped)
        if not np.isfinite(denom) or denom <= 0:
            self.alpha = np.array([0.5, 0.5], dtype=np.float32)
        else:
            self.alpha = (alpha_clipped / denom).astype(np.float32)

    # ---------------------------- update schedule ----------------------

    def apply_updates(self, sess, list_agents, list_buffers, list_buffers_new,
                      epsilon: float):
        policy_weight = float(self.alpha[1])
        inc_weight = float(self.alpha[0])

        n_policy_steps = max(1, int(round(policy_weight * self.cfg.max_extra_policy_steps)))
        n_reward_steps = max(1, int(round(inc_weight * self.cfg.max_extra_reward_steps)))

        if self.current_budget > self.cfg.incentive_budget:
            n_reward_steps = 0

        buf_adv = list_buffers[self.agent_id]

        for _ in range(n_policy_steps):
            self.agent.update(sess, buf_adv, epsilon)

        for _ in range(n_reward_steps):
            if self.agent.can_give:
                self.agent.train_reward(sess, list_buffers, list_buffers_new, epsilon)

        self.agent.update_main(sess)

    def step(self, iteration: int, sess, list_agents, list_buffers,
             list_buffers_new, mission_status: int, epsilon: float):
        self._episode_signals(list_buffers, mission_status)
        self.solve_weights()
        self.maybe_update_reference(iteration)
        self.apply_updates(sess, list_agents, list_buffers, list_buffers_new,
                           epsilon)


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

    done = False
    mission_status = 0

    while not done:
        list_actions = list(range(len(list_agents)))
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess, epsilon, prime)
            list_actions[agent.agent_id] = action

        list_rewards = list(range(len(list_agents)))
        total_reward_given = np.zeros((env.n_agents, env.n_agents))
        for idx, agent in enumerate(list_agents):
            if agent.can_give:
                reward = agent.give_reward(list_obs[agent.agent_id], list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given[idx] += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards[agent.agent_id] = reward

        list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        mission_status = int(done)

        for idx, buf in enumerate(list_buffers):
            energy_cost = list_agents[idx].calculate_energy_cost(list_obs[idx], list_actions[idx])
            buf.add([
                list_obs[idx],
                list_actions[idx],
                env_rewards[idx],
                list_obs_next[idx],
                done
            ], energy_cost)
            buf.add_r_from_others(total_reward_given)
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next

    return list_buffers, mission_status


def train(config, admo_cfg: AdaptiveMOConfig, adversary_idx: int = 0):
    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (epsilon - config.lio.epsilon_end) / max(1, config.lio.epsilon_div)

    env = room_symmetric.Env(config.env)

    if config.lio.decentralized:
        from lio_decentralized import LIO
    elif config.lio.use_actor_critic:
        from lio_ac import LIO
    else:
        from lio_agent import LIO

    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(LIO(config.lio, env.l_obs, env.l_action,
                               config.nn, 'agent_%d' % agent_id,
                               config.env.r_multiplier, env.n_agents,
                               agent_id, 1.0))

    for agent in list_agents:
        if config.lio.decentralized:
            agent.create_opp_modeling_op()
        else:
            agent.receive_list_of_agents(list_agents)
        agent.create_policy_gradient_op()
        agent.create_update_op()
        if config.lio.use_actor_critic:
            agent.create_critic_train_op()

    for agent in list_agents:
        agent.create_reward_train_op()

    if config.lio.asymmetric:
        assert config.env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(agent_id != config.lio.idx_recipient)

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0

    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    if config.lio.use_actor_critic:
        for agent in list_agents:
            sess.run(agent.list_initialize_v_ops)

    controller = AdaptiveMOController(sess, list_agents[adversary_idx],
                                      admo_cfg, config.lio.gamma,
                                      env.n_agents, adversary_idx)

    list_agent_meas = ['A%d_%s' % (agent_id, suffix)
                       for agent_id in range(1, env.n_agents + 1)
                       for suffix in ['reward_total', 'reward_env', 'n_lever', 'n_door',
                                      'received', 'given', 'r-lever', 'r-start', 'r-door',
                                      'win_rate', 'total_energy', 'reward_per_energy']]

    header = ['episode', 'step_train', 'step'] + list_agent_meas + [
        'alpha_inc', 'alpha_pol', 'returns_gap', 'team_welfare',
        'inc_cost', 'ema_success', 'ema_variance', 'steps_per_eps'
    ]

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(','.join(header) + '\n')

    step = 0
    step_train = 0

    for idx_episode in range(1, n_episodes + 1):
        list_buffers, mission_status = run_episode(sess, env, list_agents, epsilon, prime=False)
        step += len(list_buffers[0].obs)

        if config.lio.decentralized:
            for agent in list_agents:
                agent.train_opp_model(sess, list_buffers, epsilon)

        for agent in list_agents:
            agent.update(sess, list_buffers[agent.agent_id], epsilon)

        list_buffers_new, mission_status_prime = run_episode(sess, env, list_agents,
                                                             epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers, list_buffers_new, epsilon)

        # Synchronize prime -> main parameters for all agents before ADMO step
        for agent in list_agents:
            if config.lio.decentralized:
                agent.train_opp_model(sess, list_buffers_new, epsilon)
            else:
                agent.update_main(sess)

        controller.step(idx_episode, sess, list_agents, list_buffers,
                        list_buffers_new, mission_status or mission_status_prime,
                        epsilon)

        step_train += 1

        if idx_episode % period == 0:
            (reward_total, rewards_env, n_move_lever, n_move_door, rewards_received,
             rewards_given, steps_per_episode, r_lever, r_start, r_door,
             win_rate, cumulative_energy, reward_per_energy) = evaluate.test_room_symmetric(
                n_eval, env, sess, list_agents, 'lio')
            matrix_combined = np.stack([reward_total, rewards_env, n_move_lever, n_move_door,
                                        rewards_received, rewards_given,
                                        r_lever, r_start, r_door, win_rate,
                                        cumulative_energy, reward_per_energy])

            log_values = [idx_episode, step_train, step]
            for idx in range(env.n_agents):
                metrics = ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                           '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}').format(
                    *matrix_combined[:, idx])
                log_values.append(metrics)

            log_values.extend([
                f"{controller.alpha[0]:.4f}",
                f"{controller.alpha[1]:.4f}",
                f"{controller.signals['returns_gap']:.4f}",
                f"{controller.signals['team_welfare']:.4f}",
                f"{controller.signals['inc_cost']:.4f}",
                f"{controller.ema_success.value:.4f}",
                f"{controller.ema_variance.value:.4f}"
            ])

            log_values.append(f"{steps_per_episode:.2f}")

            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(','.join(map(str, log_values)) + '\n')

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d' % (model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive MO LIO training (Escape Room)')
    parser.add_argument('num', type=int, help='Experiment index for directory naming')
    parser.add_argument('--mode', choices=['adversarial', 'constructive'],
                        default='adversarial', help='Manipulation mode sign')
    parser.add_argument('--adversary', type=int, default=0,
                        help='Agent index controlled by ADMO')
    parser.add_argument('--auto-mode', action='store_true',
                        help='Enable automatic mode switching')
    parser.add_argument('--budget', type=float, default=float('inf'),
                        help='Incentive budget per episode')
    parser.add_argument('--n_agents', type=int, default=None,
                        help='Override number of agents in Escape Room')
    parser.add_argument('--min_at_lever', type=int, default=None,
                        help='Override minimum agents required at lever')

    args = parser.parse_args()

    config = config_room_lio.get_config()
    config.main.dir_name = 'er_lio_admo'
    config.main.exp_name = 'er_admo_%d_%d_trail_%d' % (args.n_agents, args.min_at_lever, args.num)

    if args.n_agents is not None:
        config.env.n_agents = args.n_agents
    if args.min_at_lever is not None:
        config.env.min_at_lever = args.min_at_lever

    mode_sign = +1 if args.mode == 'adversarial' else -1
    admo_cfg = AdaptiveMOConfig(mode=mode_sign,
                                enable_auto_mode=args.auto_mode,
                                incentive_budget=args.budget)

    train(config, admo_cfg, adversary_idx=args.adversary)
