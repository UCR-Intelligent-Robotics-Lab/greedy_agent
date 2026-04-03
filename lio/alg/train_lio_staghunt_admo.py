"""Adaptive multi-objective trainer for Iterated Prisoner's Dilemma LIO.

This script follows the baseline `train_lio_ipd.py` but augments a
chosen agent with the Adaptive Multi-Objective (ADMO) controller from
`train_lio_er_admo.py`. The controller rebalances incentive and policy
manipulation gradients using Pareto-aware weights while keeping the base
LIO implementation untouched.
"""

from __future__ import division
from __future__ import print_function

import sys
import os

path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)

import argparse
import json

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from lio.alg import config_staghunt_lio
from lio.alg import evaluate
from lio.env import staghunt

from lio.alg.train_lio_er_admo import (
    AdaptiveMOConfig,
    AdaptiveMOController,
    Buffer,
)

from lio.utils import util


def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()

    done = False
    mission_status = 0

    while not done:
        list_actions = list(range(len(list_agents)))
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess, epsilon, prime)
            if action < 0 or action >= env.l_action:
                action = int(action) % env.l_action
            list_actions[agent.agent_id] = int(action)

        list_rewards = [None] * len(list_agents)
        total_reward_given_to_each_agent = np.zeros((env.n_agents, env.n_agents))
        for idx, agent in enumerate(list_agents):
            if agent.can_give:
                safe_actions = [a % env.l_action for a in list_actions]
                action_others_1hot = util.get_action_others_1hot(safe_actions, idx, env.l_action)
                feed = {
                    agent.obs: np.array([list_obs[idx]]),
                    agent.action_others: np.array([action_others_1hot])
                }
                reward = sess.run(agent.reward_function, feed_dict=feed).flatten() * agent.r_multiplier
            else:
                reward = np.zeros(env.n_agents)
            reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            reward[idx] = 0
            total_reward_given_to_each_agent[idx] += reward
            list_rewards[idx] = np.delete(reward, idx)

        list_obs_next, env_rewards, done = env.step(list_actions)
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
            buf.add_r_from_others(total_reward_given_to_each_agent)
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

    env = staghunt.Env(config.env)

    if config.lio.decentralized:
        from lio.alg.lio_decentralized import LIO
    elif config.lio.use_actor_critic:
        from lio.alg.lio_ac import LIO
    else:
        from lio.alg.lio_agent import LIO

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
                       for suffix in ['given', 'received', 'reward_env',
                                      'reward_total', 'total_energy', 'reward_per_energy']]

    header = ['episode', 'step_train', 'step'] + list_agent_meas + [
        'alpha_inc', 'alpha_pol', 'returns_gap', 'team_welfare',
        'inc_cost', 'ema_success', 'ema_variance'
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
            (rewards_given, rewards_received, rewards_env,
             rewards_total, cumulative_energy, reward_per_energy) = evaluate.test_staghunt(
                n_eval, env, sess, list_agents)

            rewards_given_mean = np.nanmean(rewards_given, axis=0)
            rewards_received_mean = np.nanmean(rewards_received, axis=0)
            rewards_env_mean = np.nanmean(rewards_env, axis=0)
            rewards_total_mean = np.nanmean(rewards_total, axis=0)

            cumulative_energy = np.nan_to_num(cumulative_energy, nan=0.0, posinf=0.0, neginf=0.0)
            reward_per_energy = np.nan_to_num(reward_per_energy, nan=0.0, posinf=0.0, neginf=0.0)

            log_values = [idx_episode, step_train, step]
            for idx in range(env.n_agents):
                log_values.extend([
                    f"{np.nan_to_num(rewards_given_mean[idx], nan=0.0):.3e}",
                    f"{np.nan_to_num(rewards_received_mean[idx], nan=0.0):.3e}",
                    f"{np.nan_to_num(rewards_env_mean[idx], nan=0.0):.3e}",
                    f"{np.nan_to_num(rewards_total_mean[idx], nan=0.0):.3e}",
                    f"{cumulative_energy[idx]:.3e}",
                    f"{reward_per_energy[idx]:.3e}"
                ])

            alpha_safe = np.nan_to_num(controller.alpha, nan=0.5)
            if alpha_safe.sum() == 0:
                alpha_safe = np.array([0.5, 0.5], dtype=np.float32)
            else:
                alpha_safe = alpha_safe / alpha_safe.sum()

            log_values.extend([
                f"{alpha_safe[0]:.4f}",
                f"{alpha_safe[1]:.4f}",
                f"{np.nan_to_num(controller.signals.get('returns_gap', 0.0), nan=0.0):.4f}",
                f"{np.nan_to_num(controller.signals.get('team_welfare', 0.0), nan=0.0):.4f}",
                f"{np.nan_to_num(controller.signals.get('inc_cost', 0.0), nan=0.0):.4f}",
                f"{np.nan_to_num(controller.ema_success.value, nan=0.0):.4f}",
                f"{np.nan_to_num(controller.ema_variance.value, nan=0.0):.4f}"
            ])

            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(','.join(map(str, log_values)) + '\n')

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d' % (model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive MO LIO training (IPD)')
    parser.add_argument('num', type=int, help='Experiment index for directory naming')
    parser.add_argument('--mode', choices=['adversarial', 'constructive'],
                        default='adversarial', help='Manipulation mode sign')
    parser.add_argument('--adversary', type=int, default=0,
                        help='Agent index controlled by ADMO')
    parser.add_argument('--auto-mode', action='store_true',
                        help='Enable automatic mode switching')
    parser.add_argument('--budget', type=float, default=float('inf'),
                        help='Incentive budget per episode')
    
    parser.add_argument('--n_agents', type=int, default=2,
                        help='Number of agents in the IPD')

    args = parser.parse_args()

    config = config_staghunt_lio.get_config()
    config.main.dir_name = 'staghunt_lio_admo'
    config.main.exp_name = 'staghunt_admo_%d_trail_%d' % (args.n_agents, args.num)
    config.env.n_agents = args.n_agents

    mode_sign = +1 if args.mode == 'adversarial' else -1
    admo_cfg = AdaptiveMOConfig(mode=mode_sign,
                                enable_auto_mode=args.auto_mode,
                                incentive_budget=args.budget)

    train(config, admo_cfg, adversary_idx=args.adversary)
