"""Trains REFiNE agents on TEAMGrid-Switch-v0 using a minimal wrapper."""
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse
import random

import numpy as np
import tensorflow as tf

# Add greedy_agent_v1 path
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)

from lio.alg import config_room_REFiNE
from lio.env import teamgrid_switch_env
from lio.alg.REFiNE_teamgrid import REFiNE


def dry_run_env(env, n_steps):
    obs_n = env.reset()
    print("dry_run obs_n length:", len(obs_n))
    print("dry_run obs[0] shape:", np.asarray(obs_n[0]).shape)
    print("dry_run obs_dim:", env.l_obs)
    print("dry_run action_dim:", env.l_action)

    for step_idx in range(n_steps):
        action_n = [np.random.randint(env.l_action) for _ in range(env.n_agents)]
        obs_n, reward_n, done, info = env.step(action_n)
        print("dry_run step:", step_idx,
              "reward_n type:", type(reward_n),
              "reward_n sample:", reward_n)
        print("dry_run done type:", type(done), "done:", done)
        if done:
            break


def compute_fairness(list_buffers, n_agents, gamma, eps):
    n_steps = len(list_buffers[0].obs)
    if n_steps == 0:
        return 0.0
    fair_ts = []
    for t in range(n_steps):
        R = np.array([
            list_buffers[i].reward[t] + np.sum(list_buffers[i].r_from_others[t][:, i])
            for i in range(n_agents)
        ])
        num = R.sum()
        den = n_agents * np.sum(R ** 2) + eps
        f_t = (num * num) / den
        fair_ts.append((gamma ** t) * f_t)
    sum_weights = sum(gamma ** t for t in range(n_steps))
    return sum(fair_ts) / sum_weights


def compute_episode_metrics(list_buffers, n_agents, F_T):
    rewards_env = []
    rewards_total = []
    total_energy = []
    for agent_id in range(n_agents):
        buf = list_buffers[agent_id]
        rewards_env.append(np.sum(buf.reward))
        sum_r_from_other = []
        for reward in buf.r_from_others:
            temp = np.sum(reward, axis=0, keepdims=False)
            sum_r_from_other.append(temp[agent_id])
        if len(sum_r_from_other) < len(buf.reward):
            padding = [0] * (len(buf.reward) - len(sum_r_from_other))
            sum_r_from_other.extend(padding)
        rewards_total.append(np.sum(buf.reward) + np.sum(sum_r_from_other))
        total_energy.append(buf.total_energy)

    teamwork_fairness = [F_T for _ in range(n_agents)]
    matrix = np.stack([rewards_total, rewards_env, total_energy, teamwork_fairness])
    return matrix


def train(config, env, log_path):
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div

    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(
            REFiNE(config.lio, env.l_obs, env.l_action, config.nn,
                   'agent_%d' % agent_id, config.env.r_multiplier,
                   env.n_agents, agent_id, 1.0)
        )

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
            list_agents[agent_id].set_can_give(
                agent_id != config.lio.idx_recipient)

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

    list_agent_meas = []
    list_suffix = ['reward_total', 'reward_env', 'total_energy', 'teamwork_fairness']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0

    for idx_episode in range(1, n_episodes + 1):
        list_buffers, _ = run_episode(sess, env, list_agents, epsilon,
                                      prime=False)
        step += len(list_buffers[0].obs)

        if config.lio.decentralized:
            for agent in list_agents:
                agent.train_opp_model(sess, list_buffers, epsilon)

        n_agents = env.n_agents
        eps = config.lio.eps
        gamma = config.lio.gamma
        F_T = compute_fairness(list_buffers, n_agents, gamma, eps)

        for agent in list_agents:
            buf = list_buffers[agent.agent_id]
            agent.update(sess, buf, epsilon, buf.total_energy, F_T)

        list_buffers_new, _ = run_episode(sess, env, list_agents,
                                          epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers,
                                   list_buffers_new, epsilon)

        for agent in list_agents:
            if config.lio.decentralized:
                agent.train_opp_model(sess, list_buffers_new, epsilon)
            else:
                agent.update_main(sess)

        step_train += 1

        if idx_episode % period == 0:
            matrix_combined = compute_episode_metrics(
                list_buffers, env.n_agents, F_T)
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ',{:.3e},{:.3e},{:.3e},{:.3e}'.format(
                    *matrix_combined[:, idx])
            s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d' % (
                model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))


def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()

    done = False

    while not done:
        list_actions = list(range(len(list_agents)))

        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions[agent.agent_id] = action

        list_rewards = list(range(len(list_agents)))
        total_reward_given_to_each_agent = np.zeros((env.n_agents, env.n_agents))
        for idx, agent in enumerate(list_agents):
            if agent.can_give:
                reward = agent.give_reward(list_obs[agent.agent_id],
                                           list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent[idx] += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards[agent.agent_id] = reward

        list_obs_next, env_rewards, done, _ = env.step(list_actions)
        if isinstance(done, (list, tuple, np.ndarray)):
            done_flag = all(done)
        else:
            done_flag = bool(done)

        for idx, buf in enumerate(list_buffers):
            energy_cost = list_agents[idx].calculate_energy_cost(
                list_obs[idx], list_actions[idx])
            buf.add([
                list_obs[idx],
                list_actions[idx],
                env_rewards[idx],
                list_obs_next[idx],
                done_flag
            ], energy_cost)
            buf.add_r_from_others(total_reward_given_to_each_agent)
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next
        done = done_flag

    return list_buffers, done


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--dry_steps', type=int, default=5)
    args = parser.parse_args()

    config = config_room_REFiNE.get_config()
    config.env.name = 'teamgrid'
    config.env.n_agents = 2
    config.env.action_dim = 4
    config.env.obs_dim = 147
    config.main.seed = args.seed
    config.alg.n_episodes = args.episodes
    config.alg.period = args.period
    config.main.exp_name = 'teamgrid_switch'
    config.main.dir_name = 'refine_seed%d' % args.seed

    env = teamgrid_switch_env.Env(config.env)
    config.env.n_agents = env.n_agents
    config.env.action_dim = env.l_action
    config.env.obs_dim = env.l_obs

    if args.logdir is None:
        args.logdir = os.path.join('lio', 'results', 'teamgrid_switch',
                                   'refine_seed%d' % args.seed)
    log_path = args.logdir

    if args.dry_run:
        dry_run_env(env, args.dry_steps)
        sys.exit(0)

    train(config, env, log_path)
