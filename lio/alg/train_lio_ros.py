"""Trains LIO agents on Escape Room game.

Three versions of LIO:
1. LIO built on top of policy gradient
2. LIO built on top of actor-critic
3. Fully decentralized version of LIO on top of policy gradient
"""

from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.path.abspath(os.path.join('../..')))


import argparse
import json
import random

import numpy as np
import tensorflow as tf


from lio.alg import config_ipd_lio
from lio.alg import config_room_lio
from lio.alg import evaluate
from lio.env import ipd_wrapper
from lio.env import room_symmetric

import rospy
from std_msgs.msg import String

def train(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div

    if config.env.name == 'er':
        env = room_symmetric.Env(config.env)
    elif config.env.name == 'ipd':
        env = ipd_wrapper.IPD(config.env)

    if config.lio.decentralized:
        from lio_decentralized import LIO
    elif config.lio.use_actor_critic:
        from lio_ac import LIO
    else:
        from lio_agent import LIO


    agent1 = LIO(config.lio, env.l_obs, env.l_action,
                            config.nn, 'agent_0',
                            config.env.r_multiplier, env.n_agents,0)
    agent2 = LIO(config.lio, env.l_obs, env.l_action,
                            config.nn, 'agent_1',
                            config.env.r_multiplier, env.n_agents,1)
                                       


    agent1.create_policy_gradient_op([agent2])#
    agent1.create_update_op()
    
    agent2.create_policy_gradient_op([agent1])#
    agent2.create_update_op()

    agent1.create_reward_train_op([agent2])#
    agent2.create_reward_train_op([agent1])#



    config_proto = tf.compat.v1.ConfigProto()

    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())



    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'reward_env', 'n_lever', 'n_door',
                       'received', 'given', 'r-lever', 'r-start', 'r-door', 
                       'win_rate', 'total_energy', 'reward_per_energy']
    elif config.env.name == 'ipd':
        list_suffix = ['given', 'received', 'reward_env',
                       'reward_total', 'total_energy', 'reward_per_energy']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'
    else:
        header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)    

    step = 0
    step_train = 0
    for idx_episode in range(1, n_episodes + 1):
        print("episode",idx_episode)
        pub.publish("%d"%idx_episode)
        list_buffers = run_episode(sess, env, [agent1,agent2], epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)

            
        agent1.update(sess, list_buffers[0], epsilon)
        agent2.update(sess, list_buffers[1], epsilon)


        list_buffers_new = run_episode(sess, env, [agent1,agent2],
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        
        if agent1.can_give:
            agent1.train_reward(sess, list_buffers,
                                list_buffers_new, epsilon, [agent2])#
        
        if agent2.can_give:
            agent2.train_reward(sess, list_buffers,
                               list_buffers_new, epsilon, [agent1])#

        agent1.update_main(sess)
        agent2.update_main(sess)

        step_train += 1

        if idx_episode % period == 0:

            if config.env.name == 'er':
              (reward_total, rewards_env, n_move_lever, n_move_door, rewards_received,
                rewards_given, steps_per_episode, r_lever, r_start, r_door,
                win_rate, total_energy, reward_per_energy) = evaluate.test_room_symmetric(
                n_eval, env, sess, list_agents)
              matrix_combined = np.stack([reward_total, rewards_env, n_move_lever, n_move_door,
                             rewards_received, rewards_given,
                             r_lever, r_start, r_door, win_rate,
                             total_energy, reward_per_energy])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                if config.env.name == 'er':
                    s += ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                          '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}').format(
                           *matrix_combined[:, idx])
                elif config.env.name == 'ipd':
                    s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(
                        *matrix_combined[:, idx])
            if config.env.name == 'er':
                s += ',%.2f\n' % steps_per_episode
            else:
                s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

        # Calculate Total Energy and Average Reward per Energy at the end of the episode
        for agent_id, buf in enumerate(list_buffers):
            total_energy = buf.total_energy
            env_reward = sum(buf.reward)  # Only environmental rewards
            reward_per_energy = env_reward / total_energy if total_energy > 0 else 0

             

    saver.save(sess, os.path.join(log_path, model_name))
    

def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    # print(list_obs)
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions.append(action)

            
        list_rewards = []
        total_reward_given_to_each_agent = np.zeros(env.n_agents)
        for agent in list_agents:
            if agent.can_give: # here exchange happens
                reward = agent.give_reward(list_obs[agent.agent_id],
                                           list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards.append(reward)

        if env.name == 'er':
            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        elif env.name == 'ipd':
            list_obs_next, env_rewards, done = env.step(list_actions)

        for idx, buf in enumerate(list_buffers):
            energy_cost = list_agents[idx].calculate_energy_cost(list_obs[idx], list_actions[idx])
            buf.add([list_obs[idx], list_actions[idx], env_rewards[idx],
                list_obs_next[idx], done], energy_cost)
            buf.add_r_from_others(total_reward_given_to_each_agent[idx])
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next

    return list_buffers


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
        self.energy_cost = []  # Stores energy costs per step
        self.total_energy = 0  # Stores total energy consumed by the agent

    def add(self, transition, energy):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        self.energy_cost.append(energy)
        self.total_energy += energy  # Accumulate energy consumption

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, choices=['er', 'ipd'])
    args = parser.parse_args()

    if args.exp == 'er':
        config = config_room_lio.get_config()
    elif args.exp == 'ipd':
        config = config_ipd_lio.get_config()


    pub = rospy.Publisher('sim/agent_data', String, queue_size=10)
    rospy.init_node('sim', anonymous=False)

    train(config)
