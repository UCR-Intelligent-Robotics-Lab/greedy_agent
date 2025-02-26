"""Trains Meta-LIO-Fair agents with hyperparameter optimization."""

from __future__ import division
from __future__ import print_function

import sys
import os
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)

import argparse
import json
import random
import numpy as np
import tensorflow as tf
import optuna

from lio.alg import config_ipd_lio_meta_mse
from lio.alg import config_room_lio_meta_mse
from lio.alg import evaluate
from lio.env import ipd_wrapper
from lio.env import room_symmetric
from lio.alg.lio_meta_mse import MetaLIOMSE
# from lio.alg.lio_meta_mse_badenergy import MetaLIOMSEBadEnergy
from lio.alg.lio_meta_mse_greedy import greedy, adversarial
from lio_meta_mse_greedy import MetaLIOMSE as MetaLIOMSE_G
from lio_meta_mse_exploitative import ExploitativeMetaLIOMSE as MetaLIOMSE_E


def train(config):
    """Main training loop with energy fairness considerations."""
    seed = config.main.seed
    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Save config
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.lio.epsilon_start
    epsilon_step = (epsilon - config.lio.epsilon_end) / config.lio.epsilon_div

    # Initialize environment
    if config.env.name == 'er':
        env = room_symmetric.Env(config.env)
    elif config.env.name == 'ipd':
        env = ipd_wrapper.IPD(config.env)

    # Initialize agents
    list_agents = []

    
    for agent_id in range(env.n_agents):
        agent = MetaLIOMSE(
            config.lio, env.l_obs, env.l_action,
            config.nn, f'agent_{agent_id}',
            config.env.r_multiplier, env.n_agents,
            agent_id, energy_param=1.0)
        list_agents.append(agent)

    # Set up networks and connections
    for agent in list_agents:
        if config.lio.decentralized:
            agent.create_opp_modeling_op()
        else:
            agent.receive_list_of_agents(list_agents)
        agent.create_energy_networks()    
        agent.create_policy_gradient_op()
        agent.create_meta_objective()
        agent.create_update_op()

    for agent in list_agents:
        agent.create_reward_train_op()

    # Handle asymmetric agents
    if config.lio.asymmetric:
        assert env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(
                agent_id != config.lio.idx_recipient)

    # TensorFlow configuration
    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    # Tracking metrics
    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'reward_env', 'n_lever', 'n_door',
                      'received', 'given', 'r-lever', 'r-start', 'r-door',
                      'win_rate', 'total_energy', 'reward_per_energy']
    elif config.env.name == 'ipd':
        list_suffix = ['given', 'received', 'reward_env', 'reward_total',
                      'total_energy', 'reward_per_energy']

    
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))


    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    # Create log file
    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'
    else:
        header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    # Training loop
    step = 0
    step_train = 0
    best_reward_per_energy = float('-inf')

    for idx_episode in range(1, n_episodes + 1):
        # Generate initial trajectory
        list_agent_meas.append('A%d_%s' % (agent_id, suffix))

        # Standard LIO update
        list_buffers, mission_status = run_episode(
            sess, env, list_agents, epsilon, prime=False)
        step += len(list_buffers[0].obs)
         # Print diagnostics every 1000 episodes
        if idx_episode % 1000 == 0:
            # print(f"\nEpisode {idx_episode}")
            for agent_id, buf in enumerate(list_buffers):
                total_reward = sum(buf.reward)
                # print(f"Agent {agent_id} - Total Reward: {total_reward:.3f}")
                # Print a few actions to see if they vary
                # print(f"Agent {agent_id} - First 5 actions: {buf.action[:5]}")

        if config.lio.decentralized:
            for idx, agent in enumerate(list_agents):
                agent.train_opp_model(sess, list_buffers, epsilon)

        # Meta update
        for idx, agent in enumerate(list_agents):
            agent.update(sess, list_buffers[agent.agent_id], epsilon) 

        # Calculate average energy metrics
        avg_energy = np.mean([buf.total_energy for buf in list_buffers])
        rewards = [sum(buf.reward) for buf in list_buffers]
        avg_reward_per_energy = np.mean([r/e if e > 0 else 0 
                                       for r, e in zip(rewards, [buf.total_energy for buf in list_buffers])])

       
        # Generate trajectory after policy update
        list_buffers_new, _ = run_episode(sess, env, list_agents, epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        # Energy update
        energy_losses = []
        for agent in list_agents:
            energy_loss = agent.update_energy(
                sess, list_buffers_new[agent.agent_id],
                avg_energy, avg_reward_per_energy)
            energy_losses.append(energy_loss)

        # Generate trajectory after energy update
        list_buffers_energy, _ = run_episode(sess, env, list_agents, epsilon, prime=True)

        # Train reward functions
        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(
                    sess, list_buffers_energy,
                    list_buffers_new, epsilon)

        # Final updates
        for agent in list_agents:
            if config.lio.decentralized:
                agent.train_opp_model(sess, list_buffers_new, epsilon)
            else:
                agent.update_main(sess)


        step_train += 1

        # Periodic evaluation
        if idx_episode % period == 0:
            if config.env.name == 'er':
                (reward_total, rewards_env, n_move_lever, n_move_door, rewards_received,
                rewards_given, steps_per_episode, r_lever, r_start, r_door,
                win_rate, total_energy, reward_per_energy) = evaluate.test_room_symmetric(
                    n_eval, env, sess, list_agents, 'meta-lio-mse')
                matrix_combined = np.stack([reward_total, rewards_env, n_move_lever, n_move_door,
                             rewards_received, rewards_given,
                             r_lever, r_start, r_door, win_rate,
                             total_energy, reward_per_energy])
                
            elif config.env.name == 'ipd':
                (rewards_given, rewards_received, rewards_env,
                 rewards_total, total_energy, reward_per_energy) = evaluate.test_ipd(
                    n_eval, env, sess, list_agents, 'meta-lio-mse')     
                matrix_combined = np.stack([
                    rewards_given, rewards_received, rewards_env,
                    rewards_total, total_energy, reward_per_energy
                ])

            # Log results
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                if config.env.name == 'er':
                    s += ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                         '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}').format(
                        *matrix_combined[:, idx])
                else:
                    s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(
                        *matrix_combined[:, idx])

            if config.env.name == 'er':
                s += f',{steps_per_episode:.2f}\n'
            else:
                s += '\n'

            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        # Save model
        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        # Update epsilon
        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

         # Calculate Total Energy and Average Reward per Energy at the end of the episode
        for agent_id, buf in enumerate(list_buffers):
            total_energy = buf.total_energy
            env_reward = sum(buf.reward)  # Only environmental rewards
            reward_per_energy = env_reward / total_energy if total_energy > 0 else 0    

    # Final save
    saver.save(sess, os.path.join(log_path, model_name))
    














def run_episode(sess, env, list_agents, epsilon, prime=False):
    """Run single episode with energy tracking."""
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    list_energies = [0.0] * len(list_agents)
    
    done = 0
    while not done:
        list_actions = []
        
        # Get actions and energy costs
        for agent in list_agents:
            action = agent.run_actor(
                list_obs[agent.agent_id], sess, epsilon, prime)
            list_actions.append(action)
            
        
        # Calculate rewards
        list_rewards = list(range(len(list_agents)))
        total_reward_given_to_each_agent = np.zeros((env.n_agents,env.n_agents))
        
        for idx, agent in enumerate(list_agents):
            if agent.can_give:
                reward = agent.give_reward(
                    list_obs[agent.agent_id], list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent[idx] += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards[agent.agent_id] = reward

        # Environment step
        if env.name == 'er':
            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        else:
            list_obs_next, env_rewards, done = env.step(list_actions)

        # Update buffers
        for idx, buf in enumerate(list_buffers):
            # In run_episode - with NumPy inputs
            energy_cost = list_agents[idx].calculate_energy_cost(list_obs[idx], list_actions[idx])
            buf.add([
                list_obs[idx],  # Current observation
                list_actions[idx],  # Action taken
                env_rewards[idx],  # Reward received from environment
                list_obs_next[idx],  # Next observation
                done  # Whether the episode is done
            ], energy_cost)  # Add energy consumption to buffer
            buf.add_r_from_others(total_reward_given_to_each_agent)
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next

    return list_buffers, done

class Buffer(object):
    """Buffer class with energy tracking."""
    
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
        self.energy_costs = []
        self.total_energy = 0


    def add(self, transition, energy):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        self.energy_costs.append(energy)
        self.total_energy += energy

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        """Add actions from all agents for current timestep.
        
        Args:
            list_actions: List of actions, one per agent
        """
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, choices=['er', 'ipd'])
    parser.add_argument('num', type=int)
    args = parser.parse_args()

    if args.exp == 'er':
        config = config_room_lio_meta_mse.get_config()
        # For ER(4,2) experiment
        n=4 # Number of agents in the Escape Room
        m=2 # Minimum number of agents required at lever to trigger outcome
        config.main.dir_name = 'LIO_Meta_MSE_test_ER42' # Directory for Exploitative agent logs
        config.env.min_at_lever = m
        config.env.n_agents = n
        config.main.exp_name = 'er%d' % args.num
    elif args.exp == 'ipd':
        config = config_ipd_lio_meta_mse.get_config()
        config.main.dir_name = 'meta_ipd_bignum2'
        config.main.exp_name = 'ipd%d' % args.num
        config.main.seed = 12340 + args.num

    train(config)
    print("Set %d done" % args.num)