"""Trains REFiNE agents on Escape Room game.
The 2nd agent is attacked by eia, w_{i,j} (\text{lever pulling}) = 2, w_{i,j} (\text{door opening}) = 0.2, let others favor lever pulling, hate door opening.


"""


from __future__ import division
from __future__ import print_function

import sys, os, csv
# Add greedy_agent_v1 path
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path_to_add)

import argparse
import json
import os
import random

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import lio.utils.util as util




from lio.alg import config_ipd_lio
from lio.alg import config_room_REFiNE
from lio.alg import evaluate
import inspect


from lio.env import ipd_wrapper
from lio.env import room_symmetric


from lola.envs.prisoners_dilemma import IteratedPrisonersDilemma




def train(config):

    seed = config.main.seed
    # np.random.seed(seed)
    # random.seed(seed)
    # tf.set_random_seed(seed)

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
    
    from lio.alg.REFiNE_er import REFiNE
    from lio.alg.REFiNE_eia_er import REFiNEExploitative as REFiNE_E

    # track total‐fairness per episode
    fairness_history = {}
    
    

    list_agents = []

    # First agent normal
    list_agents.append(REFiNE(config.lio, env.l_obs, env.l_action,config.nn, 'agent_0',config.env.r_multiplier, env.n_agents,0, 1.0))
    
    # Second agent exploitative
    list_agents.append(REFiNE_E(config.lio, env.l_obs, env.l_action,config.nn, 'agent_1',config.env.r_multiplier, env.n_agents,1, 1.0))
    
    for agent_id in range(2, env.n_agents):
        list_agents.append(REFiNE(config.lio, env.l_obs, env.l_action,
                               config.nn, 'agent_%d' % agent_id,
                               config.env.r_multiplier, env.n_agents,
                               agent_id, 1.0))
       

     


    # list_agents.append(LIO_G(config.lio, env.l_obs, env.l_action,
    #                         config.nn, 'agent_0',
    #                         config.env.r_multiplier, env.n_agents,
    #                         0))        

    # list_agents.append(LIO_G(config.lio, env.l_obs, env.l_action,
    #                         config.nn, 'agent_1',
    #                         config.env.r_multiplier, env.n_agents,
    #                         1))        



    ####
    
    # list_agents[0].can_give = False
    # list_agents[1].can_give = False

    ####




    # for agent_id in range(env.n_agents):
    #     if config.lio.decentralized:
    #         list_agents[agent_id].create_opp_modeling_op()
    #     else:
    #         list_agents[agent_id].receive_list_of_agents(list_agents)
    #     list_agents[agent_id].create_policy_gradient_op()
    #     list_agents[agent_id].create_update_op()
    #     if config.lio.use_actor_critic:
    #         list_agents[agent_id].create_critic_train_op()

    # a = list_agents.copy()
    # a.reverse()
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

    # This handles the special case of two asymmetric agents,
    # one of which is the reward-giver and the other is the recipient  
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
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'reward_env', 'n_lever', 'n_door',
                   'received', 'given', 'r-lever', 'r-start', 'r-door', 
                   'win_rate', 'total_energy',  'teamwork_fairness']
    elif config.env.name == 'ipd':
        list_suffix = ['given', 'received', 'reward_env',
                   'reward_total', 'teamwork_fairness']
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
        # policy gradient training episode

        list_buffers, mission_status = run_episode(sess, env, list_agents, epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)
        
        if config.lio.decentralized:
            for idx, agent in enumerate(list_agents):
                agent.train_opp_model(sess, list_buffers,
                                      epsilon)

        # compute total‐fairness F_T for this episode
        n_agents = env.n_agents
        n_steps  = len(list_buffers[0].obs)
        eps      = config.lio.eps
        fair_ts  = []
        F_multiplier = config.lio.Fairness_multiplier
        gamma = config.lio.gamma
         
           
        for t in range(n_steps):
            # R_i(t) = env‐reward + incentives from others
            # build R_i(t) for *every* agent i
            R = np.array([ list_buffers[i].reward[t] + np.sum(list_buffers[i].r_from_others[t][:,i])
                       for i in range(n_agents) ])
            num = R.sum()
            den = n_agents * np.sum(R**2) + eps
            f_t = (num * num) / den
            fair_ts.append((gamma**t)*f_t)
            
        # discounted-average normalization
        sum_weights = sum(gamma**t for t in range(n_steps))
        F_T = (sum(fair_ts) / sum_weights) 
        BF_T = F_multiplier * F_T   
        #print(f"Policy Training Episode {idx_episode}: total fairness = {F_T:.6f}")
        #print(f"Policy Training Episode {idx_episode}: enlarged total fairness = {BF_T:.6f}")
        fairness_history[idx_episode] = BF_T
        
        # now call update with energy & fairness, update every agent
        for agent in list_agents:
           buf = list_buffers[agent.agent_id]
           agent.update(sess,
                 buf,
                 epsilon,
                 buf.total_energy,
                 F_T)



        
        
        # incentive training episode

        list_buffers_new, mission_status = run_episode(sess, env, list_agents,
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)
        
        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers,
                                   list_buffers_new, epsilon)

        for idx, agent in enumerate(list_agents):
            if config.lio.decentralized:
                agent.train_opp_model(sess, list_buffers_new,
                                      epsilon)
            else:
                agent.update_main(sess)

        
        
        step_train += 1

        if idx_episode % period == 0:
            # print("episode",idx_episode)
            if config.env.name == 'er':
               
               (reward_total, rewards_env, n_move_lever, n_move_door, rewards_received,
                rewards_given, steps_per_episode, r_lever, r_start, r_door,
                win_rate, cumulative_energy, teamwork_fairness) = evaluate.test_room_symmetric(
                    n_eval, env, sess, list_agents, 'REFiNE')
               matrix_combined = np.stack([reward_total, rewards_env, n_move_lever, n_move_door,
                             rewards_received, rewards_given,
                             r_lever, r_start, r_door, win_rate,
                             cumulative_energy,  teamwork_fairness])
            elif config.env.name == 'ipd':
                (rewards_given, rewards_received, rewards_env,
                 rewards_total, teamwork_fairness) = evaluate.test_ipd(
                    n_eval, env, sess, list_agents)
                matrix_combined = np.stack([rewards_given, rewards_received, rewards_env,
                                  rewards_total, teamwork_fairness])

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

        

    saver.save(sess, os.path.join(log_path, model_name))


def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    list_energies = [0.0] * len(list_agents) # Initialize energy consumption

    done = 0

    while not done:
        list_actions = list(range(len(list_agents)))
        # TODO : make agent random
        # copy_list_agents = list_agents.copy()

        # random.shuffle(copy_list_agents) # random agent finishing it task earlier
        # copy_list_agents.reverse()
        
        for idx, agent in enumerate(list_agents):
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions[agent.agent_id] = action
            # print(agent.agent_id,idx)

            # Calculate energy cost for the action
            # energy_cost = agent.calculate_energy_cost(list_obs[agent.agent_id], action)
            # list_buffers[agent.agent_id].add([
                # list_obs[agent.agent_id],  # Current observation
                # action,                    # Action taken
                # 0,                         # Placeholder for reward (to be updated later)
                # list_obs[agent.agent_id],  # Placeholder for next observation (to be updated later)
                #False                      # Placeholder for done (to be updated later)
                #], energy_cost)
            
        

                

        list_rewards = list(range(len(list_agents)))
        total_reward_given_to_each_agent = np.zeros((env.n_agents,env.n_agents))
        # total_reward_given_to_each_agent = np.zeros(env.n_agents)
        # TODO: make agent random
        # random.shuffle(list_agents)
        for idx,agent in enumerate(list_agents):
            if agent.can_give: # here exchange happens
                reward = agent.give_reward(list_obs[agent.agent_id],
                                           list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            # total_reward_given_to_each_agent += reward
            total_reward_given_to_each_agent[idx] += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards[agent.agent_id] = reward

        # print(total_reward_given_to_each_agent)

        if env.name == 'er':
            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        elif env.name == 'ipd':
            list_obs_next, env_rewards, done = env.step(list_actions)

        # Update buffers with transitions
        for idx, buf in enumerate(list_buffers):
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
        self.energy_cost.append(energy)  # Store the energy cost
        self.total_energy += energy # Update total energy consumed by the agent

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, choices=['er', 'ipd'])
    parser.add_argument('num', type=int)

    
    args = parser.parse_args()

    if args.exp == 'er':
        config = config_room_REFiNE.get_config()
        # For ER(10,6) experiment
        n=10 # Number of agents in the Escape Room
        m=6 # Minimum number of agents required at lever to trigger outcome
        config.main.dir_name = 'er_REFiNE_attack_10_6'  # Directory for exploitative agent logs (2.0, 0.2)
        config.env.min_at_lever = m
        config.env.n_agents = n
        config.main.exp_name = 'er%d'%args.num
        # config.main.seed = 12340 + args.num
        # config.main.seed = random.random()

    train(config)
    print("set %d done"%args.num)