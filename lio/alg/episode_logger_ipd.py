import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import json
import csv
import os


class EpisodeLoggerIPD:
    def __init__(self, n_agents, gamma, eps=1e-2):
        self.n_agents = n_agents
        self.steps = []
        self.step_data = []
        self.cumulative_rewards = np.zeros(n_agents)
        # IPD‐specific counters & fairness
        self.n_c     = np.zeros(n_agents, dtype=int)
        self.n_d     = np.zeros(n_agents, dtype=int)
        self.gamma   = gamma
        self.eps     = eps
        self.fair_ts = []
       
    def log_step(self, step_num: int, actions: List[int], env_rewards: List[float],
                incentives_matrix: List[List[float]]) -> None:
        """Log data for a single step."""
        # Calculate incentives received by each agent
        incentives_received = [0] * self.n_agents
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:  # Skip self-incentives
                   incentives_received[j] += incentives_matrix[i][j]
    
       # Update cumulative metrics
        for i in range(self.n_agents):
            self.cumulative_rewards[i] += env_rewards[i] + incentives_received[i]
            #self.cumulative_energy[i] += energy_costs[i]
            
        # count number of C/D 
        for i, a in enumerate(actions):
            if a == 0:       self.n_c[i] += 1
            elif a == 1:     self.n_d[i] += 1
        R = np.array([env_rewards[i] + incentives_received[i]
                      for i in range(self.n_agents)])
        num = R.sum()
        den = self.n_agents * (R**2).sum() + self.eps
        f_t = num*num / den
        self.fair_ts.append((self.gamma ** step_num) * f_t)
        
        # Store step data as a dictionary
        step = {
            'step': step_num,
            'actions': actions,
            'env_rewards': env_rewards,
            'incentives_given': incentives_matrix,
            'incentives_received': incentives_received,
            'total_rewards': self.cumulative_rewards.copy(),  
        }
        self.step_data.append(step)

        
    def save_to_file(self, filename):
        """Save detailed episode log to CSV file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Generate header row
        header = ['step']
        for i in range(self.n_agents):
            header.extend([
                f'A{i+1}_action',
                f'A{i+1}_env_reward'
            ])
            # Add columns for incentives given to each other agent
            for j in range(self.n_agents):
                if i != j:
                    header.append(f'A{i+1}_incentive_given_{j+1}')
            header.extend([
                f'A{i+1}_incentives_received',
                f'A{i+1}_total_reward',        # New cumulative reward column
            ])

        # Write data
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write each step's data
            for step in self.step_data:
                row = [step['step']]
                for i in range(self.n_agents):
                    # Add action and env reward
                    row.extend([
                        step['actions'][i],
                        step['env_rewards'][i]
                    ])
                    # Add incentives given to other agents
                    incentives = step['incentives_given'][i]
                    for j in range(self.n_agents):
                        if i != j:
                            row.append(incentives[j])
                    # Add incentives received, energy consumed, and cumulative totals
                    row.extend([
                        step['incentives_received'][i],
                        step['total_rewards'][i],           # Add cumulative reward
                    ])
                writer.writerow(row)

    
        

    
    

def run_and_log_episode(env, agents, sess):
    """Run a complete episode and log all relevant information."""
    logger = EpisodeLoggerIPD(len(agents), gamma=agents[0].gamma)
    list_obs = env.reset()
    done = False
    step = 0
    
    while not done:
        # Get actions from all agents
        list_actions = []
        for idx, agent in enumerate(agents):
            action = agent.run_actor(list_obs[agent.agent_id], sess, epsilon=0)
            list_actions.append(action)
            
        # Calculate incentives given by each agent
        list_rewards = [[] for _ in range(len(agents))]  # Initialize rewards list
        incentives_matrix = np.zeros((len(agents), len(agents)))
        for idx, agent in enumerate(agents):
            if agent.can_give:
                reward = agent.give_reward(list_obs[agent.agent_id], list_actions, sess)
                incentives_matrix[idx] = reward
                reward = np.delete(reward, agent.agent_id)  # Remove self-reward
                list_rewards[idx] = reward
            else:
                reward = np.zeros(len(agents))
                list_rewards[idx] = np.delete(reward, agent.agent_id)
                
        # Environment step with both actions and rewards
        if env.name == 'er':
            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        else:
            list_obs_next, env_rewards, done = env.step(list_actions)
        
        
            
        # Log step data
        logger.log_step(step, list_actions, env_rewards, 
                       incentives_matrix.tolist())
        
        list_obs = list_obs_next
        step += 1

   
        
    return logger

