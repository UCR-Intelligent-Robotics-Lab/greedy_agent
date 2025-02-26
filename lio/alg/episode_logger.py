import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import json
import csv
import os


class EpisodeLogger:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.steps = []
        self.step_data = []
        self.cumulative_rewards = np.zeros(n_agents)
        self.cumulative_energy = np.zeros(n_agents)

    def log_step(self, step_num: int, actions: List[int], env_rewards: List[float],
                incentives_matrix: List[List[float]], energy_costs: List[float]) -> None:
        """Log data for a single step."""
        # Calculate incentives received by each agent
        incentives_received = [sum(agent_col) for agent_col in zip(*incentives_matrix)]

        # Update cumulative metrics
        for i in range(self.n_agents):
            self.cumulative_rewards[i] += env_rewards[i] + incentives_received[i]
            self.cumulative_energy[i] += energy_costs[i]

        
        # Store step data as a dictionary
        step = {
            'step': step_num,
            'actions': actions,
            'env_rewards': env_rewards,
            'incentives_given': incentives_matrix,
            'incentives_received': incentives_received,
            'energy_consumed': energy_costs,
            'total_rewards': self.cumulative_rewards.copy(),  
            'total_consumed_energy': self.cumulative_energy.copy()
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
                f'A{i+1}_energy_consumed',
                f'A{i+1}_total_reward',        # New cumulative reward column
                f'A{i+1}_total_consumed_energy'  # New cumulative energy column
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
                        step['energy_consumed'][i],
                        step['total_rewards'][i],           # Add cumulative reward
                        step['total_consumed_energy'][i]    # Add cumulative energy
                    ])
                writer.writerow(row)

    def plot_cumulative_reward(self, save_dir=None):
        """Plot cumulative rewards over time."""
        
        
        steps = [step['step'] for step in self.step_data]
        
        # Plot cumulative rewards
        plt.figure(figsize=(10, 5))
        for i in range(self.n_agents):
            rewards = [step['total_rewards'][i] for step in self.step_data]
            plt.plot(steps, rewards, label=f'LIO Agent {i+1}{"(Exploitative)" if i+1==2 else ""}')
        plt.xlabel('Steps')
        plt.ylabel('Total Rewards')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(min(steps), max(steps)+1))
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'LIO_Exploitative_cumulative_rewards_ER42.png'))
        plt.close()
        

    def plot_cumulative_energy(self, save_dir=None):
        """Plot cumulative energy consumption over time."""
        
        
        steps = [step['step'] for step in self.step_data]
        
        
        # Plot cumulative energy
        plt.figure(figsize=(10, 5))
        for i in range(self.n_agents):
            energy = [step['total_consumed_energy'][i] for step in self.step_data]
            plt.plot(steps, energy, label=f'LIO Agent {i+1}{"(Exploitative)" if i+1==2 else ""}')
        plt.xlabel('Steps')
        plt.ylabel('Total Energy Consumed')
        plt.title('Cumulative Energy Consumption')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(min(steps), max(steps)+1))
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'LIO_Exploitative_cumulative_energy_cost_ER42.png'))
        plt.close()    

    

def run_and_log_episode(env, agents, sess):
    """Run a complete episode and log all relevant information."""
    logger = EpisodeLogger(len(agents))
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
        
        # Calculate energy costs
        energy_costs = []
        for idx, agent in enumerate(agents):
            energy_cost = agent.calculate_energy_cost(list_obs[idx], list_actions[idx])
            energy_costs.append(energy_cost)
            
        # Log step data
        logger.log_step(step, list_actions, env_rewards, 
                       incentives_matrix.tolist(), energy_costs)
        
        list_obs = list_obs_next
        step += 1


        
    return logger

