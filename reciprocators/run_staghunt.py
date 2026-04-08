import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agents.reciprocator import Reciprocator
from src.agents.naive_learner import NaiveLearner
from src.agents.voi import ValueOfInfluence

class GridworldStagHunt:
    def __init__(self, device, grid_size=5, max_steps=50):
        self.device = device
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_agents = 2
        self.l_obs = 10
        self.l_action = 5
        self.reset()

    def reset(self):
        self.step_count = 0
        self.state = np.random.randint(0, self.grid_size, size=(10,))
        return self._get_obs()

    def _get_obs(self):
        obs = self.state.astype(np.float32) / self.grid_size
        # shape: (num_agents, bsz=1, l_obs)
        return torch.tensor(obs, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_agents, 1, 1)

    def step(self, actions_tensor):
        # actions_tensor shape: (1, num_agents)
        actions = actions_tensor[0].cpu().numpy()
        self.step_count += 1
        
        for i in range(2):
            if actions[i] == 0: self.state[2*i+1] -= 1 # Up
            elif actions[i] == 1: self.state[2*i+1] += 1 # Down
            elif actions[i] == 2: self.state[2*i] -= 1 # Left
            elif actions[i] == 3: self.state[2*i] += 1 # Right
            
            self.state[2*i] = np.clip(self.state[2*i], 0, self.grid_size-1)
            self.state[2*i+1] = np.clip(self.state[2*i+1], 0, self.grid_size-1)

        # Random Stag move
        if np.random.rand() < 0.5:
            stag_a = np.random.randint(0, 5)
            if stag_a == 0: self.state[5] -= 1
            elif stag_a == 1: self.state[5] += 1
            elif stag_a == 2: self.state[4] -= 1
            elif stag_a == 3: self.state[4] += 1
            self.state[4] = np.clip(self.state[4], 0, self.grid_size-1)
            self.state[5] = np.clip(self.state[5], 0, self.grid_size-1)
            
        rewards = np.zeros(2)
        
        a1_pos = self.state[0:2]
        a2_pos = self.state[2:4]
        stag_pos = self.state[4:6]
        h1_pos = self.state[6:8]
        h2_pos = self.state[8:10]

        if np.array_equal(a1_pos, stag_pos) and np.array_equal(a2_pos, stag_pos):
            rewards[0] += 5.0
            rewards[1] += 5.0
            self.state[4:6] = np.random.randint(0, self.grid_size, size=(2,))
        else:
            if np.array_equal(a1_pos, h1_pos):
                rewards[0] += 1.0
                self.state[6:8] = np.random.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a1_pos, h2_pos):
                rewards[0] += 1.0
                self.state[8:10] = np.random.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a2_pos, h1_pos):
                rewards[1] += 1.0
                self.state[6:8] = np.random.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a2_pos, h2_pos):
                rewards[1] += 1.0
                self.state[8:10] = np.random.randint(0, self.grid_size, size=(2,))

        done = self.step_count >= self.max_steps
        obs_tensor = self._get_obs()
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float).unsqueeze(0)
        done_tensor = torch.tensor([done], device=self.device, dtype=torch.bool)
        
        return obs_tensor, rewards_tensor, done_tensor, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    env = GridworldStagHunt(device)

    influence_kwargs = {
        'num_players': env.num_agents,
        'state_dim': (env.l_obs,),
        'gamma': 0.99,
        'device': device,
        'rnn': False,
        'n_latent_var': 64,
        'lr': 1e-3,
        'target_period': 1,
        'target_batch_size': 16,
        'num_train_batches': 4
    }
    influence_estimator = ValueOfInfluence(**influence_kwargs)

    agents = []
    for i in range(env.num_agents):
        reciprocator_kwargs = {
            'state_dim': (env.l_obs,),
            'num_actions': env.l_action,
            'n_latent_var': 64,
            'lr': 1e-4,
            'gamma': 0.99,
            'K_epochs': 10,
            'eps_clip': 0.1,
            'entropy_weight': 0.01,
            'num_agents': env.num_agents,
            'player_idx': i,
            'influence_estimator': influence_estimator,
            'reciprocal_reward_weight': 5.0,
            'reciprocal_reward_type': 'petty',
            'device': device,
            'normalize_reciprocal_reward': False,
            'rnn': False
        }
        agents.append(Reciprocator(**reciprocator_kwargs))

    from lio.alg import config_staghunt_lio
    config = config_staghunt_lio.get_config()
    
    from evaluate import test_staghunt
    n_eval = config.alg.n_eval
    period = config.alg.period

    exp_name = f"staghunt{args.num}"
    dir_name = 'staghunt_reciprocators'
    log_path = os.path.join(os.path.dirname(__file__), '..', 'lio', 'results', exp_name, dir_name)
    os.makedirs(log_path, exist_ok=True)

    list_agent_meas = []
    list_suffix = ['given', 'received', 'reward_env', 'reward_total', 'total_energy', 'reward_per_energy']
    for agent_id in range(1, env.num_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))
            
    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += '\n'
    
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    print("Starting training on Spatial Stag Hunt...")
    episode_count = 0
    step_train = 0
    step = 0
    last_obs = env.reset()

    for episode in tqdm(range(1, args.episodes + 1)):
        episode_count += 1
        for agent in agents:
            agent.reset()
        last_obs = env.reset()

        done = torch.tensor([False], device=device)
        while not done.any():
            actions_list = [agent.act(last_obs[i]) for i, agent in enumerate(agents)]
            actions = torch.stack(actions_list, dim=0).unsqueeze(0)  # (1, num_agents)
            obs, rewards, done, info = env.step(actions)
            step += 1

            for i, agent in enumerate(agents):
                agent.observe((None, rewards[:, i], None, None))
                
            influence_estimator.observe((last_obs[0], actions, rewards, done, None))

            last_obs = obs

        influence_estimator.store()
        influence_estimator.update()
        for agent in agents:
            agent.update()

        influence_estimator.episode_reset()
        step_train += 1

        if episode % period == 0:
            (rewards_given, rewards_received, rewards_env,
             rewards_total, cumulative_energy, reward_per_energy) = test_staghunt(n_eval, env, agents, device)
            
            cumulative_energy_expanded = np.tile(cumulative_energy, (rewards_given.shape[0], 1))
            reward_per_energy_expanded = np.tile(reward_per_energy, (rewards_given.shape[0], 1))

            matrix_combined = np.stack([
                rewards_given, 
                rewards_received, 
                rewards_env,
                rewards_total, 
                cumulative_energy_expanded, 
                reward_per_energy_expanded
            ])
            matrix_mean = np.mean(matrix_combined, axis=1)

            s = '%d,%d,%d' % (episode, step_train, step)
            for idx in range(env.num_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(*matrix_mean[:, idx])
            s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

if __name__ == '__main__':
    main()
