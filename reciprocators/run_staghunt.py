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

    print("Starting training on Spatial Stag Hunt...")
    episode_count = 0
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

            for i, agent in enumerate(agents):
                agent.observe((None, rewards[:, i], None, None))
                
            influence_estimator.observe((last_obs[0], actions, rewards, done, None))

            last_obs = obs

        influence_estimator.store()
        influence_estimator.update()
        for agent in agents:
            agent.update()

        influence_estimator.episode_reset()

if __name__ == '__main__':
    main()
