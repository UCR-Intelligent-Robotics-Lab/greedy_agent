import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lio.env import room_symmetric
from lio.alg import config_room_lio
from src.agents.reciprocator import Reciprocator
from src.agents.naive_learner import NaiveLearner
from src.agents.voi import ValueOfInfluence

class EnvWrapper:
    def __init__(self, lio_env, device):
        self.env = lio_env
        self.device = device
        self.num_agents = lio_env.n_agents
        self.l_obs = lio_env.l_obs
        self.l_action = lio_env.l_action

    def reset(self):
        obs = self.env.reset()
        return torch.tensor(np.array(obs), device=self.device, dtype=torch.float).unsqueeze(1)

    def step(self, actions: torch.Tensor):
        list_actions = actions[0].tolist()
        # each agent provides reward values to (n_agents - 1) others
        list_rewards = [np.zeros(self.num_agents - 1) for _ in range(self.num_agents)]
        if self.env.name == 'er':
            obs, rewards, done = self.env.step(list_actions, list_rewards)
        else:
            obs, rewards, done = self.env.step(list_actions)
        
        obs_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float).unsqueeze(1)
        rewards_tensor = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).unsqueeze(0)
        done_tensor = torch.tensor([bool(done)], device=self.device, dtype=torch.bool)
        return obs_tensor, rewards_tensor, done_tensor, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    config = config_room_lio.get_config()
    
    lio_env = room_symmetric.Env(config.env)
    env = EnvWrapper(lio_env, device)

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
            'reciprocal_reward_type': 'linear',
            'device': device,
            'normalize_reciprocal_reward': False,
            'rnn': False
        }
        agents.append(Reciprocator(**reciprocator_kwargs))

    print("Starting training on ER...")
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
                
            state_tensor = last_obs[0].unsqueeze(0)
            action_tensor = actions
            reward_tensor = rewards

            # ValueOfInfluence uses joint_memory; align with its API.
            influence_estimator.joint_memory.states.append(state_tensor)
            influence_estimator.joint_memory.actions.append(action_tensor)
            influence_estimator.joint_memory.rewards.append(reward_tensor)

            # update_influence_balance is not present in this version; use direct influence estimator memory
            last_obs = obs

        influence_estimator.store()
        influence_estimator.update()
        for agent in agents:
            agent.update()

        influence_estimator.episode_reset()

if __name__ == '__main__':
    main()
