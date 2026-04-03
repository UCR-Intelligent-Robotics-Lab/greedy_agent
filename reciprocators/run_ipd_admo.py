import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lio.env import ipd_wrapper
from lio.alg import config_ipd_lio
from src.agents.reciprocator import Reciprocator
from src.agents.naive_learner import NaiveLearner
from src.agents.voi import ValueOfInfluence
from admo import AdaptiveMOConfig, AdaptiveMOController

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
        obs, rewards, done = self.env.step(list_actions)
        
        obs_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float).unsqueeze(1)
        rewards_tensor = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).unsqueeze(0)
        done_tensor = torch.tensor([bool(done)], device=self.device, dtype=torch.bool)
        return obs_tensor, rewards_tensor, done_tensor, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', choices=['adversarial', 'constructive'], default='adversarial')
    parser.add_argument('--adversary', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = config_ipd_lio.get_config()
    
    lio_env = ipd_wrapper.IPD(config.env)
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
            'reciprocal_reward_type': 'petty',
            'device': device,
            'normalize_reciprocal_reward': False,
            'rnn': False
        }
        agents.append(Reciprocator(**reciprocator_kwargs))

    mode_sign = +1 if args.mode == 'adversarial' else -1
    admo_cfg = AdaptiveMOConfig(mode=mode_sign, base_k_epochs=10, base_reciprocal_weight=5.0)
    admo_controller = AdaptiveMOController(agents[args.adversary], admo_cfg, gamma=0.99, n_agents=env.num_agents, agent_id=args.adversary)

    print(f"Starting ADMO ({args.mode}) training on IPD...")
    episode_count = 0
    last_obs = env.reset()

    for episode in tqdm(range(1, args.episodes + 1)):
        episode_count += 1
        for agent in agents:
            agent.reset()
        last_obs = env.reset()
        episode_rewards = []

        done = torch.tensor([False], device=device)
        while not done.any():
            actions_list = [agent.act(last_obs[i]) for i, agent in enumerate(agents)]
            actions = torch.stack(actions_list, dim=0).unsqueeze(0)  # (1, num_agents)
            obs, rewards, done, info = env.step(actions)
            episode_rewards.append(rewards)

            for i, agent in enumerate(agents):
                agent.observe((None, rewards[:, i], None, None))
                
            state_tensor = last_obs[0].unsqueeze(0)
            action_tensor = actions
            reward_tensor = rewards

            influence_estimator.joint_memory.states.append(state_tensor)
            influence_estimator.joint_memory.actions.append(action_tensor)
            influence_estimator.joint_memory.rewards.append(reward_tensor)

            # update_influence_balance not available; we rely on influence estimator computed at update()
            last_obs = obs

        # Apply ADMO step
        all_rewards = torch.cat(episode_rewards, dim=0) # shape: (T, num_agents)
        admo_controller.step(all_rewards)

        influence_estimator.store()
        influence_estimator.update()
        for agent in agents:
            agent.update()

        influence_estimator.episode_reset()

if __name__ == '__main__':
    main()
