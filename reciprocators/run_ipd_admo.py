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
    parser.add_argument('--num', type=int, default=1)
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

    from evaluate import test_ipd
    n_eval = config.alg.n_eval
    period = config.alg.period

    exp_name = f"ipd_admo_{args.num}"
    dir_name = 'ipd_reciprocators'
    log_path = os.path.join(os.path.dirname(__file__), '..', 'lio', 'results', exp_name, dir_name)
    os.makedirs(log_path, exist_ok=True)

    list_agent_meas = []
    list_suffix = ['given', 'received', 'reward_env', 'reward_total', 'total_energy', 'reward_per_energy']
    for agent_id in range(1, env.num_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    header += ',alpha_inc,alpha_pol,returns_gap,team_welfare,inc_cost\n'

    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    print(f"Starting ADMO ({args.mode}) training on IPD...")
    episode_count = 0
    step_train = 0
    step = 0
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
            step += 1
            episode_rewards.append(rewards)

            for i, agent in enumerate(agents):
                agent.observe((None, rewards[:, i], None, None))

            state_tensor = last_obs[0].unsqueeze(0)
            action_tensor = actions
            reward_tensor = rewards

            influence_estimator.joint_memory.states.append(state_tensor)
            influence_estimator.joint_memory.actions.append(action_tensor)
            influence_estimator.joint_memory.rewards.append(reward_tensor)

            last_obs = obs

        # Apply ADMO step
        all_rewards = torch.cat(episode_rewards, dim=0) # shape: (T, num_agents)
        admo_controller.step(all_rewards)

        influence_estimator.store()
        influence_estimator.update()
        for agent in agents:
            agent.update()

        influence_estimator.episode_reset()
        step_train += 1

        if episode % period == 0:
            (rewards_given, rewards_received, rewards_env,
             rewards_total, cumulative_energy, reward_per_energy) = test_ipd(n_eval, env, agents, device)

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

            alpha_safe = np.nan_to_num(admo_controller.alpha, nan=0.5)
            if alpha_safe.sum() == 0:
                alpha_safe = np.array([0.5, 0.5], dtype=np.float32)
            else:
                alpha_safe = alpha_safe / alpha_safe.sum()

            admo_logs = [
                f"{alpha_safe[0]:.4f}",
                f"{alpha_safe[1]:.4f}",
                f"{np.nan_to_num(admo_controller.signals.get('returns_gap', 0.0), nan=0.0):.4f}",
                f"{np.nan_to_num(admo_controller.signals.get('team_welfare', 0.0), nan=0.0):.4f}",
                f"{np.nan_to_num(admo_controller.signals.get('inc_cost', 0.0), nan=0.0):.4f}"
            ]
            s += ',' + ','.join(admo_logs) + '\n'

            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

if __name__ == '__main__':
    main()
