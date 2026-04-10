import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lio.env import staghunt
from lio.alg import config_staghunt_lio
from src.agents.reciprocator import Reciprocator
from src.agents.naive_learner import NaiveLearner
from src.agents.voi import ValueOfInfluence
from evaluate import test_staghunt
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
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--mode', choices=['adversarial', 'constructive'], default='adversarial')
    parser.add_argument('--adversary', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    config = config_staghunt_lio.get_config()
    config.env.n_agents = args.n_agents
    
    lio_env = staghunt.Env(config.env)
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

    
    mode_sign = +1 if args.mode == 'adversarial' else -1
    admo_cfg = AdaptiveMOConfig(mode=mode_sign, enable_auto_mode=False)
    
    # Wrap adversary with ADMO controller
    adversary_agent = agents[args.adversary]
    admo_controller = AdaptiveMOController(adversary_agent, admo_cfg, gamma=0.99, n_agents=env.num_agents, agent_id=args.adversary)

    n_eval = config.alg.n_eval
    period = config.alg.period

    exp_name = f"staghunt_admo_{args.n_agents}_trail_{args.num}"
    dir_name = f"staghunt_reciprocators_{args.n_agents}"
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

    print(f"Starting ADMO training on Stag Hunt ({args.n_agents})...")
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
            actions = torch.stack(actions_list, dim=0).unsqueeze(0)
            obs, rewards, done, info = env.step(actions)
            step += 1

            if args.mode == 'adversarial':
                rewards[:, args.adversary] = -rewards[:, args.adversary]

            for i, agent in enumerate(agents):
                agent.observe((None, rewards[:, i], None, None))
                
            state_tensor = last_obs[0].unsqueeze(0)
            action_tensor = actions
            reward_tensor = rewards

            influence_estimator.joint_memory.states.append(state_tensor)
            influence_estimator.joint_memory.actions.append(action_tensor)
            influence_estimator.joint_memory.rewards.append(reward_tensor)

            last_obs = obs


        influence_estimator.store()
        influence_estimator.update()
        for i, agent in enumerate(agents):
            if i != args.adversary:
                agent.update()
                
        # ADMO updates the adversary
        # we need to pass mission_status, which is roughly done.any()
        admo_controller.step(episode, done.any().item(), influence_estimator.joint_memory)


        influence_estimator.episode_reset()
        step_train += 1

        if episode % period == 0:
            (rewards_given, rewards_received, rewards_env,
             rewards_total, cumulative_energy, reward_per_energy) = test_staghunt(n_eval, env.env, agents, device)
            
            cumulative_energy_expanded = np.tile(cumulative_energy, (rewards_given.shape[0], 1)) if rewards_given.ndim > 1 else cumulative_energy
            reward_per_energy_expanded = np.tile(reward_per_energy, (rewards_given.shape[0], 1)) if rewards_given.ndim > 1 else reward_per_energy

            matrix_combined = np.stack([
                rewards_given, 
                rewards_received, 
                rewards_env,
                rewards_total, 
                cumulative_energy_expanded, 
                reward_per_energy_expanded
            ])
            matrix_mean = np.mean(matrix_combined, axis=1) if matrix_combined.ndim > 2 else matrix_combined
            
            s = '%d,%d,%d' % (episode, step_train, step)
            for idx in range(env.num_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(*matrix_mean[:, idx])
            s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

if __name__ == '__main__':
    main()
