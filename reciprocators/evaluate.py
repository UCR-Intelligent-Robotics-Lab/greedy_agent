import numpy as np
import torch

def test_er(n_eval, env, list_agents, device):
    env_agents = env.num_agents
    rewards_total = np.zeros(env_agents)
    rewards_env = np.zeros(env_agents)
    n_move_lever = np.zeros(env_agents)
    n_move_door = np.zeros(env_agents)
    rewards_received = np.zeros(env_agents)
    r_lever = np.zeros(env_agents)
    r_start = np.zeros(env_agents)
    r_door = np.zeros(env_agents)
    rewards_given = np.zeros(env_agents)
    win_rate = np.zeros(env_agents)
    cumulative_energy = np.zeros(env_agents)
    cumulative_env_rewards = np.zeros(env_agents)
    reward_per_energy = np.zeros(env_agents)
    win = 0
    lose = 0
    total_steps = 0

    for _ in range(1, n_eval + 1):
        list_obs = env.reset()
        done_any = False
        steps = 0

        while not done_any:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                # Greedy action for evaluation
                with torch.no_grad():
                    state_tensor = list_obs[idx].unsqueeze(0)
                    action_probs = agent.ppo.policy_old.actor(state_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                list_actions.append(action)

                if action % 3 == 0:
                    n_move_lever[idx] += 1
                elif action % 3 == 2:
                    n_move_door[idx] += 1

            actions = torch.tensor([list_actions], device=device)
            obs_next, env_rewards, done_tensor, _ = env.step(actions)
            env_rewards_np = env_rewards.cpu().numpy()[0]
            done_any = done_tensor.any().item()
            
            for idx in range(env_agents):
                cumulative_env_rewards[idx] += env_rewards_np[idx]

            list_obs = obs_next
            steps += 1
            total_steps += 1
            if done_any:
                if env_rewards_np[0] > 0:
                    win += 1
                else:
                    lose += 1

    return (rewards_total, cumulative_env_rewards, n_move_lever, n_move_door, 
            rewards_received, rewards_given, total_steps/n_eval, r_lever, r_start, r_door,
            np.array([win / n_eval]*env_agents), cumulative_energy, reward_per_energy)

def test_ipd(n_eval, env, list_agents, device):
    env_agents = env.num_agents
    rewards_given = np.zeros((n_eval, env_agents))
    rewards_received = np.zeros((n_eval, env_agents))
    rewards_env = np.zeros((n_eval, env_agents))
    rewards_total = np.zeros((n_eval, env_agents))
    cumulative_energy = np.zeros(env_agents)
    reward_per_energy = np.zeros(env_agents)

    for idx_ep in range(n_eval):
        list_obs = env.reset()
        done_any = False
        while not done_any:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                with torch.no_grad():
                    action_probs = agent.ppo.policy_old.actor(list_obs[idx].unsqueeze(0))
                    action = torch.argmax(action_probs, dim=-1).item()
                list_actions.append(action)

            actions = torch.tensor([list_actions], device=device)
            obs_next, env_rewards, done_tensor, _ = env.step(actions)
            env_rewards_np = env_rewards.cpu().numpy()[0]
            done_any = done_tensor.any().item()
            
            rewards_env[idx_ep] += env_rewards_np
            rewards_total[idx_ep] += env_rewards_np

            list_obs = obs_next

    return rewards_given, rewards_received, rewards_env, rewards_total, cumulative_energy, reward_per_energy

def test_staghunt(n_eval, env, list_agents, device):
    env_agents = env.num_agents
    rewards_given = np.zeros((n_eval, env_agents))
    rewards_received = np.zeros((n_eval, env_agents))
    rewards_env = np.zeros((n_eval, env_agents))
    rewards_total = np.zeros((n_eval, env_agents))
    cumulative_energy = np.zeros(env_agents)
    reward_per_energy = np.zeros(env_agents)

    for idx_ep in range(n_eval):
        list_obs = env.reset()
        done_any = False
        while not done_any:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                with torch.no_grad():
                    action_probs = agent.ppo.policy_old.actor(list_obs[idx].unsqueeze(0))
                    action = torch.argmax(action_probs, dim=-1).item()
                list_actions.append(action)

            actions = torch.tensor([list_actions], device=device)
            obs_next, env_rewards, done_tensor, _ = env.step(actions)
            env_rewards_np = env_rewards.cpu().numpy()[0]
            done_any = done_tensor.any().item()
            
            rewards_env[idx_ep] += env_rewards_np
            rewards_total[idx_ep] += env_rewards_np

            list_obs = obs_next

    return rewards_given, rewards_received, rewards_env, rewards_total, cumulative_energy, reward_per_energy
