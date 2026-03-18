import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class AdaptiveMOConfig:
    mode: int = +1
    c1: float = 0.1
    c2: float = 0.1
    beta_cost: float = 0.0
    max_grad_norm: float = 1.0
    base_k_epochs: int = 10
    base_reciprocal_weight: float = 5.0
    ema_decay: float = 0.9

class MovingAverage:
    def __init__(self, decay: float):
        self.decay = decay
        self.value = 0.0
        self.initialised = False
    def update(self, sample: float):
        if not self.initialised:
            self.value = sample
            self.initialised = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * sample
        return self.value

class AdaptiveMOController:
    def __init__(self, agent, cfg: AdaptiveMOConfig, gamma: float, n_agents: int, agent_id: int):
        self.agent = agent
        self.cfg = cfg
        self.gamma = gamma
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.alpha = np.array([0.5, 0.5], dtype=np.float32)
        self.signals = {'returns_gap': 0.0, 'team_welfare': 0.0, 'inc_cost': 0.0}

    def _discount_vector(self, length: int):
        return np.power(self.gamma, np.arange(length, dtype=np.float32))

    def step(self, episode_env_rewards):
        if torch.is_tensor(episode_env_rewards):
            episode_env_rewards = episode_env_rewards.detach().cpu().numpy()
        
        T = episode_env_rewards.shape[0]
        discounts = self._discount_vector(T)
        
        adv_returns = episode_env_rewards[:, self.agent_id]
        others_returns = np.delete(episode_env_rewards, self.agent_id, axis=1)
        mean_others = np.mean(others_returns, axis=1) if others_returns.shape[1] > 0 else np.zeros_like(adv_returns)
        
        returns_gap = np.sum(discounts * (adv_returns - mean_others))
        team_welfare = np.sum(discounts * np.sum(episode_env_rewards, axis=1))
        
        inc_cost = 0.0
        self.signals.update({'returns_gap': returns_gap, 'team_welfare': team_welfare, 'inc_cost': inc_cost})
        
        g_inc = np.array([returns_gap - self.cfg.beta_cost * inc_cost], dtype=np.float32)
        g_pol = np.array([-self.cfg.mode * team_welfare], dtype=np.float32)
        
        g_inc = np.clip(g_inc, -self.cfg.max_grad_norm, self.cfg.max_grad_norm)
        g_pol = np.clip(g_pol, -self.cfg.max_grad_norm, self.cfg.max_grad_norm)
        
        G = np.stack([g_inc, g_pol], axis=1)
        gram = G.T @ G
        e = np.ones(2, dtype=np.float32)
        c = np.array([self.cfg.c1, self.cfg.c2], dtype=np.float32)
        
        M = np.block([[gram, e[:, None]], [e[None, :], np.zeros((1, 1))]])
        rhs = np.concatenate([-gram @ c, [1 - np.dot(e, c)]])
        
        try:
            solution = np.linalg.solve(M + 1e-6 * np.eye(3), rhs)
            alpha_raw = solution[:2]
        except np.linalg.LinAlgError:
            alpha_raw = np.array([0.5, 0.5], dtype=np.float32)
            
        alpha_clipped = np.maximum(alpha_raw + c, c)
        denom = np.sum(alpha_clipped)
        self.alpha = (alpha_clipped / denom).astype(np.float32) if denom > 0 else np.array([0.5, 0.5], dtype=np.float32)
        
        self.agent.ppo.K_epochs = max(1, int(round(self.cfg.base_k_epochs * float(self.alpha[0]) * 2.0)))
        self.agent.reciprocal_reward_weight = self.cfg.base_reciprocal_weight * float(self.alpha[1]) * 2.0
