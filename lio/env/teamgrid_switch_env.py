import gym
import numpy as np


class Env(object):

    def __init__(self, config_env):
        self.config = config_env
        self.name = 'teamgrid'
        self.env_id = 'TEAMGrid-Switch-v0'
        self._env = self._make_env()

        self.n_agents = getattr(self.config, 'n_agents', 2)
        self.include_agent_id = getattr(self.config, 'include_agent_id', False)

        self.l_action = self._get_action_dim()
        obs_shape = self._get_obs_shape()
        self.obs_dim = int(np.prod(obs_shape))
        if self.include_agent_id:
            self.l_obs = self.obs_dim + self.n_agents
        else:
            self.l_obs = self.obs_dim

    def _get_action_dim(self):
        action_space = self._env.action_space
        if isinstance(action_space, (list, tuple)):
            return int(action_space[0].n)
        return int(action_space.n)

    def _get_obs_shape(self):
        obs_space = self._env.observation_space
        if isinstance(obs_space, (list, tuple)):
            return obs_space[0].shape
        return obs_space.shape

    def _make_env(self):
        try:
            return gym.make(self.env_id)
        except gym.error.Error:
            # Some TEAMGrid installs require an explicit import to register envs.
            for module_name in ("teamgrid", "gym_teamgrid", "teamgrid_gym"):
                try:
                    __import__(module_name)
                    return gym.make(self.env_id)
                except Exception:
                    continue
            raise

    def seed(self, seed=None):
        if seed is not None:
            self._env.reset(seed=seed)
        return [seed]

    def _flatten_obs(self, obs, agent_id):
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.include_agent_id:
            agent_id_1hot = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_1hot[agent_id] = 1.0
            obs_vec = np.concatenate([obs_vec, agent_id_1hot], axis=0)
        return obs_vec

    def _process_obs(self, obs_n):
        if isinstance(obs_n, np.ndarray) and obs_n.ndim >= 4:
            obs_n = [obs_n[i] for i in range(self.n_agents)]
        if not isinstance(obs_n, (list, tuple)):
            obs_n = [obs_n for _ in range(self.n_agents)]
        return [self._flatten_obs(obs, i) for i, obs in enumerate(obs_n)]

    def reset(self):
        reset_out = self._env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_n, _ = reset_out
        else:
            obs_n = reset_out
        return self._process_obs(obs_n)

    def step(self, action_n):
        step_out = self._env.step(action_n)
        if len(step_out) == 5:
            obs_n, reward_n, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs_n, reward_n, done, info = step_out
            done = bool(done)

        if isinstance(reward_n, np.ndarray):
            reward_n = reward_n.tolist()
        elif not isinstance(reward_n, (list, tuple)):
            reward_n = [reward_n for _ in range(self.n_agents)]

        obs_n = self._process_obs(obs_n)
        return obs_n, reward_n, done, info

    def close(self):
        self._env.close()
