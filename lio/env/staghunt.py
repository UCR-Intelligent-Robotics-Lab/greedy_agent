import numpy as np

class Env(object):
    def __init__(self, config_env):
        self.config = config_env
        self.n_agents = getattr(self.config, 'n_agents', 2)
        self.name = 'staghunt'
        self.l_action = 5
        self.max_steps = 50
        self.grid_size = 5
        self.n_hares = self.n_agents
        # obs is [agents pos (2*n_agents) + stag pos (2) + hares pos (2*n_hares)]
        self.l_obs = (2 * self.n_agents) + 2 + (2 * self.n_hares)
        self.rng = np.random.RandomState()
        self.reset()

    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        self.state = self.rng.randint(0, self.grid_size, size=(self.l_obs,))
        return self._get_obs()

    def _get_obs(self):
        obs = self.state.astype(np.float32) / self.grid_size
        return [obs.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
        for i in range(self.n_agents):
            if actions[i] == 0: self.state[2*i+1] -= 1 # Up
            elif actions[i] == 1: self.state[2*i+1] += 1 # Down
            elif actions[i] == 2: self.state[2*i] -= 1 # Left
            elif actions[i] == 3: self.state[2*i] += 1 # Right
            
            self.state[2*i] = np.clip(self.state[2*i], 0, self.grid_size-1)
            self.state[2*i+1] = np.clip(self.state[2*i+1], 0, self.grid_size-1)

        # Random Stag move
        stag_idx = 2 * self.n_agents
        if self.rng.rand() < 0.5:
            stag_a = self.rng.randint(0, 5)
            if stag_a == 0: self.state[stag_idx+1] -= 1
            elif stag_a == 1: self.state[stag_idx+1] += 1
            elif stag_a == 2: self.state[stag_idx] -= 1
            elif stag_a == 3: self.state[stag_idx] += 1
            self.state[stag_idx] = np.clip(self.state[stag_idx], 0, self.grid_size-1)
            self.state[stag_idx+1] = np.clip(self.state[stag_idx+1], 0, self.grid_size-1)
            
        rewards = np.zeros(self.n_agents)
        
        agents_pos = [self.state[2*i:2*i+2] for i in range(self.n_agents)]
        stag_pos = self.state[stag_idx:stag_idx+2]
        hares_pos = [self.state[stag_idx+2+(2*i):stag_idx+4+(2*i)] for i in range(self.n_hares)]

        # Check if ALL agents are on stag
        all_on_stag = all(np.array_equal(pos, stag_pos) for pos in agents_pos)
        
        if all_on_stag:
            for i in range(self.n_agents):
                rewards[i] += 5.0
            self.state[stag_idx:stag_idx+2] = self.rng.randint(0, self.grid_size, size=(2,))
        else:
            for i in range(self.n_agents):
                for j in range(self.n_hares):
                    if np.array_equal(agents_pos[i], hares_pos[j]):
                        rewards[i] += 1.0
                        self.state[stag_idx+2+(2*j):stag_idx+4+(2*j)] = self.rng.randint(0, self.grid_size, size=(2,))

        done = 1 if self.steps >= self.max_steps else 0
        list_obs_next = self._get_obs()
        return list_obs_next, rewards.tolist(), done
