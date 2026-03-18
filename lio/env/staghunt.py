import numpy as np

class Env(object):
    def __init__(self, config_env):
        self.config = config_env
        self.n_agents = 2
        self.name = 'staghunt'
        self.l_action = 5
        self.l_obs = 10
        self.max_steps = 50
        self.grid_size = 5
        self.rng = np.random.RandomState()
        self.reset()

    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        self.state = self.rng.randint(0, self.grid_size, size=(10,))
        return self._get_obs()

    def _get_obs(self):
        obs = self.state.astype(np.float32) / self.grid_size
        return [obs.copy(), obs.copy()]

    def step(self, actions):
        self.steps += 1
        
        for i in range(2):
            if actions[i] == 0: self.state[2*i+1] -= 1 # Up
            elif actions[i] == 1: self.state[2*i+1] += 1 # Down
            elif actions[i] == 2: self.state[2*i] -= 1 # Left
            elif actions[i] == 3: self.state[2*i] += 1 # Right
            
            self.state[2*i] = np.clip(self.state[2*i], 0, self.grid_size-1)
            self.state[2*i+1] = np.clip(self.state[2*i+1], 0, self.grid_size-1)

        # Random Stag move
        if self.rng.rand() < 0.5:
            stag_a = self.rng.randint(0, 5)
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
            self.state[4:6] = self.rng.randint(0, self.grid_size, size=(2,))
        else:
            if np.array_equal(a1_pos, h1_pos):
                rewards[0] += 1.0
                self.state[6:8] = self.rng.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a1_pos, h2_pos):
                rewards[0] += 1.0
                self.state[8:10] = self.rng.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a2_pos, h1_pos):
                rewards[1] += 1.0
                self.state[6:8] = self.rng.randint(0, self.grid_size, size=(2,))
            if np.array_equal(a2_pos, h2_pos):
                rewards[1] += 1.0
                self.state[8:10] = self.rng.randint(0, self.grid_size, size=(2,))

        done = 1 if self.steps >= self.max_steps else 0
        list_obs_next = self._get_obs()
        return list_obs_next, rewards.tolist(), done
