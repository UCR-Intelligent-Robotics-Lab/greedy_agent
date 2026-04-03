import numpy as np

class IPD(object):
    def __init__(self, config):
        self.n_agents = 2
        self.l_action = 2
        self.l_obs = 5
        self.max_steps = config.max_steps
        self.name = 'ipd'
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.l_obs)
        init_state[-1] = 1
        observations = [init_state, init_state]
        return observations

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]

        state = np.zeros(self.l_obs)
        state[ac0 * 2 + ac1] = 1
        observations = [state, state]

        done = (self.step_count == self.max_steps)

        return observations, rewards, done
