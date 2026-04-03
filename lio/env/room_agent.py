import numpy as np


class Actor(object):

    def __init__(self, agent_id, n_agents, l_obs):

        self.agent_id = agent_id
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.position = 1
        self.total_given = np.zeros(self.n_agents - 1)

    def act(self, action, reward_given, observe_given=True):

        self.position = action
        if observe_given:
            reward_given = np.asarray(reward_given, dtype=float).flatten()
            if reward_given.shape[0] == self.n_agents:
                # Exclude self reward share to maintain (n_agents-1) vector
                reward_given = np.delete(reward_given, self.agent_id)
            elif reward_given.shape[0] != self.n_agents - 1:
                raise ValueError(f"Unexpected reward_given length {reward_given.shape[0]} for agent {self.agent_id}")
            self.total_given += reward_given

    def get_obs(self, state, observe_given=True):
        obs = np.zeros(self.l_obs)
        # position of self
        obs[state[self.agent_id]] = 1
        list_others = list(range(0, self.n_agents))
        del list_others[self.agent_id]
        # positions of other agents
        for idx, other_id in enumerate(list_others):
            obs[3*(idx + 1) + state[other_id]] = 1

        # total amount given to other agents
        if observe_given:
            obs[-(self.n_agents - 1):] = self.total_given

        return obs

    def reset(self, randomize=False):
        if randomize:
            self.position = np.random.randint(3)
        else:
            self.position = 1
        self.total_given = np.zeros(self.n_agents - 1)
