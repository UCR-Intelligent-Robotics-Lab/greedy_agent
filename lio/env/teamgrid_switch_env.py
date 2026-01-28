import gym
import numpy as np


class Env(object):

    def __init__(self, config_env):
        self.config = config_env
        self.name = 'teamgrid'
        self.env_id = 'TEAMGrid-Switch-v0'
        self._env = self._make_env()
        self._enforce_size()
        self._fixed_layout_initialized = False
        self._fixed_switch_pos = None
        self._fixed_goal_pos = None
        self._episode_mode = None
        self._episode_train_ep = None
        self._episode_test_ep = None
        self._episode_step = 0
        self._episode_switch_step = -1
        self._episode_goal_step = -1
        self._episode_switch_toggler = None
        self._episode_goal_reacher = None
        self._episode_timeout_printed = False
        self._divider_x_cache = None

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
        size = getattr(self.config, 'size', None)
        kwargs = {}
        if size is not None:
            kwargs['size'] = size

        def _try_make():
            try:
                return gym.make(self.env_id, **kwargs)
            except TypeError:
                if kwargs:
                    return gym.make(self.env_id)
                raise

        try:
            return _try_make()
        except gym.error.Error:
            # Some TEAMGrid installs require an explicit import to register envs.
            for module_name in ("teamgrid", "gym_teamgrid", "teamgrid_gym"):
                try:
                    __import__(module_name)
                    return _try_make()
                except Exception:
                    continue
            raise

    def _enforce_size(self):
        size = getattr(self.config, 'size', None)
        if size is None:
            return
        env = getattr(self._env, 'unwrapped', self._env)
        width = 2 * int(size) + 3
        height = int(size) + 2
        if hasattr(env, 'width'):
            try:
                env.width = width
            except Exception:
                pass
        if hasattr(env, 'height'):
            try:
                env.height = height
            except Exception:
                pass

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

    def _divider_x(self):
        if self._divider_x_cache is not None:
            return self._divider_x_cache
        env = getattr(self._env, 'unwrapped', self._env)
        width = getattr(env, 'width', None)
        if width is None:
            return None
        self._divider_x_cache = int(width // 2)
        return self._divider_x_cache

    def room_id_from_pos(self, pos):
        if pos is None:
            return None
        divider = self._divider_x()
        if divider is None:
            return None
        return 1 if int(pos[0]) < divider else 2

    def _left_room_bounds(self):
        env = getattr(self._env, 'unwrapped', self._env)
        width = getattr(env, 'width', None)
        height = getattr(env, 'height', None)
        divider = self._divider_x()
        if width is None or height is None or divider is None:
            return None
        x_low = 1
        x_high = divider
        y_low = 1
        y_high = height - 1
        return x_low, x_high, y_low, y_high

    def _sample_empty_left_pos(self, occupied):
        env = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env, 'grid', None)
        bounds = self._left_room_bounds()
        if grid is None or bounds is None:
            return None
        x_low, x_high, y_low, y_high = bounds
        max_tries = 1000
        for _ in range(max_tries):
            x = int(env._rand_int(x_low, x_high))
            y = int(env._rand_int(y_low, y_high))
            if (x, y) in occupied:
                continue
            if grid.get(x, y) is not None:
                continue
            return (x, y)
        return None

    def _ensure_agents_left_room(self):
        env = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env, 'grid', None)
        agents = getattr(env, 'agents', [])
        if grid is None or not agents:
            return
        positions = self.get_agent_positions()
        room_ids = [self.room_id_from_pos(pos) for pos in positions]
        if all(room_id == 1 for room_id in room_ids if room_id is not None):
            return

        for pos in positions:
            if pos is None:
                continue
            grid.set(pos[0], pos[1], None)

        occupied = set()
        if self._fixed_switch_pos:
            occupied.add(tuple(self._fixed_switch_pos))
        if self._fixed_goal_pos:
            occupied.add(tuple(self._fixed_goal_pos))

        for agent in agents:
            pos = self._sample_empty_left_pos(occupied)
            if pos is None:
                continue
            grid.set(pos[0], pos[1], agent)
            agent.cur_pos = np.array(pos)
            occupied.add(pos)
            if hasattr(env, '_rand_int'):
                agent.dir = int(env._rand_int(0, 4))

    def reset(self):
        reset_out = self._env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_n, _ = reset_out
        else:
            obs_n = reset_out
        if not self._fixed_layout_initialized:
            switch_pos, goal_pos = self.get_debug_positions()
            self._fixed_switch_pos = switch_pos
            self._fixed_goal_pos = goal_pos
            self._fixed_layout_initialized = True
            print("[DEBUG][TEAMGRID][LAYOUT] fixed switch_pos=%s, fixed goal_pos=%s"
                  % (self._fixed_switch_pos, self._fixed_goal_pos))
        else:
            self._apply_fixed_layout()
            env_unwrapped = getattr(self._env, 'unwrapped', self._env)
            if hasattr(env_unwrapped, 'gen_obss'):
                obs_n = env_unwrapped.gen_obss()
        self._ensure_agents_left_room()
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        if hasattr(env_unwrapped, 'gen_obss'):
            obs_n = env_unwrapped.gen_obss()
        self._reset_episode_tracking()
        self._print_episode_start()
        return self._process_obs(obs_n)

    def set_episode_context(self, mode, train_ep=None, test_ep=None):
        self._episode_mode = mode
        self._episode_train_ep = train_ep
        self._episode_test_ep = test_ep

    def get_agent_positions(self):
        env = getattr(self._env, 'unwrapped', self._env)
        agents = getattr(env, 'agents', [])
        positions = []
        for agent in agents:
            pos = getattr(agent, 'cur_pos', None)
            if pos is None:
                positions.append(None)
            else:
                positions.append((int(pos[0]), int(pos[1])))
        return positions

    def get_episode_event_timesteps(self):
        return self._episode_switch_step, self._episode_goal_step

    def _reset_episode_tracking(self):
        self._episode_step = 0
        self._episode_switch_step = -1
        self._episode_goal_step = -1
        self._episode_switch_toggler = None
        self._episode_goal_reacher = None
        self._episode_timeout_printed = False

    def _format_episode_prefix(self):
        mode = self._episode_mode or 'EP'
        if mode == 'TEST':
            return "[DEBUG][TEAMGRID][TEST ][train_ep=%s][test_ep=%s]" % (
                self._episode_train_ep, self._episode_test_ep)
        if mode == 'TRAIN':
            return "[DEBUG][TEAMGRID][TRAIN][ep=%s]" % self._episode_train_ep
        return "[DEBUG][TEAMGRID][%s]" % mode

    def _print_episode_start(self):
        positions = self.get_agent_positions()
        if len(positions) >= 2:
            pos_a1 = positions[0]
            pos_a2 = positions[1]
        elif len(positions) == 1:
            pos_a1 = positions[0]
            pos_a2 = None
        else:
            pos_a1 = None
            pos_a2 = None
        room_a1 = self.room_id_from_pos(pos_a1)
        room_a2 = self.room_id_from_pos(pos_a2)
        prefix = self._format_episode_prefix()
        print("%s A1_pos=%s, room=%s; A2_pos=%s, room=%s" % (
            prefix, pos_a1, room_a1, pos_a2, room_a2))

    def _find_object_by_type(self, obj_type):
        env = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env, 'grid', None)
        if grid is None:
            return None, None
        for y in range(grid.height):
            for x in range(grid.width):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if getattr(obj, 'type', None) == obj_type:
                    return obj, (x, y)
        return None, None

    def _apply_fixed_layout(self):
        env = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env, 'grid', None)
        if grid is None:
            return
        switch_obj, switch_pos = self._find_object_by_type('switch')
        goal_obj = None
        goal_pos = None
        goals = getattr(env, 'goals', None)
        if goals:
            goal_obj = goals[0]
            cur_pos = getattr(goal_obj, 'cur_pos', None)
            if cur_pos is not None:
                goal_pos = (int(cur_pos[0]), int(cur_pos[1]))
        if goal_pos is None:
            goal_obj, goal_pos = self._find_object_by_type('ball')

        if self._fixed_switch_pos and switch_obj and switch_pos != self._fixed_switch_pos:
            grid.set(switch_pos[0], switch_pos[1], None)
            grid.set(self._fixed_switch_pos[0], self._fixed_switch_pos[1], switch_obj)
            switch_obj.cur_pos = np.array(self._fixed_switch_pos)

        if self._fixed_goal_pos and goal_obj and goal_pos != self._fixed_goal_pos:
            grid.set(goal_pos[0], goal_pos[1], None)
            grid.set(self._fixed_goal_pos[0], self._fixed_goal_pos[1], goal_obj)
            goal_obj.cur_pos = np.array(self._fixed_goal_pos)

    def get_debug_positions(self):
        env = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env, 'grid', None)
        switch_pos = None
        goal_pos = None
        if grid is None:
            return switch_pos, goal_pos
        for y in range(grid.height):
            for x in range(grid.width):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if switch_pos is None and getattr(obj, 'type', None) == 'switch':
                    switch_pos = (x, y)
                if goal_pos is None and getattr(obj, 'type', None) in ('ball', 'goal'):
                    goal_pos = (x, y)
                if switch_pos is not None and goal_pos is not None:
                    return switch_pos, goal_pos
        return switch_pos, goal_pos

    def step(self, action_n):
        env_unwrapped = getattr(self._env, 'unwrapped', self._env)
        grid = getattr(env_unwrapped, 'grid', None)
        actions_enum = getattr(env_unwrapped, 'actions', None)
        pre_toggled = getattr(env_unwrapped, 'toggled', False)
        toggle_action = getattr(actions_enum, 'toggle', None) if actions_enum else None
        forward_action = getattr(actions_enum, 'forward', None) if actions_enum else None
        switch_toggler = None
        goal_reacher = None
        if grid is not None and actions_enum is not None:
            agents = getattr(env_unwrapped, 'agents', [])
            for idx, agent in enumerate(agents):
                try:
                    fwd_pos = agent.front_pos
                except Exception:
                    continue
                fwd_cell = grid.get(int(fwd_pos[0]), int(fwd_pos[1]))
                if (not pre_toggled and toggle_action is not None
                        and action_n[idx] == toggle_action
                        and fwd_cell is not None
                        and getattr(fwd_cell, 'type', None) == 'switch'):
                    switch_toggler = idx
                if (forward_action is not None
                        and action_n[idx] == forward_action
                        and fwd_cell is not None
                        and getattr(fwd_cell, 'type', None) in ('ball', 'goal')):
                    goal_reacher = idx
        step_out = self._env.step(action_n)
        if len(step_out) == 5:
            obs_n, reward_n, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs_n, reward_n, done, info = step_out
            done = bool(done)

        self._episode_step += 1
        if (self._episode_switch_step < 0 and not pre_toggled
                and getattr(env_unwrapped, 'toggled', False)):
            self._episode_switch_step = self._episode_step
            self._episode_switch_toggler = switch_toggler
            prefix = self._format_episode_prefix()
            toggler = 'A%d' % (switch_toggler + 1) if switch_toggler is not None else 'A?'
            print("%s SWITCH toggled by %s at step=%d" % (
                prefix, toggler, self._episode_switch_step))

        reward_list = reward_n
        if isinstance(reward_list, np.ndarray):
            reward_list = reward_list.tolist()
        if not isinstance(reward_list, (list, tuple)):
            reward_list = [reward_list for _ in range(self.n_agents)]
        success = max(reward_list) > 0 if reward_list else False
        if self._episode_goal_step < 0 and success:
            if goal_reacher is None and reward_list:
                goal_reacher = int(np.argmax(reward_list))
            self._episode_goal_step = self._episode_step
            self._episode_goal_reacher = goal_reacher
            prefix = self._format_episode_prefix()
            reacher = 'A%d' % (goal_reacher + 1) if goal_reacher is not None else 'A?'
            print("%s SUCCESS: GOAL reached by %s at step=%d, reward=%s" % (
                prefix, reacher, self._episode_goal_step, reward_list))
        if done and not success and not self._episode_timeout_printed:
            prefix = self._format_episode_prefix()
            print("%s TIMEOUT at step=%d, reward=%s" % (
                prefix, self._episode_step, reward_list))
            self._episode_timeout_printed = True

        if isinstance(reward_n, np.ndarray):
            reward_n = reward_n.tolist()
        elif not isinstance(reward_n, (list, tuple)):
            reward_n = [reward_n for _ in range(self.n_agents)]

        obs_n = self._process_obs(obs_n)
        return obs_n, reward_n, done, info

    def close(self):
        self._env.close()
