# -*- coding:utf-8 -*-
# gridworld(cliff-walking) environment for AI3603

import gym
from gym import spaces
from gym.utils import seeding


class Grid(object):
    def __init__(self, x: int = None, y: int = None, type: int = 0, reward: int = 0.0,
                 value: float = 0.0):
        self.x = x
        self.y = y
        self.type = type
        self.reward = reward
        self.value = value
        self.name = None
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(
            self.x, self.y, self.type, self.reward, self.value, self.name
        )


class GridMatrix(object):
    def __init__(self, n_width: int, n_height: int, default_type: int = 0,
                 default_reward: float = 0.0, default_value: float = 0.0):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(
                    x, y, self.default_type, self.default_reward, self.default_value
                ))

    def get_grid(self, x, y=None):
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert(xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height), \
            "coordinates should be in reasonable range"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type


class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, n_width: int = 10, n_height: int = 7, u_size = 40,
                 default_reward: float = 0, default_type=0, windy=False):
        self.u_size = u_size
        self.n_width = n_width
        self.n_height = n_height
        self.width = u_size * n_width
        self.height = u_size * n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(
            n_width = self.n_width,
            n_height = self.n_height,
            default_reward = self.default_reward,
            default_type = self.default_type,
            default_value = 0.0
        )
        self.reward = 0
        self.action = None
        self.windy = windy

        # 0, 1, 2, 3, 4 represent left, right, up, down, -.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)
        self.ends = [(7, 3)]
        self.start = (0, 3)
        self.types = []
        self.rewards = []
        self.refresh_setting()
        self.viewer = None
        self.seed()
        self.reset()

    def _adjust_size(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if self.windy:
            if new_x in [3, 4, 5, 8]:
                new_y += 1
            elif new_x in [6, 7]:
                new_y += 2

        if action == 0:
            new_x -= 1
        elif action == 1:
            new_x += 1
        elif action == 2:
            new_y += 1
        elif action == 3:
            new_y -= 1

        elif action == 4:
            new_x, new_y = new_x - 1, new_y - 1
        elif action == 5:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 7:
            new_x, new_y = new_x + 1, new_y + 1

        if new_x < 0:
            new_x = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_height:
            new_y = self.n_height - 1

        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)

        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        info = { "x": new_x, "y": new_y }
        return self.state, self.reward, done, info

    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert isinstance(y, int), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1

    def refresh_setting(self):
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert isinstance(x, tuple) , "incomplete coordinate values"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    def close(self):
        self.render(close=True)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < -5:
                        rect.set_color(0.3, 0.3, 0.3)
                    elif r == -0.1:
                        rect.set_color(0.9, 0.9, 0.9)
                    elif r < 0:
                        print(r)
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)

                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        rect.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass

            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def CliffWalk():
    env = GridWorldEnv(n_width = 12, n_height = 4, u_size = 60, default_reward = -1,
                       default_type = 0, windy = False)
    env.action_space = spaces.Discrete(4)
    env.start = (0, 0)
    env.ends = [(11, 0)]
    env.rewards.append((11, 0, 10))
    for i in range(10):
        env.rewards.append((i + 1, 0, -100))
        env.ends.append((i + 1, 0))
    env.refresh_setting()
    return env
