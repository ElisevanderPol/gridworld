import numpy as np
from collections import namedtuple
import gym
from gym import spaces
import matplotlib.pyplot as plt
from rlpyt.envs.gym import GymSpaceWrapper, GymEnvWrapper
import random

EnvStep = namedtuple("EnvStep",
    ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class GridEnv(gym.Env):

    def __init__(self, size=(7, 7), timelimit=100, stochastic=True, p=0.15):
        """
        """
        # self.env = self
        self.stochastic = stochastic
        self.p = p
        self.size = size
        self.c_x = int(np.floor(size[0]/2.))
        self.c_y = int(np.floor(size[1]/2.))
        self.timelimit = timelimit

        self.time = 0
        self._action_space = spaces.Discrete(5)
        self.reward_range = [-0.1, 1.]
        self.metadata = {}
        self._observation_space = spaces.Box(low=0, high=1,
                                             shape=(1, size[0]*3,
                                                    size[1]*3))

        self.action_dict = {0: [0, 0],   # nothing
                            1: [1, 0],   # up
                            2: [0, 1],   # right
                            3: [-1, 0],  # down
                            4: [0, -1]}  # left

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a namedtuple containing other diagnostic information from the previous action
        """
        self.time += 1
        agent_move = self.action_dict[int(action)]
        self.move_agent(agent_move)

        if self.stochastic:
            # Stochastic move for prey
            r = random.random()
            if r < self.p:
                n_a = np.random.randint(1, 5)
                prey_move = self.action_dict[n_a]
            else:
                prey_move = self.action_dict[0]
        else:
            prey_move = self.action_dict[0]
        self.move_prey(prey_move)


        prey_x, prey_y = self.prey_loc

        agent_x, agent_y = self.agent_loc

        distance = self.distance(self.prey_loc, self.agent_loc)

        d_x, d_y = self.distance_coord(*distance)

        if (distance[0] == 0 and distance[1] == 0):
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False
        if self.time == self.timelimit:
            done = True

        render = self.render(d_x, d_y)
        return render, reward, done, {}

    def render(self, d_x, d_y):
        """
        Render image representation of the state
        """
        new_grid = np.zeros((1, self.size[0]*3, self.size[1]*3))


        xp = 3*d_x
        yp = 3*d_y

        new_grid[0, xp:xp+3, yp:yp+3] = self.prey_img()

        xc = 3*self.c_x
        yc = 3*self.c_y
        new_grid[0, xc:xc+3, yc:yc+3] = self.agent_img()

        return new_grid

    def prey_img(self):
        """
        """
        img = np.zeros((3, 3))
        img[1] = 1
        img[:, 1] = 1
        return img

    def agent_img(self):
        """
        """
        img = np.zeros((3, 3))
        img[1][1] = 1
        img[0][0] = 1
        img[2][0] = 1
        img[0][2] = 1
        img[2][2] = 1
        return img

    def block_img(self):
        """
        """
        img = np.ones((3, 3))
        return img

    def distance_coord(self, dx, dy):
        """
        """
        d_x = int(dx + self.c_x)
        d_y = int(dy + self.c_y)
        return d_x, d_y

    def distance(self, a, b):
        """
        Toroidal distance [probably not the best way to do this]
        """
        wrap_around_x = wrap_around_y = False
        abs_x = np.abs(a[0]-b[0])

        x_dist = min(abs_x, self.size[0]-abs_x)
        if a[0] > b[0]:
            x_dist *= -1

        if abs_x > (self.size[0] - abs_x):
            wrap_around_x = True
        if wrap_around_x:
            x_dist *= -1

        abs_y = np.abs(a[1]-b[1])

        y_dist = min(abs_y, self.size[1]-abs_y)
        if a[1] > b[1]:
            y_dist *= -1

        if abs_y > (self.size[0] - abs_y):
            wrap_around_y = True
        if wrap_around_y:
            y_dist *= -1

        return x_dist, y_dist

    def move_agent(self, move):
        """
        Update agent location
        """
        x, y = self.agent_loc
        n_x = move[0] + x
        n_y = move[1] + y
        n_x, n_y = self.wrap_edges(n_x, n_y)
        self.agent_loc = n_x, n_y

    def move_prey(self, move):
        """
        Update prey location
        """
        x, y = self.prey_loc
        n_x = move[0] + x
        n_y = move[1] + y
        n_x, n_y = self.wrap_edges(n_x, n_y)
        self.prey_loc = n_x, n_y

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.time = 0
        # Random initial prey location
        prey_x = np.random.randint(0, self.size[0])
        prey_y = np.random.randint(0, self.size[1])
        self.prey_loc = [prey_x, prey_y]

        # Random initial agent location
        agent_x = np.random.randint(0, self.size[0])
        agent_y = np.random.randint(0, self.size[1])
        self.agent_loc = [agent_x, agent_y]

        distance = self.distance(self.prey_loc, self.agent_loc)

        d_x, d_y = self.distance_coord(*distance)

        render = self.render(d_x, d_y)

        return render

    def center(self, x, y):
        """
        Center grid at agent
        """
        centered_x = x - self.c_x
        centered_y = y - self.c_y
        return centered_x, centered_y

    def wrap_edges(self, x, y):
        """
        Toroidal grid
        """
        if x == -1:
            n_x = self.size[0]-1
        elif x == self.size[0]:
            n_x = 0
        else:
            n_x = x
        if y == -1:
            n_y = self.size[1]-1
        elif y == self.size[1]:
            n_y = 0
        else:
            n_y = y
        return n_x, n_y



    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        """Horizon of the environment, if it has one."""
        raise NotImplementedError

    def close(self):
        """Clean up operation."""
        pass


def make(*args, info_example=None, **kwargs):
    env = GridEnv((7, 7))
    return GymEnvWrapper(env)
