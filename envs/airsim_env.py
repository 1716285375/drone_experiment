import numpy as np

import gymnasium as gym
from gymnasium import spaces


class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.s1 = gym.spaces.Box(-40.0, 40.0, shape=(1, 3), dtype=np.float32)  # 长机与僚机相对速度
        self.s2 = gym.spaces.Box(-180.0, 180.0, shape=(1, 3), dtype=np.float32)  # 长机与僚机相对航向角
        self.s3 = gym.spaces.Box(0.0, 1.0, shape=(1,))  # 队形期望误差
        self.s3 = gym.spaces.Box(0.0, 1.0, shape=(1,))  # 队形期望误差
        self.s3 = gym.spaces.Box(0.0, 1.0, shape=(1,))  # 队形期望误差
        # self.s4 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.s5 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.s6 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.s7 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.s8 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.s9 = gym.spaces.Box(0.0, 1.0, shape=(1,))
        # self.space = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7, self.s8, self.s9]
        self.space = [self.s1, self.s2, self.s3]
        self.observation_space = spaces.Tuple(self.space)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()


