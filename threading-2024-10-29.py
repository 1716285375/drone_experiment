import threading
import airsim

import gymnasium as gym
from gymnasium import register
from envs.drone_env import AirSimDroneEnv

import random


register(
    id='AirSimDrone-v0',
    entry_point='__main_:AirSimDroneEnv',
    kwargs={
        'ip': '127.0.0.1',
        'image_shape': (144, 256, 1)
    }
)


if __name__ == '__main__':
    env = gym.make('AirSimDrone-v0')

    for episode in range(10):
        env.reset()
        done = False

        while not done:
            a1 = random

