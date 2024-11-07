import threading

import gymnasium as gym
from gymnasium import register
from envs.drone_env import AirSimDroneEnv

import random


register(
    id='AirSimDrone-v0',
    entry_point='__main__:AirSimDroneEnv',
    kwargs={
        'ip': '127.0.0.1',
        'image_shape': (144, 256, 1),
        # "render_modes": "rgb_array",
    }
)


if __name__ == '__main__':

    env = gym.make('AirSimDrone-v0')

    for episode in range(10):
        env.reset()
        done = False

        while not done:
            a1 = random.randint(0, 2)
            a2 = random.randint(0, 2)
            action = [a1, a2]

            state, reward, terminated, truncated, info = env.step(action)
            print(state, action, reward, terminated, truncated, info)
            done = terminated or truncated

        print('Episode:', episode, 'Done', done)

