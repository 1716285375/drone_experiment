import sys
sys.path.remove()
sys.path.append(r'/D:/AI_Project/RL/airsim-all/experiment')

from stable_baselines3.common.env_checker import check_env
from envs.drone_env import *
from gymnasium.spaces import Space
from gymnasium import register
import gymnasium as gym

if __name__ == '__main__':


    register(
        id='AirSimDrone-v0',                          # 环境名字
        entry_point='__main__:AirSimDroneEnv',      # 创建环境的入口点
        # reward_threshold=np.inf,                    # 一个所能获得奖励的最大值
        # nondeterministic=True,                      # 环境是否是随机的
        # max_episode_steps=10,                       # 一个回合所能执行的最大步数
        # order_enforce=True,                         # 是否启用顺序执行器包装器以确保用户以正确的顺序运行函数
        # disable_env_checker=True,                   # 是否禁用 gymnasium.wrappers.PassiveEnvChecker
        # additional_wrappers=True,                   # 额外的包装器
        # vector_entry_point=True,                    # 用于创建矢量环境的入口点
        kwargs={
            'ip': '127.0.0.1',
            'image_shape': (144, 256, 1),
        }
    )
    env = gym.make("AirSimDroneEnv-v0")
    # It will check your custom environment and output additional warnings if needed
    check_env(env)