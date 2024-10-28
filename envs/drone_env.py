import os
import time
import numpy as np
import json
import threading

import airsim
import gymnasium as gym
from gymnasium import spaces
from envs.airsim_env import AirSimEnv
from common.airsim_utils import *


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip, image_shape):
        super().__init__(image_shape)
        self.ip_address = ip
        self.client = airsim.MultirotorClient(ip=self.ip_address)  # 加载无人机
        self.cfg = json.load(open(r"C:\Users\jie\Documents\AirSim\settings.json", encoding='utf-8'))
        self.drones = {}    # 无人机个数

        # self._setup_flight()    # 初始化无人机状态

        self.current_step = 0
        self.max_episode_steps = 300

        self.action_space = gym.spaces.MultiDiscrete([3, 3])    # 动作空间

        self.state = {
            "leader": {
                "positions": airsim.Vector3r(),
                "prev_position": airsim.Vector3r(),
                "velocity": airsim.Vector3r(),
                "yaw_angle": tuple(),
                "collision": False,
                "prev_command": {
                    "velocity": 0,
                    "yaw_angle": 0,
                }
            },
            "follower": {
                "positions": airsim.Vector3r(),
                "prev_position": airsim.Vector3r(),
                "velocity": airsim.Vector3r(),
                "yaw_angle": tuple(),
                "collision": False,
                "prev_command": {
                    "velocity": 0,
                    "yaw_angle": 0,
                }
            },
            "pos-z": -20,
            "formation": np.array([-2, 2])
        }

        # 速度、偏航角的取值范围
        self.range = {
            "velocity": np.array([30.0, 70.0]),
            "heading_angle": np.array([-20.0, 20.0]),
            "distance": np.array([])
        }
        self.name = None
        self.thread = threading.Thread(target=self.run, args=(self.name))
        self.initialize()

    def __del__(self):
        pass

    def initialize(self):
        for i, (key, value) in enumerate(self.cfg["Vehicles"].items()):
            if i == 0:
                self.thread.start()
            self.drones[key] = value
        pass

    def run(self, key):
        while True:
            vx = 1
            vy = 1
            self.client.moveByVelocityZAsync(vx, vy, self.state["pos-z"], 1, vehicle_name=self.drones[key])
        pass

    def _setup_flight(self):
        self.client.reset()
        self.current_step = 0
        takeoff_height = self.state["pos-z"]
        for key, value in self.drones.items():
            self.client.enableApiControl(True, vehicle_name=key)
            self.client.armDisarm(True, vehicle_name=key)
            # Set home positions and velocity
            self.client.moveToZAsync(takeoff_height, 5, vehicle_name=key).join()   # 移动到指定的位置：（x, y, z），速度为2m/s

    def _get_obs(self):
        # 获取无人机状态估计信息
        drone_state = []
        for key, value in self.drones.items():
            drone_state.append(self.client.getMultirotorState(vehicle_name=key))
        # print("leader: ", drone_state[0])
        # print_split()
        # print("follower: ", drone_state[1])

        # 长机的状态
        self.state["leader"]["prev_position"] = self.state["leader"]["positions"]
        # print(type(self.state["leader"]["prev_position"]))
        self.state["leader"]["positions"] = drone_state[0].kinematics_estimated.position
        # print(type(self.state["leader"]["positions"]))
        self.state["leader"]["velocity"] = drone_state[0].kinematics_estimated.linear_velocity
        # print(type(self.state["leader"]["velocity"]))
        self.state["leader"]["yaw_angle"] = airsim.to_eularian_angles(
            drone_state[0].kinematics_estimated.orientation)[2]
        # print(type(self.state["leader"]["heading_angle"]))

        # 僚机的状态
        self.state["follower"]["prev_position"] = self.state["follower"]["positions"]
        self.state["follower"]["positions"] = drone_state[1].kinematics_estimated.position
        self.state["follower"]["velocity"] = drone_state[1].kinematics_estimated.linear_velocity
        self.state["follower"]["yaw_angle"] = airsim.to_eularian_angles(
            drone_state[1].kinematics_estimated.orientation)[2]

        # 获取碰撞信息
        self.state["leader"]["collision"] = drone_state[0].collision.has_collided     # 长机
        self.state["follower"]["collision"] = drone_state[1].collision.has_collided   # 僚机

        state = [
            vec3_sub(self.state["leader"]["velocity"], self.state["follower"]["velocity"]),     # 长机与僚机的相对速度
            vec3_sub(self.state["leader"]["positions"], self.state["follower"]["positions"]),   # 长机与僚机的相对航向角
            self.state["leader"]["yaw_angle"][0] - self.state["follower"]["yaw_angle"][0],      # 队形误差
            self.state["follower"]["prev_command"]["velocity"],                                 # 上一时刻的速度指定
            self.state["follower"]["prev_command"]["yaw_angle"],                                # 上一时刻的偏航指令
        ]

        # print(vec3_sub(self.state["leader"]["velocity"], self.state["follower"]["velocity"]))
        # print(type(vec3_sub(self.state["leader"]["velocity"], self.state["follower"]["velocity"])))

        return state

    def _do_action(self, action):
        """指行动作"""
        vd, pd = self.interpret_action(action)
        if vd <= self.range["velocity"][0]:
            vd = self.range["velocity"][0]
        elif vd >= self.range["velocity"][1]:
            vd = self.range["velocity"][1]

        if pd <= self.range["heading_angle"][0]:
            pd = self.range["heading_angle"][0]
        elif pd >= self.range["heading_angle"][1]:
            pd = self.range["heading_angle"][1]

        speed = self.state["follower"]["velocity"]
        yaw = airsim.to_eularian_angles(self.client.getMultirotorState(vehicle_name="UAV2").
                                        kinematics_estimated.orientation)[2]
        self.client.rotateToYawAsync(yaw + pd, vehicle_name="UAV2").join()
        self.client.moveByVelocityZAsync(speed.x_val + vd, 0,
                                         self.state["follower"]["positions"].z_val, 1, vehicle_name="UAV2").join()

        self.state["follower"]["prev_command"]["velocity"] = speed.x_val + vd
        self.state["follower"]["prev_command"]["yaw_angle"] = yaw + pd

    def _compute_reward(self, state):
        """动作执行后所获得的计算奖励"""
        if 0 <= np.abs(state[0]) <= 0.1:
            r1 = 10
        elif 0.1 < np.abs(state[0]) <= 5:
            r1 = -np.tanh(np.abs(state[0]) - 3)
        else:
            r1 = -1

        if np.abs(state[1]) <= 1:
            r2 = 10
        elif 1 < np.abs(state[1]) <= 10:
            r2 = 5
        else:
            r2 = -1

        r3 = -0.001 * np.abs(np.power(self.state["follower"]["prev_command"]["velocity"], 2))

        r4 = -0.001 * np.abs(np.power(self.state["follower"]["prev_command"]["yaw_angle"], 2))

        return r1 + r2 + r3 + r4

    def step(self, action):
        self.current_step += 1
        # 执行动作
        self._do_action(action)
        # 观察环境
        obs = self._get_obs()
        # 计算奖励
        reward = self._compute_reward(obs)
        # 返回状态等相关信息

        if self.current_step >= self.max_episode_steps:
            truncated = True
        else:
            truncated = False

        if self.state["follower"]["collision"]:
            terminated = True
        else:
            terminated = False

        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self):
        """重置回合"""
        self._setup_flight()
        return self._get_obs()

    @staticmethod
    def interpret_action(action):
        """执行某一种动作策略"""
        a1 = action[0]
        a2 = action[1]

        # 速度
        if a1 == -0:
            vd = - 5
        elif a1 == 1:
            vd = 0
        else:
            vd = 5

        # 偏航角
        if a2 == 0:
            pd = -5
        elif a2 == 1:
            pd = 0
        else:
            pd = 5

        return vd, pd

    def close(self):
        """停止"""
        self.client.reset()
        time.sleep(2)
        for key, value in self.drones.items():
            self.client.enableApiControl(False, vehicle_name=key)
            self.client.armDisarm(False, vehicle_name=key)
