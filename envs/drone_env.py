import os
import time
import numpy as np
import json
import threading

import airsim
import gymnasium as gym
from gymnasium import spaces
from envs.airsim_env import AirSimEnv
from envs.drones import Drone
from common.airsim_utils import *


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip, image_shape):
        super().__init__(image_shape)
        self.ip_address = ip                                       # 调用者的ip地址
        self.airsim = Drone(ip=self.ip_address)                    # 无人机客户端
        self.drone_client = self.airsim.clients

        self.leader = self.drone_client[0]
        self.follower = self.drone_client[1]
        self.measure = self.drone_client[2]                         # 单开一个client轮询状态

        self.airsim_cfg = json.load(open(r"C:\Users\jie\Documents\AirSim\settings.json", encoding='utf-8'))        # 加载AirSim配置文件
        self.drone_name = {}    # 无人机名字
        self.drone_id2key = {}

        self.current_step = 0
        self.max_episode_steps = 5

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

        self.is_leader_thread_activa = False        # 标识控制leader无人机的线程是否开启
        self.is_follower_get_observe = False        # 标识follwer无人机是否在观察
        self.can_get_observe = False
        self.is_leader_thread_first_start = True           # 判读是否线程第一次启动
        self.thread = threading.Thread(target=self.fly_with_thread)
        self.initialize()

    def __del__(self):
        pass

    def initialize(self):
        print_split()
        print(self.drone_client)

        print_split()
        print(self.leader)

        print_split()
        print(self.follower)

        # 读取所有无人机的名字
        for i, (key, value) in enumerate(self.airsim_cfg["Vehicles"].items()):
            self.drone_name[key] = value
            self.drone_id2key[i] = key

        print_split()
        print(self.drone_name)

        print_split()
        print(self.drone_id2key)

        time.sleep(0.2)

    def fly_with_thread(self):
        while True:
            vx = 5
            vy = 0
            # print_output("begin: leader is flying")
            self.can_get_observe = False
            if not self.is_follower_get_observe and self.is_leader_thread_activa:
                self.leader.moveByVelocityZAsync(vx, vy, self.state["pos-z"], 1, vehicle_name=self.drone_id2key[0]).join()
                self.can_get_observe = True
            time.sleep(1)
            # print_output("end: leader is flying")
            pass
        pass

    def _setup_flight(self):
        if self.is_leader_thread_activa:
            self.is_leader_thread_activa = False

        self.leader.reset()
        self.follower.reset()

        print(self.leader)
        print(self.follower)
        time.sleep(0.2)

        self.current_step = 0
        takeoff_height = self.state["pos-z"]

        self.leader.enableApiControl(True, vehicle_name=self.drone_id2key[0])
        self.leader.armDisarm(True, vehicle_name=self.drone_id2key[0])

        self.follower.enableApiControl(True, vehicle_name=self.drone_id2key[1])
        self.follower.armDisarm(True, vehicle_name=self.drone_id2key[1])

        self.leader.moveToZAsync(takeoff_height, 5, vehicle_name=self.drone_id2key[0])  # 移动到指定的位置：（x, y, z），速度为2m/s
        self.follower.moveToZAsync(takeoff_height, 5, vehicle_name=self.drone_id2key[1]).join()   # 移动到指定的位置：（x, y, z），速度为2m/s

        time.sleep(0.2)

        if self.is_leader_thread_first_start:
            self.is_leader_thread_first_start = False
            self.is_leader_thread_activa = True
            print_output("debug-start thread")
            self.thread.start()
            print_output("debuf-end thread")

        else:
            self.is_leader_thread_activa = True
        # else:
        #     self.is_leader_thread_activa = False

    def _get_obs(self):
        # 获取无人机状态估计信息
        self.is_follower_get_observe = True
        if not self.can_get_observe:
            time.sleep(1)

        drone_state = [self.leader.getMultirotorState(vehicle_name=self.drone_id2key[0]),
                       self.follower.getMultirotorState(vehicle_name=self.drone_id2key[1])]
        # print_split()
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
            self.state["leader"]["yaw_angle"] - self.state["follower"]["yaw_angle"],            # 队形误差
            self.state["follower"]["prev_command"]["velocity"],                                 # 上一时刻的速度指定
            self.state["follower"]["prev_command"]["yaw_angle"],                                # 上一时刻的偏航指令
        ]

        # print(vec3_sub(self.state["leader"]["velocity"], self.state["follower"]["velocity"]))
        # print(type(vec3_sub(self.state["leader"]["velocity"], self.state["follower"]["velocity"])))

        self.is_follower_get_observe = False
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

        yaw = airsim.to_eularian_angles(self.follower.getMultirotorState(vehicle_name=self.drone_id2key[1]).
                                        kinematics_estimated.orientation)[2]
        self.follower.rotateToYawAsync(yaw + pd, vehicle_name=self.drone_id2key[1]).join()
        self.follower.moveByVelocityZAsync(speed.x_val + vd, 0,
                                           self.state["follower"]["positions"].z_val, 1,
                                           vehicle_name=self.drone_id2key[1]).join()

        self.state["follower"]["prev_command"]["velocity"] = speed.x_val + vd
        self.state["follower"]["prev_command"]["yaw_angle"] = yaw + pd

    def _compute_reward(self, state):
        """动作执行后所获得的计算奖励"""
        if 0 <= vec3_magnitude(state[0]) <= 0.1:
            r1 = 10
        elif 0.1 < vec3_magnitude(state[0]) <= 5:
            r1 = -np.tanh(vec3_magnitude(state[0]) - 3)
        else:
            r1 = -1

        if vec3_magnitude(state[1]) <= 1:
            r2 = 10
        elif 1 < vec3_magnitude(state[1]) <= 10:
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
            self.is_leader_thread_activa = False
        else:
            truncated = False

        if self.state["follower"]["collision"] or self.state["leader"]["collision"]:
            terminated = True
            self.is_leader_thread_activa = False
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
        # self.leader.reset()
        # self.follower.reset()
        #
        #
        # self.leader.enableApiControl(False, vehicle_name=self.drone_id2key[0])
        # self.leader.armDisarm(False, vehicle_name=self.drone_id2key[0])
        #
        # self.follower.enableApiControl(False, vehicle_name=self.drone_id2key[1])
        # self.follower.armDisarm(False, vehicle_name=self.drone_id2key[1])
        pass
