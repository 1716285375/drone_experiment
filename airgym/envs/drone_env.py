import os

import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
from PIL import Image

import gym
from gym import spaces

from experiment.airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length  # 无人机执行动作时的偏移步长
        self.image_shape = image_shape

        self.img_dir = os.path.join(os.getcwd(), 'data', 'images')    # 图片存储目录

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        self.takeoff_height = -1  # 设定的飞行高度，例如10米

        self.goal = np.array([18, 19, -1])

        self.drone = airsim.MultirotorClient(ip=ip_address)  # 加载无人机
        self.action_space = spaces.Discrete(5)  # 动作空间：5种动作
        self._setup_flight()    # 初始化无人机状态

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        pass

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToZAsync(self.takeoff_height, 3).join()   # 移动到指定的位置：（x, y, z），速度为10m/s
        self.drone.moveByVelocityAsync(1, -0.6, 0, 1).join()    # 以指定x, y, z方向的速度移动，持续时间：5s

    def transform_obs(self, responses):
        """处理无人机获取的图片"""
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([1, 84, 84])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        """指行动作"""
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        # if abs(quad_vel.z_val - self.takeoff_height) > 0.1:  # 如果高度偏差较大，调整高度
        #     vz = (quad_vel.z_val - self.takeoff_height) * 0.1
        # else:
        #     vz = 0
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            0,
            3).join()

    def _compute_reward(self):
        """动作执行后所获得的计算奖励"""
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )
        dist = np.sqrt(np.power((self.goal[0] - quad_pt[0]), 2) +
                       np.power((self.goal[1] - quad_pt[1]), 2) +
                       np.power((self.goal[2] - quad_pt[2]), 2)
                       )

        done = 0
        if self.state["collision"] or dist > 80:
            reward = -1
            if self.state["collision"]:
                done = 1
        elif 40 < dist < 80:
            reward = -dist / 40 + 1
        elif 1 < dist < 40:
            reward = pow(0.5, 0.15 * dist)
        else:
            reward = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        """重置回合"""
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        """执行某一种动作策略"""
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def close(self):
        """停止"""
        self.drone.reset()
        time.sleep(2)
        self.drone.enableApiControl(False)
        self.drone.armDisarm(False)
