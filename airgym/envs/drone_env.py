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
        self.ip_address = ip_address
        self.step_length = step_length  # 无人机执行动作时的偏移步长
        self.steps = 0
        self.image_shape = image_shape

        self.img_dir = os.path.join(os.getcwd(), 'data', 'images')    # 图片存储目录

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        self.takeoff_height = -5  # 设定的飞行高度，例如10米

        # self.goal = np.array([np.random.randint(300, 601), np.random.randint(300, 601), -1])
        self.goal = np.array([33, -34, -5])

        self.drone = airsim.MultirotorClient(ip=self.ip_address)  # 加载无人机
        self.action_space = spaces.Discrete(5)  # 动作空间：5种动作
        # self._setup_flight()    # 初始化无人机状态

        self.image_request = [
            # png format
            airsim.ImageRequest(0, airsim.ImageType.Scene),
            # uncompressed RGB array bytes
            airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
            # floating point uncompressed image
            airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True)
        ]

    def __del__(self):
        pass

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToZAsync(self.takeoff_height, 3).join()   # 移动到指定的位置：（x, y, z），速度为10m/s
        self.drone.moveByVelocityAsync(1, -0.6, 0, 5).join()    # 以指定x, y, z方向的速度移动，持续时间：5s

    def transform_obs(self, responses):
        """处理无人机获取的图片"""
        img1d = np.array(responses[2].image_data_float, dtype=np.float32)

        # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[2].height, responses[2].width))

        # img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        # img_rgb = img1d.reshape(responses[1].height, responses[1].width, 3)
        # write to png
        # filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # airsim.write_png(os.path.join(self.img_dir, filename + '.png'), img_rgb)

        from PIL import Image

        # image = Image.fromarray(img2d)
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((55, 100)))

        return im_final.reshape([1, 55, 100])

    def _get_obs(self):
        responses = self.drone.simGetImages(self.image_request)
        # print(responses)  # debug
        image = self.transform_obs(responses)
        # print(image.shape)
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
            5).join()

    def _compute_reward(self):
        """动作执行后所获得的计算奖励"""
        quad_pt_curr = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        quad_pt_prev = np.array(
            list(
                (
                    self.state["prev_position"].x_val,
                    self.state["prev_position"].y_val,
                    self.state["prev_position"].z_val,
                )
            )
        )
        dist_curr = np.sqrt(np.power((self.goal[0] - quad_pt_curr[0]), 2) +
                            np.power((self.goal[1] - quad_pt_curr[1]), 2) +
                            np.power((self.goal[2] - quad_pt_curr[2]), 2))

        dist_prev = np.sqrt(np.power((self.goal[0] - quad_pt_prev[0]), 2) +
                            np.power((self.goal[1] - quad_pt_prev[1]), 2) +
                            np.power((self.goal[2] - quad_pt_prev[2]), 2))
        done = 0
        if self.state["collision"]:
            reward = -100
        else:
            if self.steps >= 30:
                reward = -100
                self.steps = 0
                # self.goal = np.array([np.random.randint(300, 601), np.random.randint(300, 601), -1])
                done = 1
            else:
                if dist_curr < 7:
                    reward = 100
                    done = 1
                else:
                    reward = dist_prev - dist_curr

        return reward, done

    def step(self, action):
        self._do_action(action)
        self.steps += 1     # 当前回合次数加1
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
