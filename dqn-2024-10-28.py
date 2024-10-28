import yaml
import time
# from Q_learning_brain import QLearningTable
from drone_position_ctrl_env import DronePosCtrl
from Q_learning_and_Sarsa_brain import SarsaTable
from Q_learning_and_Sarsa_brain import QLearningTable

max_episodes = 100


def q_learning_start():
    for episode in range(max_episodes):
        # initial observation
        env.AirSim_client.reset()
        env.env_setting()
        time.sleep(2)
        env.takeoff()
        time.sleep(3)
        observation = env.reset()

        while True:
            # environment refresh
            env.render()

            # choose action based on observation
            action = q_learning_client.choose_action(str(observation))
            print('observation: ', observation)

            # take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            print('next observation: ', next_observation)
            print('reward: ', reward)

            # to learn from this transition
            q_learning_client.learn(str(observation), action, reward, str(next_observation))

            # refresh observation
            observation = next_observation

            if done:
                break

    print('Learning process over!')
    env.reset()


def sarsa_learning_start():
    for episode in range(max_episodes):
        # initial observation
        env.AirSim_client.reset()
        env.env_setting()
        time.sleep(2)
        env.takeoff()
        time.sleep(3)
        observation = env.reset()

        action = sarsa_learning_client.choose_action(str(observation))

        while True:
            # environment refresh
            env.render()

            # take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            print('next observation: ', next_observation)
            print('reward: ', reward)

            # choose action based on observation
            next_action = sarsa_learning_client.choose_action(str(next_observation))

            # to learn from this transition
            sarsa_learning_client.learn(str(observation), action, reward, str(next_observation), next_action)

            # refresh observation
            observation = next_observation
            action = next_action

            if done:
                break

    print('Learning process over!')
    env.reset()


if __name__ == '__main__':
    with open('../data/configs.yaml', 'r', encoding='utf-8') as configs_file:
        _configs = yaml.load(configs_file.read(), Loader=yaml.FullLoader)

    env = DronePosCtrl(configs=_configs, vehicle_index=0)
    q_learning_client = QLearningTable(actions=list(range(env.n_actions)))
    sarsa_learning_client = SarsaTable(actions=list(range(env.n_actions)))

    q_learning_start()
    # sarsa_learning_start()
    q_learning_client.show_q_table()
    # sarsa_learning_client.show_q_table()

二、Q-learning和Sarsa learning代码实现
import numpy as np
import pandas as pd


class BaseRL(object):
    def __init__(self, action_spaces, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_spaces
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append this state to the table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        else:
            pass

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.rand() < self.epsilon:
            # choose the optimal action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # randomly select a action
            action = np.random.choice(self.actions)

        return action

    def learn(self, *args):
        pass

    def show_q_table(self):
        print('Q-table:\n', self.q_table)


# off-policy
class QLearningTable(BaseRL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(state=state)
        q_predict = self.q_table.loc[state, action]
        if next_state != 'terminal':
            q_target = reward   self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward

        self.q_table.loc[state, action]  = self.learning_rate * (q_target - q_predict)


# on-policy
class SarsaTable(BaseRL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if next_state != 'terminal':
            # next state is not terminal
            q_target = reward   self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward

        self.q_table.loc[state, action]  = self.learning_rate * (q_target - q_predict)

三、环境交互实现
import sys
import time
import yaml
import airsim
import random
import threading
import numpy as np

sys.path.append('..')


class DronePosCtrl(object):
    def __init__(self, configs, vehicle_index):

        self.configs = configs

        # >---------------->>>  label for threading   <<<----------------< #
        # 方便开多线程单独控制每台无人机
        self.drone_index = vehicle_index
        self.base_name = configs['base_name']
        self.now_drone_name = self.base_name   str(vehicle_index)
        # >---------------->>>   --------------------------------------    <<<----------------< #

        # >---------------->>>  position settings   <<<----------------< #
        self.target_position = [8.0, 0.0, 2.0]
        self.myself_position = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0}

        # 极半径常量
        self.polar_radius = 6356725
        # 赤道半径常量
        self.equatorial_radius = 6378137
        # 记录原点的 gps 纬度, 经度以及高度
        self.origin_info = {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}

        # >---------------->>>   --------------------------------------    <<<----------------< #

        # , 'move front', 'move back'
        self.action_spaces = ['move_front', 'move_back']
        self.n_actions = len(self.action_spaces)

        if configs['multi_computing']:
            # create API client for ctrl
            self.AirSim_client = airsim.MultirotorClient(str(configs['simulation_address']))

        else:
            self.AirSim_client = airsim.MultirotorClient()

        self.AirSim_client.confirmConnection()
        self.env_setting()
        # self.takeoff()

    def env_setting(self):
        if self.drone_index == -1:
            for index in self.configs['vehicle_index']:
                self.AirSim_client.enableApiControl(True, vehicle_name=self.base_name str(index))
                self.AirSim_client.armDisarm(True, vehicle_name=self.base_name   str(index))
        else:
            self.AirSim_client.enableApiControl(True, vehicle_name=self.now_drone_name)
            self.AirSim_client.armDisarm(True, vehicle_name=self.now_drone_name)

    def reset(self):
        # self.AirSim_client.reset()
        # for index in self.configs['vehicle_index']:
        #     self.AirSim_client.enableApiControl(False, vehicle_name=self.base_name   str(index))
        #     self.AirSim_client.armDisarm(False, vehicle_name=self.base_name   str(index))

        gt_dict = self.get_ground_truth_pos(vehicle_name=self.now_drone_name)
        return gt_dict['position']

    def takeoff(self):

        if self.AirSim_client.getMultirotorState().landed_state == airsim.LandedState.Landed:
            print(f'Drone{self.drone_index} is taking off now···')
            if self.drone_index == -1:
                for index in self.configs['vehicle_index']:
                    # 需要判断是不是最后那台
                    if not index == self.configs['vehicle_index'][len(self.configs['vehicle_index']) - 1]:
                        self.AirSim_client.takeoffAsync(timeout_sec=10, vehicle_name=self.base_name str(index))
                    else:
                        self.AirSim_client.takeoffAsync(timeout_sec=10, vehicle_name=self.base_name str(index)).join()
            elif self.drone_index == self.configs['target_vehicle_index']:
                self.AirSim_client.takeoffAsync(timeout_sec=10, vehicle_name=self.now_drone_name).join()
            else:
                self.AirSim_client.takeoffAsync(timeout_sec=10, vehicle_name=self.now_drone_name)
        else:
            print(f'Drone{self.drone_index} is flying··· ')
            if self.drone_index == -1:
                for index in self.configs['vehicle_index']:
                    # 需要判断是不是最后那台
                    if not index == self.configs['vehicle_index'][len(self.configs['vehicle_index']) - 1]:
                        self.AirSim_client.hoverAsync(vehicle_name=self.base_name str(index))
                    else:
                        self.AirSim_client.hoverAsync(vehicle_name=self.base_name str(index)).join()
            else:
                self.AirSim_client.hoverAsync(vehicle_name=self.now_drone_name).join()

    def get_ground_truth_pos(self, vehicle_name='Drone0'):
        temp_pos = [0.0, 0.0, 0.0]
        temp_vel = [0.0, 0.0, 0.0]

        vehicle_state = self.AirSim_client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)

        temp_pos[0] = round(vehicle_state.position.x_val, 1)
        temp_pos[1] = round(vehicle_state.position.y_val, 1)
        temp_pos[2] = round(vehicle_state.position.z_val, 1)

        temp_vel[0] = vehicle_state.linear_velocity.x_val
        temp_vel[1] = vehicle_state.linear_velocity.y_val
        temp_vel[2] = vehicle_state.linear_velocity.z_val

        ground_truth_dict = {
            'position': temp_pos,
            'velocity': temp_vel
        }
        return ground_truth_dict

    def move_by_position(self, position_3d, vehicle_name='Drone0'):
        print('position input: ', position_3d)
        # 索引为-1时表示控制全部
        if self.drone_index == -1:
            for drone_index in self.configs['vehicle_index']:
                # 只控制除目标无人机外的所有无人机
                if not drone_index == self.configs['target_vehicle_index']:
                    self.AirSim_client.moveToPositionAsync(position_3d[0], position_3d[1], position_3d[2], timeout_sec=2,
                                                           velocity=2, vehicle_name=self.base_name   str(drone_index))
                else:
                    pass
        else:
            # 对当前线程控制的无人机对象施加持续0.5秒的速度控制
            if vehicle_name != self.now_drone_name:
                self.AirSim_client.moveToPositionAsync(position_3d[0], position_3d[1], position_3d[2], timeout_sec=2,
                                                       velocity=2, vehicle_name=vehicle_name)
            else:
                self.AirSim_client.moveToPositionAsync(position_3d[0], position_3d[1], position_3d[2], timeout_sec=2,
                                                       velocity=2, vehicle_name=self.now_drone_name)

    def move_by_velocity(self, velocity_3d, vehicle_name='Drone0'):
        print('velocity: ', velocity_3d)
        # 索引为-1时表示控制全部
        if self.drone_index == -1:
            for drone_index in self.configs['vehicle_index']:
                # 只控制除目标无人机外的所有无人机
                if not drone_index == self.configs['target_vehicle_index']:
                    self.AirSim_client.moveByVelocityAsync(velocity_3d[0], velocity_3d[1], velocity_3d[2],
                                                           duration=0.6, drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                           yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                                                           vehicle_name=self.base_name   str(drone_index))
                else:
                    pass
        else:
            # 对当前线程控制的无人机对象施加持续0.5秒的速度控制
            if vehicle_name != self.now_drone_name:
                self.AirSim_client.moveByVelocityAsync(velocity_3d[0], velocity_3d[1], velocity_3d[2], duration=0.6,
                                                       vehicle_name=vehicle_name)
            else:
                self.AirSim_client.moveByVelocityAsync(velocity_3d[0], velocity_3d[1], velocity_3d[2], duration=0.6,
                                                       vehicle_name=self.now_drone_name)

    def step(self, action):
        status = self.get_ground_truth_pos()
        now_position = status['position']
        desired_velocity = [0.0, 0.0, 0.0]
        desired_position = now_position
        desired_position[2] = 0.0

        # move ahead
        if self.action_spaces[action] == self.action_spaces[0]:
            if now_position[0] < self.target_position[0]:
                desired_velocity[0] = 2.0
                desired_position[0]  = 1.5
            else:
                desired_velocity[0] = 0.0
                desired_position[0]  = 0.0

        # move back
        elif self.action_spaces[action] == self.action_spaces[1]:
            if now_position[0] > 0:
                desired_velocity[0] = -2.0
                desired_position[0] -= 1.5
            else:
                desired_velocity[0] = 0.0
                desired_position[0] -= 0.0

        # self.move_by_velocity(desired_velocity)
        self.move_by_position(desired_position)
        time.sleep(2)
        self.AirSim_client.hoverAsync(vehicle_name=self.now_drone_name).join()

        status = self.get_ground_truth_pos()
        next_position = status['position']

        if now_position[0] >= self.target_position[0]:
            reward = 100
            done = True
            next_position = 'terminal'
            print('task finished!')
        else:
            if next_position[0] - now_position[0] < 0:
                reward = -10
            else:
                reward = 0

            done = False

            if now_position[0] <= -1:
                reward = -100
                done = True
                next_position = 'terminal'

        return next_position, reward, done

    def render(self):
        pass

    # def env_test(self):
    #     # state = env.reset()
    #
    #     for i in range(10):
    #         action_index = random.randint(0, len(self.action_spaces)-1)
    #         action = self.action_spaces[action_index]
    #         state, reward, done = env.step(action)
    #
    #         if done:
    #             env.reset()
    #             return None


# if __name__ == '__main__':
#     with open('../data/configs.yaml', 'r', encoding='utf-8') as configs_file:
#         _configs = yaml.load(configs_file.read(), Loader=yaml.FullLoader)
#
#     env = DronePosCtrl(configs=_configs, vehicle_index=0)
#     env.env_test()