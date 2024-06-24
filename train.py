import random
import math

import gym
from gym.envs.registration import register
from experiment.airgym.envs.drone_env import AirSimDroneEnv

import airsim

import torch

from tqdm import tqdm
import matplotlib.pyplot as plt


import warnings
# 屏蔽UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略所有 DeprecationWarning 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


from model.model import *
from common.utils import *


register(
    id='AirSimEnv-v0',
    entry_point='__main__:AirSimDroneEnv',
    kwargs={
        'ip_address': '127.0.0.1',
        'step_length': 0.5,
        'image_shape': (144, 256, 1)}
)

BATCH_SIZE = 32        # 每次训练批次大小
MAX_CAPACITY = 1000    # 经验回放区最大容量

INPUT_CHANNELS = 1      # 输入的图像通道数
NUM_ACTIONS = 5         # 动作数量

epsilon = 0.1
epsilon_end = 0.01
epsilon_start = 0.90
epsilon_decay = 1000
frame_index = 1

learning_rate = 0.001

GAMMA = 0.99            # 时间折扣率

NUM_EPISODES = 500     # 最大尝试次数


count = 0
target_update = 100

MINIMAL_SIZE = 4        #

replay_buffer = ReplayBuffer(MAX_CAPACITY)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


if __name__ == "__main__":
    # from gym.utils.env_checker import check_env
    # env = gym.make('AirSimEnv-v0')
    # check_env(env)

    # model = CNN(env.observation_space.shape, env.action_space.n)
    # target_model = CNN(env.observation_space.shape, env.action_space.n)
    # memory = Memory(1000)
    # dqn_learn(env, model, target_model, memory, device=DEVICE)

    # for i in range(10):
    #     with tqdm(total=int(EPOCH_LENGTH / 10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(EPOCH_LENGTH / 10)):
    #             episode_return = 0
    #             env.reset()
    #             done = False
    #             while not done:
    #                 next_state, reward, done, _ = env.step(action)
    #                 replay_buffer.add(state, action, reward, next_state, done)
    #                 state = next_state
    #                 episode_return += reward
    #                 # 当buffer数据的数量超过一定值后,才进行Q网络训练
    #                 if len(replay_buffer) > MINIMAL_SIZE:
    #                     b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(BATCH_SIZE)
    #                     transition_dict = {
    #                         'states': b_s,
    #                         'actions': b_a,
    #                         'next_states': b_ns,
    #                         'rewards': b_r,
    #                         'dones': b_d
    #                     }
    #                     agent.update(transition_dict)
    #             return_list.append(episode_return)
    #             if (i_episode + 1) % 10 == 0:
    #                 pbar.set_postfix({
    #                     'episode':
    #                         '%d' % (num_episodes / 10 * i + i_episode + 1),
    #                     'return':
    #                         '%.3f' % np.mean(return_list[-10:])
    #                 })
    #             pbar.update(1)
    model = CNN(INPUT_CHANNELS, NUM_ACTIONS)
    # 加载模型参数
    # model.load_state_dict(torch.load("./result/trained_mod/mod-rl-v0.pth"))

    target_model = CNN(INPUT_CHANNELS, NUM_ACTIONS)
    model.to(DEVICE)
    target_model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    env = gym.make('AirSimEnv-v0')
    return_list = []

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        # print(obs.shape)
        done = False
        episode_return = 0
        print("**Episode: {}**".format(episode,))
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * frame_index / epsilon_decay)
            frame_index += 1
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                # print(state.shape)
                action = model(state.to(DEVICE)).argmax().item()
            next_obs, reward, done, _ = env.step(action)
            print("Action: {} | Reward: {} | done: {}".format(action, reward, done))

            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            if len(replay_buffer) > BATCH_SIZE:
                # update_q_function()
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
                actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(DEVICE)
                rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(DEVICE)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
                dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(DEVICE)

                q_values = model(states).gather(1, actions)     # 按照gather中的张量元素的大小顺序重新排列
                max_next_q_values = target_model(next_states).max(1)[0].detach()    # max(1)返回最大索引，max(1)[0]返回最大索引对应的值
                target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

                loss = criterion(target_q_values, q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if count % target_update == 0:
                    target_model.load_state_dict(
                        model.state_dict())  # 更新目标网络
                count += 1

            episode_return += reward

            # env.render()
        print("return: {}".format(episode_return))
        print("------------------------------------------------------")
        return_list.append(episode_return)
        if episode > 0 and episode % MINIMAL_SIZE/10 == 0:
            x_ = np.arange(len(return_list))
            plt.plot(x_, return_list)
            plt.show()
    env.close()
    save_model(target_model.state_dict(), "./result/trained_mod")
