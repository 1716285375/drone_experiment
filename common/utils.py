from collections import deque
import numpy as np
import random
import os

import torch


class ReplayBuffer(object):
    """
    Fixed-size buffer to store experience tuples.
    经验回放缓存区
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)    # 队列,先进先出

    def push(self, state, action, reward, next_state, done):
        """将数据加入buffer"""
        if len(self.buffer) >= self.buffer.maxlen:
            self.buffer.popleft()
        self.buffer.append((state, action, reward, next_state, done))  # 将数据加入buffer

    def sample(self, batch_size):
        """随机检索Batch_Size大小的样本并返回"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        """返回当前buffer的长度"""
        return len(self.buffer)


# # 创建一个 ReplayBuffer 实例
# replay_buffer = ReplayBuffer(100)
#
# # 向缓冲区添加一些经验回放数据
# replay_buffer.push(np.array([1, 2, 3]), 4, 5, (6, 7, 8), False)
# # replay_buffer.push((9, 10, 11), 12, 18, (14, 5, 16), True)
# # replay_buffer.push((9, 20, 17), 16, 14, (4, 15, 16), True)
# # replay_buffer.push((9, 30, 15), 13, 13, (14, 15, 6), True)
#
# state, action, reward, next_state, done = replay_buffer.sample(1)
# print(state, action, reward, next_state, done)
# print(type(state), action, reward, next_state, done)


def update_q_function():
    pass


def update_main_q_function(main_q_network):
    main_q_network.train()
    # 求损失函数
    # 更新连接参数
    pass


def update_target_q_function(target_q_network, main_q_network):
    target_q_network.load_state_dict(main_q_network.state_dict())
    pass


def save_model(model_state_dict, dst_dir, model_name='mod-rl-v0'):
    """假设你已经训练了模型，并希望保存它的状态"""
    if not dst_dir:
        os.makedirs(dst_dir)
    model_path = os.path.join(dst_dir, model_name + '.pth')
    torch.save(model_state_dict, model_path)
