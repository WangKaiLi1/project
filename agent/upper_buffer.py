import numpy as np
import torch


class UpperReplayBuffer:
    """
    上层经验回放池
    存储大时间步的轨迹
    """
    def __init__(self, args):
        self.buffer_size = args.dqn_buffer_size
        self.num_agents = args.num_uavs
        self.upper_state_dim = args.upper_state_dim  # args.history_window_size + 
        self.upper_states = np.zeros((self.buffer_size, self.upper_state_dim), dtype=np.float32)
        self.roles = np.zeros((self.buffer_size, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_upper_states = np.zeros((self.buffer_size, self.upper_state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, upper_state, role, reward, next_upper_state, done):
        """添加经验"""
        self.upper_states[self.ptr] = upper_state
        self.roles[self.ptr] = role
        self.rewards[self.ptr] = reward
        self.next_upper_states[self.ptr] = next_upper_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, device='cpu'):
        """采样批次数据"""
        indices = np.random.randint(0, self.size, size=batch_size)

        upper_states = torch.FloatTensor(self.upper_states[indices]).to(device)
        roles = torch.FloatTensor(self.roles[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_upper_states = torch.FloatTensor(self.next_upper_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)

        return upper_states, roles, rewards, next_upper_states, dones

    def __len__(self):
        return self.size