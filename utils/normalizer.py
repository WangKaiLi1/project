import numpy as np
import torch
from collections import deque


class ActionNormalizer:
    """动作归一化器"""
    def __init__(self, action_dim, num_agents, clip_range=5.0):
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.clip_range = clip_range

        # 统计信息
        self.mean = np.zeros(action_dim)
        self.std = np.ones(action_dim)
        self.count = 0

        # 用于在线更新的缓冲区
        self.buffer = deque(maxlen=10000)

    def update(self, actions):
        """
        更新归一化统计
        actions: [batch_size, num_agents, action_dim] 或 [num_agents, action_dim]
        """
        if len(actions.shape) == 3:
            batch_size = actions.shape[0]
            actions_flat = actions.reshape(-1, self.action_dim)
        else:
            actions_flat = actions.reshape(-1, self.action_dim)

        # 添加到缓冲区
        for act in actions_flat:
            self.buffer.append(act)

        # 定期更新统计
        if len(self.buffer) >= 100:
            buffer_array = np.array(self.buffer)
            self.mean = buffer_array.mean(axis=0)
            self.std = buffer_array.std(axis=0) + 1e-8
            self.count = len(buffer_array)

    def normalize(self, actions):
        """归一化动作"""
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
            normalized = (actions_np - self.mean) / self.std
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            return torch.FloatTensor(normalized).to(actions.device)
        else:
            normalized = (actions - self.mean) / self.std
            return np.clip(normalized, -self.clip_range, self.clip_range)

    def denormalize(self, normalized_actions):
        """反归一化动作"""
        if isinstance(normalized_actions, torch.Tensor):
            actions_np = normalized_actions.detach().cpu().numpy()
            denormalized = actions_np * self.std + self.mean
            return torch.FloatTensor(denormalized).to(normalized_actions.device)
        else:
            return normalized_actions * self.std + self.mean


class ObservationNormalizer:
    """观测归一化器"""
    def __init__(self, obs_dim, clip_range=10.0):
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.mean = np.zeros(obs_dim)
        self.std = np.ones(obs_dim)
        self.count = 0
        self.epsilon = 1e-8

    def update(self, obs_batch):
        """批量更新统计"""
        if isinstance(obs_batch, torch.Tensor):
            obs_batch = obs_batch.cpu().numpy()

        # 展平
        if len(obs_batch.shape) > 2:
            obs_batch = obs_batch.reshape(-1, self.obs_dim)

        # 在线更新均值和方差
        batch_mean = obs_batch.mean(axis=0)
        batch_std = obs_batch.std(axis=0)
        batch_count = obs_batch.shape[0]

        # 合并统计
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # 更新均值
        new_mean = self.mean + delta * batch_count / total_count

        # 更新方差
        m_a = self.std ** 2 * self.count
        m_b = batch_std ** 2 * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_std = np.sqrt(M2 / total_count) + self.epsilon

        self.mean = new_mean
        self.std = new_std
        self.count = total_count

    def normalize(self, obs):
        """归一化观测"""
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
            normalized = (obs_np - self.mean) / self.std
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            return torch.FloatTensor(normalized).to(obs.device)
        else:
            normalized = (obs - self.mean) / self.std
            return np.clip(normalized, -self.clip_range, self.clip_range)