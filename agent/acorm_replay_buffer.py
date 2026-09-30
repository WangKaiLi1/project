import numpy as np
import torch


class AcormReplayBuffer:
    """
    存储内容:
    - obs_n: 观测
    - s: 全局状态
    - last_a_n: 上一步角色
    - a_n: 当前角色
    - r: 奖励
    - dw: done标志
    - active: 有效标志
    """
    def __init__(self, args):
        self.N = args.num_uavs
        self.obs_dim = args.upper_obs_dim
        self.state_dim = args.upper_state_dim
        self.action_dim = args.upper_action_dim
        self.step_limit = args.episode_length  # 大时间步步数
        self.buffer_size = args.upper_buffer_size

        self.buffer_num = 0  # 当前buffer编号
        self.current_size = 0

        # 缓冲区初始化
        self.buffer = {
            # 观测: (buffer_size, episode_limit+1, N, obs_dim)
            'obs_n': np.zeros([self.buffer_size, self.step_limit + 1, self.N, self.obs_dim]),
            # 全局状态: (buffer_size, episode_limit+1, state_dim)
            's': np.zeros([self.buffer_size, self.step_limit + 1, self.state_dim]),
            # 上一步角色: (buffer_size, episode_limit+1, N)
            'last_a_n': np.zeros([self.buffer_size, self.step_limit + 1, self.N]),
            # 当前动作: (buffer_size, episode_limit, N)
            'a_n': np.zeros([self.buffer_size, self.step_limit + 1, self.N]),
            # 角色选择: (buffer_size, episode_limit, N)
            #'role_n': np.zeros([self.buffer_size, self.step_limit, self.N]),
            # 奖励: (buffer_size, episode_limit, 1)
            'r': np.zeros([self.buffer_size, self.step_limit, 1]),
            # done标志: (buffer_size, episode_limit, 1)
            'dw': np.zeros([self.buffer_size, self.step_limit, 1]),
            # 有效标志: (buffer_size, episode_limit, 1)
            'active': np.zeros([self.buffer_size, self.step_limit, 1])
        }
        self.transition_lens = np.zeros(self.buffer_size)

    def store_transition(self, step, s, obs_n, last_a_n, a_n, r, dw):
        """
        Args:
            step: 当前步数
            obs_n: (N, obs_dim) 观测
            s: (state_dim,) 全局状态
            last_a_n: (N, ) 上一步角色
            a_n: (N, ) 当前角色
            r: float 奖励
            dw: bool done标志
        """
        self.buffer['obs_n'][self.buffer_num][step] = obs_n
        self.buffer['s'][self.buffer_num][step] = s
        self.buffer['last_a_n'][self.buffer_num][step] = last_a_n
        self.buffer['a_n'][self.buffer_num][step] = a_n
        self.buffer['r'][self.buffer_num][step] = r
        self.buffer['dw'][self.buffer_num][step] = dw
        self.buffer['active'][self.buffer_num][step] = 1.0

    def store_last_step(self, step, s, obs_n, last_a_n):
        """
        存储最后一步 (用于计算下一状态的Q值)
        Args:
            episode_step: 最后步数
            obs_n: (N, obs_dim) 最终观测
            s: (state_dim,) 最终全局状态
            last_a_n: (N, ) 最后动作
        """
        self.buffer['obs_n'][self.buffer_num][step] = obs_n
        self.buffer['s'][self.buffer_num][step] = s
        self.buffer['last_a_n'][self.buffer_num][step] = last_a_n
        #self.buffer['a_n'][self.buffer_num][step] = np.zeros([self.N, self.action_dim])
        self.buffer['active'][self.buffer_num][step:] = 0
        self.transition_lens[self.buffer_num] = step  # 记录这条经验的长度
        self.buffer_num = (self.buffer_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        Args:
            batch_size: 采样数量
        Returns:
            batch: dict of tensors
            max_episode_len: 最大episode长度
        """
        index = np.random.choice(self.current_size, size=batch_size, replace=False)
        max_len = int(np.max(self.transition_lens[index]))

        batch = {}
        for key in self.buffer.keys():
            if key in ['obs_n', 's']:
                # 需要max_len + 1步
                batch[key] = torch.tensor(self.buffer[key][index, :max_len + 1], dtype=torch.float32)
            elif key == 'last_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_len + 1], dtype=torch.long)
            elif key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_len], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_len], dtype=torch.float32)

        return batch, max_len

    def __len__(self):
        return self.current_size

    def can_sample(self, batch_size):
        return self.current_size >= batch_size