import numpy as np
import torch

class SumTree:
    """求和树数据结构 - 用于高效的优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority, data_index):
        """添加优先级"""
        tree_index = data_index + self.capacity - 1
        self.update(tree_index, priority)

    def update(self, tree_index, priority):
        """更新优先级"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # 向上传播变化
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        """根据值获取叶子节点"""
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index

    @property
    def total_priority(self):
        """返回总优先级"""
        return self.tree[0]


class ReplayBuffer:
    """支持优先级采样"""
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.num_agents = args.num_uavs
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        self.use_per = getattr(args, 'use_per', False)  # 是否使用PER

        if self.use_per:
            self.tree = SumTree(self.buffer_size)
            self.alpha = getattr(args, 'per_alpha', 0.6)
            self.beta = getattr(args, 'per_beta', 0.4)
            self.beta_increment = getattr(args, 'per_beta_increment', 0.001)
            self.epsilon = 1e-6  # 防止优先级为0
            self.max_priority = 1.0

            print(f"✓ 使用优先级经验回放 (PER)")
            print(f"  Alpha={self.alpha}, Beta={self.beta}, Beta增长={self.beta_increment}")
        else:
            print(f"✓ 使用标准经验回放")

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # 如果使用PER，添加优先级
        if self.use_per:
            priority = self.max_priority ** self.alpha
            self.tree.add(priority, self.ptr)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, device='cpu'):
        """
        采样批次数据
        返回：
            - 如果use_per=False: 返回7个tensor (原接口)
            - 如果use_per=True: 返回9个tensor (多了indices和is_weights)
        """
        if not self.use_per:
            indices = np.random.randint(0, self.size, size=batch_size)

            obs = torch.FloatTensor(self.obs[indices]).to(device)
            states = torch.FloatTensor(self.states[indices]).to(device)
            actions = torch.FloatTensor(self.actions[indices]).to(device)
            rewards = torch.FloatTensor(self.rewards[indices]).to(device)
            next_obs = torch.FloatTensor(self.next_obs[indices]).to(device)
            next_states = torch.FloatTensor(self.next_states[indices]).to(device)
            dones = torch.FloatTensor(self.dones[indices]).to(device)

            return obs, states, actions, rewards, next_obs, next_states, dones

        else:
            # ========== 优先级采样 ==========
            indices = []
            priorities = []
            segment = self.tree.total_priority / batch_size

            # 增加beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                value = np.random.uniform(a, b)

                tree_index, priority, data_index = self.tree.get_leaf(value)

                indices.append(data_index)
                priorities.append(priority)

            indices = np.array(indices)

            # 计算重要性采样权重
            sampling_probs = np.array(priorities) / self.tree.total_priority
            is_weights = np.power(self.size * sampling_probs, -self.beta)
            is_weights /= is_weights.max()  # 归一化

            # 转换为tensor
            obs = torch.FloatTensor(self.obs[indices]).to(device)
            states = torch.FloatTensor(self.states[indices]).to(device)
            actions = torch.FloatTensor(self.actions[indices]).to(device)
            rewards = torch.FloatTensor(self.rewards[indices]).to(device)
            next_obs = torch.FloatTensor(self.next_obs[indices]).to(device)
            next_states = torch.FloatTensor(self.next_states[indices]).to(device)
            dones = torch.FloatTensor(self.dones[indices]).to(device)
            is_weights = torch.FloatTensor(is_weights).to(device)

            return obs, states, actions, rewards, next_obs, next_states, dones, indices, is_weights

    def update_priorities(self, indices, td_errors):
        """更新优先级 - 仅在use_per=True时调用"""
        if not self.use_per:
            return

        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx + self.tree.capacity - 1, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size


'''class ReplayBuffer:
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.num_agents = args.num_uavs
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        """添加经验"""
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, device='cpu'):
        """采样批次数据"""
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = torch.FloatTensor(self.obs[indices]).to(device)
        states = torch.FloatTensor(self.states[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_obs = torch.FloatTensor(self.next_obs[indices]).to(device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)

        return obs, states, actions, rewards, next_obs, next_states, dones

    def __len__(self):
        return self.size'''