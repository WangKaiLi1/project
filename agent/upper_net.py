import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RoleActor(nn.Module):
    """
    上层角色选择Actor网络（使用LSTM捕捉时序信息）
    输入：滑动窗口 + UAV状态
    输出：角色概率（充电/服务）
    """
    def __init__(self, window_size, uav_state_dim, num_agents, hidden_dim=128, lstm_layers=2):
        super(RoleActor, self).__init__()

        self.window_size = window_size
        self.uav_state_dim = uav_state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        # LSTM处理时序窗口
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间槽的任务数
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # 处理UAV状态
        self.uav_fc = nn.Sequential(
            nn.Linear(uav_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # 输出层：直接输出 N 个无人机的充电概率
        # hidden_dim // 2 -> num_agents
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, self.num_agents),
            nn.Sigmoid()  # 保证输出在 0-1 之间作为概率
        )

    def forward(self, task_window, uav_state):
        """
        task_window: [batch, window_size] - 滑动窗口
        uav_state: [batch, uav_state_dim] - UAV状态
        """
        batch_size = task_window.shape[0]

        # 1. LSTM处理时序窗口
        # [batch, window_size] -> [batch, window_size, 1]
        window_input = task_window.unsqueeze(-1)
        lstm_out, (h_n, c_n) = self.lstm(window_input)
        # 取最后一个时间步的输出
        lstm_feature = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # 2. 处理UAV状态
        uav_feature = self.uav_fc(uav_state)  # [batch, hidden_dim]

        # 3. 特征融合
        combined = torch.cat([lstm_feature, uav_feature], dim=-1)  # [batch, hidden_dim*2]
        fused = self.fusion(combined)  # [batch, hidden_dim//2]

        # 4. 输出充电概率
        charge_prob = self.output(fused)  # [batch, num_agents]

        return charge_prob

    def get_action(self, task_window, uav_state, noise):
        """
        获取角色动作
        epsilon: 探索率
        """
        charge_prob = self.forward(task_window, uav_state)
        #print(charge_prob.shape)    #torch.Size([1, 2])
        probs = noise.GaussianNoise(charge_prob)
        # noise = torch.randn(self.action_dim - 2, device=device) * sigma
        # action = action + noise
        # print("action:", action)
        probs = torch.clamp(probs, 0, 1)
        # Epsilon-greedy探索
        '''if np.random.rand() < epsilon:
            # 随机选择
            role = torch.randint(0, 2, (charge_prob.shape[1],), device=charge_prob.device)
        else:'''
        # 根据概率采样
        threshold = torch.rand_like(probs)
        role = (probs > threshold).long().squeeze(-1)

        return role, charge_prob


class RoleCritic(nn.Module):
    """
    上层Critic网络
    评估角色选择的价值
    """
    def __init__(self, window_size, state_dim, num_agents, hidden_dim=128, lstm_layers=2):
        super(RoleCritic, self).__init__()

        # LSTM处理时序窗口
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # 处理全局状态和动作
        input_dim = hidden_dim + state_dim + num_agents  # LSTM特征 + 状态 + 角色动作

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, task_window, state, roles):
        """
        task_window: [batch, window_size]
        state: [batch, state_dim]
        roles: [batch, num_agents]
        """
        # LSTM处理窗口
        window_input = task_window.unsqueeze(-1)
        lstm_out, _ = self.lstm(window_input)
        lstm_feature = lstm_out[:, -1, :]

        # 拼接所有特征
        x = torch.cat([lstm_feature, state, roles], dim=-1)

        # 输出Q值
        q_value = self.fc(x)
        return q_value