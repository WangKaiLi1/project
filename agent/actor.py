import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RoleActor(nn.Module):
    """角色选择Actor网络
    -输出离散动作(0或1)"""
    def __init__(self, obs_dim, hidden_dim=256):
        super(RoleActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出2个动作的logits

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, obs):
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        #logits = self.fc3(x)
        logits = torch.sigmoid(self.fc3(x))  # 充电概率

        return logits

    def get_action(self, observation, noise):
        """获取动作，支持epsilon-greedy探索"""
        logits = self.forward(observation)
        #probs = F.softmax(logits, dim=-1)
        logits = noise.GaussianNoise(logits)
        # noise = torch.randn(self.action_dim - 2, device=device) * sigma
        # action = action + noise
        # print("action:", action)
        probs = torch.clamp(logits, 0, 1)
        threshold = torch.rand_like(probs)
        role = (probs < threshold).long().squeeze(-1)

        '''if np.random.rand() < epsilon:
            role = torch.randint(0, 2, (observation.shape[0],))
        else:
            role = torch.argmax(probs, dim=-1)'''

        return role, probs


class ServiceActor(nn.Module):
    """服务Actor网络
    - 输出连续动作(速度)和离散动作(方向)"""
    def __init__(self, obs_dim, action_dim, hidden_dim, args):
        super(ServiceActor, self).__init__()
        self.vh_min = args.vh_min
        self.vh_max = args.vh_max
        self.vt_min = args.vt_min
        self.vt_max = args.vt_max
        self.area_length = args.area_length
        self.area_width = args.area_width
        self.action_dim = action_dim
        self.num_directions = args.num_directions

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.direction = nn.Linear(hidden_dim, self.num_directions)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.action_dim - 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, obs):
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        #direction_logits = self.direction(x)
        action = torch.sigmoid(self.fc4(x))

        return action


    def get_action(self, obs, noise):
        device = obs.device
        action = self.forward(obs)
        #offload = action[:, 3:]  # [0, 1]
        # 添加噪声
        action = noise.GaussianNoise(action)
        #print("action:", action)
        action = torch.clamp(action, 0, 1)
        vh = action[:, 0] * (self.vh_max - self.vh_min) + self.vh_min
        #vt = action[:, 1] * (self.vt_max - self.vt_min) + self.vt_min
        #print(f"direction:{action[:, 1]}")
        direction_rad = action[:, 1] * 2 * np.pi
        # 将连续方向离散化到最近的离散方向
        direction_idx = torch.round(action[:, 1] * (2 * np.pi) / (2 * np.pi / self.num_directions))
        direction_idx = direction_idx.long()
        offload = action[:, 2:]

        # 改进的边界检查
        #direction_rad, direction_idx = self.check_and_adjust_direction(
        #    obs, vh, direction_rad, direction_idx
        #)

        #action[:, 1] = direction_rad / (2 * np.pi)

        return action, vh, direction_rad, direction_idx, offload

    def check_and_adjust_direction(self, obs, vh, direction_rad, direction_idx):
        """检查移动后是否会超出边界，并调整方向"""
        batch_size = obs.shape[0]
        x_norm = obs[:, 0]  # 当前位置
        y_norm = obs[:, 1]
        # 反归一化
        x = x_norm * self.area_length
        y = y_norm * self.area_width
        # 计算下一时刻位置（假设时间步长为1）
        dt = 1.0
        dx = vh * torch.cos(direction_rad) * dt
        dy = vh * torch.sin(direction_rad) * dt

        # 调整方向避免出界
        for i in range(batch_size):
            new_x = x[i] + dx[i]
            new_y = y[i] + dy[i]

            # 如果会超出边界，调整方向
            if new_x < 0 or new_x > self.area_length or new_y < 0 or new_y > self.area_width:
                # 计算朝向中心的向量
                center_x = self.area_length / 2
                center_y = self.area_width / 2
                vec_to_center = torch.tensor(
                    [center_x - x[i], center_y - y[i]],
                    device=obs.device
                )

                # 计算朝向中心的角度
                new_direction = torch.atan2(vec_to_center[1], vec_to_center[0])
                if new_direction < 0:
                    new_direction += 2 * np.pi

                direction_rad[i] = new_direction
                direction_idx[i] = torch.round(new_direction / (2 * np.pi / self.num_directions))
                direction_idx[i] = direction_idx[i].long()

        return direction_rad, direction_idx
