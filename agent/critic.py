import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """中心化Critic网络
    - 接收所有智能体的状态和动作"""
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim * num_agents

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, states, actions):
        """
        states: [batch_size, num_agents, state_dim]
        actions: [batch_size, num_agents, action_dim]
        """
        # 展平所有智能体的状态和动作
        batch_size = states.shape[0]
        x = torch.cat([states.reshape(batch_size, -1), actions.reshape(batch_size, -1)], dim=-1)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        q = self.q_value(x)

        return q