import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AgentEmbeddingNetwork(nn.Module):
    """
    输入: o_t + a_{t-1} (连续动作)
    输出: e_t
    """
    def __init__(self, obs_dim, action_dim, embedding_dim, window_size, device, hidden_dim=128, lstm_layers=2):
        super(AgentEmbeddingNetwork, self).__init__()
        self.window_size = window_size
        self.device = device
        self.input_dim = obs_dim - window_size + action_dim + hidden_dim
        self.embedding_dim = embedding_dim

        # LSTM 处理时序窗口 (任务到达率历史)
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间槽的任务数
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # 线性层处理 UAV 状态
        self.uav_fc = nn.Sequential(
            nn.Linear(obs_dim - window_size + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # self.fc1 = nn.Linear(self.input_dim, embedding_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, embedding_dim)
        self.gru = nn.GRUCell(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

        self.rnn_hidden = None

    def forward(self, obs, last_action, detach=False):
        """
        Args:
            obs: (batch_size, obs_dim)
            last_action: (batch_size, action_dim) 连续动作
            detach: 是否截断梯度
        """
        # 确保 last_action 是二维的
        if last_action.dim() == 1:
            last_action = last_action.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)

        task_window = obs[:, :self.window_size]
        uav_obs = obs[:, self.window_size:]
        #task_window = torch.FloatTensor(task_window).to(self.device)
        #uav_obs = torch.FloatTensor(uav_obs).to(self.device)

        window_input = task_window.unsqueeze(-1)  # [batch, window_size, 1]
        lstm_out, _ = self.lstm(window_input)
        lstm_feature = lstm_out[:, -1, :]  # 取最后一个时间步 [batch, hidden_dim]

        uav_input = torch.cat([uav_obs, last_action], dim=-1)
        uav_feature = self.uav_fc(uav_input)  # [batch, hidden_dim]

        #inputs = torch.cat([lstm_feature, uav_obs, last_action], dim=-1)
        #inputs = torch.cat([obs, last_action], dim=-1)
        inputs = torch.cat([lstm_feature, uav_feature], dim=-1)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.gru(x, self.rnn_hidden)
        output = self.fc2(self.rnn_hidden)
        if detach:
            return output.detach()
        return output

    def init_hidden(self, batch_size, device):
        self.rnn_hidden = torch.zeros(batch_size, self.embedding_dim).to(device)
        #self.rnn_hidden = None


class AgentEmbeddingDecoder(nn.Module):
    """智能体嵌入解码器 (预训练用)"""
    def __init__(self, embedding_dim, output_dim):
        super(AgentEmbeddingDecoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, output_dim)

    def forward(self, agent_embedding):
        x = F.relu(self.fc1(agent_embedding))
        return self.fc2(x)


class RoleEmbeddingNetwork(nn.Module):
    """角色嵌入网络"""
    def __init__(self, agent_embedding_dim, role_embedding_dim, use_layer_norm=False):
        super(RoleEmbeddingNetwork, self).__init__()
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.net = nn.Sequential(
                nn.Linear(agent_embedding_dim, role_embedding_dim),
                nn.LayerNorm(role_embedding_dim)
            )
        else:
            self.net = nn.Linear(agent_embedding_dim, role_embedding_dim)

    def forward(self, agent_embedding, detach=False):
        output = self.net(agent_embedding)
        #output = torch.sigmoid(self.net(agent_embedding))
        if detach:
            return output.detach()
        output = torch.sigmoid(output)
        return output


class RoleDecoder(nn.Module):
    """
    角色解码器
    输入: z_t (角色嵌入)
    输出: 角色选择概率分布参数 (μ, σ)
    """
    def __init__(self, role_embedding_dim, hidden_dim=64, threshold=0.5):
        super(RoleDecoder, self).__init__()
        self.fc1 = nn.Linear(role_embedding_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_log_sigma = nn.Linear(hidden_dim, 1)
        self.threshold = threshold  # 判定阈值

    def forward(self, role_embedding):
        x = F.relu(self.fc1(role_embedding))
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        sigma = torch.exp(log_sigma).clamp(min=1e-6, max=1.0)
        return mu, sigma

    def sample_role_gumbel(self, role_embedding, temperature, hard=False):
        """Gumbel-Softmax采样 (可微分)"""
        mu, sigma = self.forward(role_embedding)
        logits = torch.cat([-mu, mu], dim=-1)  # [充电, 服务]
        role_prob = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        role = role_prob.argmax(dim=-1)
        return role, role_prob

    def sample_role_threshold(self, role_embedding, deterministic=False, temperature=1.0):
        """阈值比较采样"""
        mu, sigma = self.forward(role_embedding)
        if deterministic:
            prob = torch.sigmoid(mu)
            role = (prob > 0.5).long().squeeze(-1)
        else:
            # 标准重参数化公式: z = mu + std * eps
            # 其中 eps 是从标准正态分布采样
            eps = torch.randn_like(sigma)  # mu
            sample = mu + sigma * eps
            # 软阈值比较 (用Sigmoid模拟 v > threshold)
            # 我们计算：prob = sigmoid( (v - threshold) / temp )
            # 当 v >> threshold 时，prob -> 1 (服务/充电); 当 v << threshold 时，prob -> 0
            # 这里的 temperature (温度系数) 很关键：
            # temp 越小，曲线越陡峭，越像真实的阈值比较 ( > 0.5 )
            #prob = torch.sigmoid(sample)
            #threshold = torch.rand_like(prob)
            prob = torch.sigmoid((sample - self.threshold) / temperature)
            # 【Straight-Through Estimator (STE)】（工业级标准技巧）
            # 前向传播：得到硬的 0 或 1
            # 反向传播：保留 role_prob 的梯度
            #role = (prob > threshold).long().squeeze(-1)
            role_hard = (prob > 0.5).float()
            role = role_hard - prob.detach() + prob

        return role, prob

    def get_role_prob(self, role_embedding):
        """获取角色概率 (用于策略梯度)"""
        mu, sigma = self.forward(role_embedding)
        prob_service = torch.sigmoid(mu)  # 选择服务的概率
        prob_charge = 1 - prob_service  # 选择充电的概率
        return torch.cat([prob_charge, prob_service], dim=-1)


class QNetworkRNN(nn.Module):
    """个体Q网络 (RNN)"""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(QNetworkRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.rnn_hidden = None

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        return self.fc2(self.rnn_hidden)

    def init_hidden(self, batch_size, device):
        self.rnn_hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        #self.rnn_hidden = None

class QNetworkRNNA(nn.Module):
    """个体Q网络 (RNN)"""
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(QNetworkRNNA, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.rnn_hidden = None

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        return self.fc2(self.rnn_hidden)

    def init_hidden(self, batch_size, device):
        self.rnn_hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        #self.rnn_hidden = None

class WQMIX_Unrestricted_Mixer(nn.Module):
    """
    WQMIX 中的无约束全局评估网络 (Q* 或 q_hat_star)
    用于计算准确的 TD Target，不包含绝对值权重，不受单调性约束
    """
    def __init__(self, N, state_dim, att_dim, qmix_hidden_dim=64):
        super(WQMIX_Unrestricted_Mixer, self).__init__()
        self.N = N
        self.state_dim = state_dim
        # 你的 att_eval 的最后一维是 N * att_out_dim
        self.att_dim = att_dim

        # 输入：每个 agent 的局部 Q (N) + 全局状态 s + 注意力特征 att
        self.input_dim = N + state_dim + att_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, 1)
        )

    def forward(self, q_values, state, att):
        """
        q_values: (batch_size, seq_len, N)
        state: (batch_size, seq_len, state_dim)
        att: (batch_size, seq_len, N * att_out_dim)
        """
        batch_size = q_values.shape[0]
        seq_len = q_values.shape[1]

        q_values = q_values.reshape(-1, self.N)
        state = state.reshape(-1, self.state_dim)
        att = att.reshape(-1, self.att_dim)

        inputs = torch.cat([q_values, state, att], dim=-1)
        q_star = self.net(inputs)

        return q_star.view(batch_size, seq_len, 1)