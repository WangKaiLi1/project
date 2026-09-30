import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from net.lnet import *
from net.attention import MultiHeadAttention

class RECL_MIX(nn.Module):
    def __init__(self, args):
        super(RECL_MIX, self).__init__()
        self.args = args
        self.N = args.num_uavs
        self.state_dim = args.upper_state_dim
        self.mix_input_dim = args.upper_state_dim + args.num_uavs * args.att_out_dim
        self.batch_size = args.upper_batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num

        self.state_embed_dim = args.state_embed_dim
        self.state_fc = nn.Linear(args.upper_state_dim, args.upper_state_dim)
        self.state_gru = nn.GRUCell(args.upper_state_dim, args.num_uavs * args.state_embed_dim)
        self.state_gru_hidden = None
        self.attention_net = MultiHeadAttention(
            n_heads=args.n_heads,
            att_dim=args.att_dim,
            att_out_dim=args.att_out_dim,
            soft_temperature=args.soft_temperature,
            dim_q=args.state_embed_dim,
            dim_k=args.role_embedding_dim,
            dim_v=args.role_embedding_dim
        )

        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)
        """
        if self.hyper_layers_num == 2:
            #print("hyper_layers_num=2")
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim)
            )
        elif self.hyper_layers_num == 1:
            #print("hyper_layers_num=1")
            self.hyper_w1 = nn.Linear(self.mix_input_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.mix_input_dim, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qmix_hidden_dim, 1)
        )

    def forward(self, q, s, att):
        """
        Args:
            q: (batch_size, max_ep_len, N) - 每个智能体的Q值
            s: (batch_size, max_ep_len, state_dim) - 全局状态
        Returns:
            q_total: (batch_size, max_ep_len, 1) - 混合后的全局Q值
        """
        #max_ep_len = q.shape[1]
        q = q.view(-1, 1, self.N)
        s = s.reshape(-1, self.state_dim)
        att = att.reshape(-1, att.shape[2])
        state = torch.cat([s, att], dim=-1)

        w1 = torch.abs(self.hyper_w1(state)).view(-1, self.N, self.qmix_hidden_dim)
        b1 = self.hyper_b1(state).view(-1, 1, self.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(-1, self.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(self.batch_size, -1, 1)

        return q_total

    def init_hidden(self, batch_size, device):
        self.state_gru_hidden = None
        #self.rnn_hidden = torch.zeros(batch_size, self.embedding_dim).to(device)


class RECL_NET(nn.Module):
    def __init__(self, args):
        super(RECL_NET, self).__init__()
        self.args = args
        self.N = args.num_uavs
        self.obs_dim = args.upper_obs_dim
        self.action_dim = args.upper_action_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        # 智能体嵌入网络
        self.agent_embedding_net = AgentEmbeddingNetwork(obs_dim=args.upper_obs_dim, action_dim=args.upper_action_dim, embedding_dim=args.agent_embedding_dim, window_size=args.history_window_size, device=args.device)
        # 智能体嵌入解码器
        self.agent_embedding_decoder = AgentEmbeddingDecoder(embedding_dim=args.agent_embedding_dim, output_dim=args.upper_obs_dim-args.history_window_size) #  + args.num_uavs
        # 角色嵌入网络
        self.role_embedding_net = RoleEmbeddingNetwork(agent_embedding_dim=args.agent_embedding_dim, role_embedding_dim=args.role_embedding_dim,
                                    use_layer_norm=args.use_layer_norm)
        # 目标角色嵌入网络
        self.role_embedding_target_net = RoleEmbeddingNetwork(agent_embedding_dim=args.agent_embedding_dim, role_embedding_dim=args.role_embedding_dim,
                                    use_layer_norm=args.use_layer_norm)
        self.role_embedding_target_net.load_state_dict(self.role_embedding_net.state_dict())
        # 对比学习参数
        self.W = nn.Parameter(torch.rand(args.role_embedding_dim, args.role_embedding_dim))

    def forward(self, obs, action, detach=False):
        agent_embedding = self.agent_embedding_net(obs, action, detach)
        role_embedding = self.role_embedding_net(agent_embedding)
        return role_embedding

    def encoder_decoder_forward(self, obs, action):
        agent_embedding = self.agent_embedding_forward(obs, action, detach=False)
        decoder_out = self.agent_embedding_decoder(agent_embedding)
        return decoder_out

    def agent_embedding_forward(self, obs, action, detach=False, return_trend=False):
        # 如果需要趋势，则从底层 net 获取两个返回值
        if return_trend:
            return self.agent_embedding_net(obs, action, detach, return_trend=True)
        return self.agent_embedding_net(obs, action, detach)

    def role_embedding_forward(self, agent_embedding, detach=False, ema=False):
        if ema:
            return self.role_embedding_target_net(agent_embedding, detach)
        return self.role_embedding_net(agent_embedding, detach)

    def role_decode(self, role_embedding, use_gumbel, temperature=0.5, deterministic=False):
        if use_gumbel:
            return self.role_decoder.sample_role_gumbel(role_embedding, temperature, hard=False)
        return self.role_decoder.sample_role_threshold(role_embedding, deterministic)

    # 训练时,断开梯度
    def batch_role_embed_forward(self, batch_o, batch_a, max_episode_len, detach=False):
        self.init_hidden(batch_o.shape[0] * self.N, self.args.device)
        agent_embeddings = []
        role_embeddings = []
        for t in range(max_episode_len + 1):  # t = 0,1,2...(max_episode_len-1), max_episode_len
            agent_embedding = self.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                                                           batch_a[:, t].reshape(-1, self.action_dim),
                                                           detach=detach)  # agent_embedding.shape=(batch_size*N, agent_embed_dim)
            agent_embedding = agent_embedding.reshape(batch_o.shape[0], self.N, -1)  # shape=(batch_size,N, agent_embed_dim)
            agent_embeddings.append(agent_embedding.reshape(batch_o.shape[0], self.N, -1))

        agent_embeddings = torch.stack(agent_embeddings, dim=1).reshape(-1, self.agent_embedding_dim)  # agent_embeddings.shape=(batch_size*(max_episode_len+1)*N, agent_embed_dim)
        role_embeddings = (self.role_embedding_forward(agent_embeddings, detach=False, ema=False).reshape(-1, max_episode_len + 1, self.N, self.role_embedding_dim))
        agent_embeddings = agent_embeddings.reshape(-1, max_episode_len + 1, self.N, self.agent_embedding_dim)
        return agent_embeddings, role_embeddings

    def init_hidden(self, batch_size, device):
        self.agent_embedding_net.init_hidden(batch_size, device)

class ACORM_MEC_Agent:
    def __init__(self, args):
        self.args = args
        self.N = args.num_uavs
        self.obs_dim = args.upper_obs_dim
        self.state_dim = args.upper_state_dim
        self.action_dim = args.upper_action_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.att_out_dim = args.att_out_dim
        self.num_clusters = 2  # 固定两个角色聚类
        self.num_terminals = args.num_terminals
        self.batch_size = args.upper_batch_size
        self.gamma = args.upper_gamma
        self.tau = args.upper_tau
        self.role_tau = args.role_tau
        self.device = args.device
        self.epsilon = args.epsilon_start
        self.add_agent_id = args.add_agent_id
        self.window_size = args.history_window_size
        self.build_networks(args)
        self.build_optimizers(args)
        self.train_step = 0
        self.QMIX_input_dim = args.upper_obs_dim  # 个体q网络的输入
        self.use_gumbel = self.args.use_gumbel
        self.gumbel_temperature = args.gumbel_temperature
        if self.args.add_last_action:
            #print("------add last action------")
            self.QMIX_input_dim += self.action_dim
        '''if self.add_agent_id:
            #print("------add agent id------")
            self.QMIX_input_dim += self.N'''
        self.QMIX_input_dim += self.role_embedding_dim

    def build_networks(self, args):
        self.RECL = RECL_NET(args).to(self.device)
        q_input_dim = args.upper_obs_dim + args.role_embedding_dim
        if args.add_last_action:
            q_input_dim += args.upper_action_dim

        self.eval_Q_net = QNetworkRNNA(input_dim=q_input_dim, hidden_dim=args.rnn_hidden_dim, output_dim=2).to(self.device)
        self.target_Q_net = QNetworkRNNA(input_dim=q_input_dim, hidden_dim=args.rnn_hidden_dim, output_dim=2).to(self.device)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        self.eval_mix_net = RECL_MIX(args).to(self.device)
        self.target_mix_net = RECL_MIX(args).to(self.device)
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        # 注意：这里的 att_dim 需要根据您 RECL_MIX 中 attention_net 输出的维度来设定，您的代码里是 N * att_out_dim
        att_total_dim = self.N * self.args.att_out_dim

        self.eval_q_star_net = WQMIX_Unrestricted_Mixer(self.N, self.state_dim, att_total_dim, args.qmix_hidden_dim).to(
            self.device)
        self.target_q_star_net = WQMIX_Unrestricted_Mixer(self.N, self.state_dim, att_total_dim,
                                                          args.qmix_hidden_dim).to(self.device)
        self.target_q_star_net.load_state_dict(self.eval_q_star_net.state_dict())

    def build_optimizers(self, args):
        self.role_parameters = (list(self.RECL.role_embedding_net.parameters())
                                + list(self.RECL.agent_embedding_net.parameters()) + [self.RECL.W])
        self.role_embedding_optimizer = torch.optim.Adam(self.role_parameters, lr=args.lr)
        self.RECL_parameters = list(self.RECL.parameters())
        self.RECL_optimizer = torch.optim.Adam(self.RECL_parameters, lr=args.recl_lr)
        self.encoder_decoder_params = (list(self.RECL.agent_embedding_net.parameters()) + list(self.RECL.agent_embedding_decoder.parameters()))
        self.encoder_decoder_optimizer = torch.optim.Adam(self.encoder_decoder_params, lr=args.agent_embedding_lr)

        #self.eval_parameters = (list(self.eval_mix_net.parameters()) + [p for q_net in self.eval_Q_nets for p in q_net.parameters()])
        #self.qmix_optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)
        #self.eval_parameters = (list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters()))
        #self.qmix_optimizer = torch.optim.Adam(self.eval_parameters, lr=args.qmix_lr)

        # 将无约束网络参数加入 Q 网络的优化器中！
        self.qmix_parameters = (list(self.eval_Q_net.parameters()) +
                                 list(self.eval_mix_net.parameters()) +
                                 list(self.eval_q_star_net.parameters()))  # 新增 q_star_net

        self.qmix_optimizer = torch.optim.Adam(self.qmix_parameters, lr=args.qmix_lr)

        # 学习率调度器
        if args.use_lr_decay:
            self.role_lr_scheduler = StepLR(self.role_embedding_optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
            self.qmix_lr_scheduler = StepLR(self.qmix_optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)

    def choose_role(self, obs_n, last_a_n):
        """
        Args:
            obs_n: (N, obs_dim) numpy array
            last_a_n: (N, action_dim) numpy array
            epsilon: 探索率
        Returns:
            roles: (N,) (0:充电, 1:服务)
            role_embeddings: (N, role_embedding_dim) 角色嵌入
        """
        with torch.no_grad():
            obs = torch.tensor(obs_n, dtype=torch.float32).to(self.device)
            last_action = torch.tensor(last_a_n, dtype=torch.float32).to(self.device)

            # 新增：推理前根据当前输入的数量（N）初始化隐藏状态
            # self.RECL.init_hidden(obs.shape[0], self.device)

            agent_embedding = self.RECL.agent_embedding_forward(obs, last_action)
            #agent_embedding = self.RECL.agent_embedding_forward(task_window, uav_obs, last_action)
            role_embedding = self.RECL.role_embedding_forward(agent_embedding)
            # epsilon-greedy角色选择
            if np.random.random() < self.epsilon:
                # 随机角色
                roles = np.random.randint(0, 2, size=self.N)
            else:
                # 确保 last_action 是二维的
                if last_action.dim() == 1:
                    last_action = last_action.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)

                if self.args.add_last_action:
                    inputs = torch.cat([obs, last_action], dim=-1)

                inputs = torch.cat([inputs, role_embedding], dim=-1)

                q_values = self.eval_Q_net(inputs)
                roles = torch.argmax(q_values, dim=-1).cpu().numpy()

            return roles, role_embedding


    def get_inputs(self, batch):
        inputs = copy.deepcopy(batch['obs_n'])
        if self.args.add_last_action:
            #last_a = batch['last_a_n']
            if batch['last_a_n'].dim() == 3:
                batch['last_a_n'] = batch['last_a_n'].unsqueeze(-1)
            inputs = np.concatenate((inputs, batch['last_a_n']), axis=-1)

        inputs = torch.tensor(inputs, dtype=torch.float32)

        inputs = inputs.to(self.device)
        batch_o = batch['obs_n'].to(self.device)
        batch_s = batch['s'].to(self.device)
        batch_r = batch['r'].to(self.device)
        batch_a = batch['a_n'].to(self.device)
        batch_last_a = batch['last_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_dw = batch['dw'].to(self.device)
        return inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_active, batch_dw

    def train(self, replay_buffer):
        self.train_step += 1
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_active, batch_dw = self.get_inputs(batch)
        batch = self.to_device(batch)
        recl_loss = torch.tensor(0.0).to(self.device)
        recl_loss = recl_loss.item()

        role_decoder_loss = torch.tensor(0.0).to(self.device)
        role_decoder_loss = role_decoder_loss.item()

        if self.train_step % self.args.train_recl_freq == 0:
            recl_loss = self.update_recl(batch_o, batch_last_a, batch_active, max_episode_len)
            self.soft_update(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)

        qmix_loss = self.update_wqmix(inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_active, batch_dw, max_episode_len)
        #print(f"更新Q网络")
        #role_decoder_loss = self.update_role_decoder(inputs, batch_o, batch_s, batch_last_a, batch_active, max_episode_len)

        if self.args.use_hard_update:
            if self.train_step % self.args.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
                self.target_q_star_net.load_state_dict(self.eval_q_star_net.state_dict()) # 新增
        else:
            self.soft_update(self.eval_Q_net, self.target_Q_net, self.tau)
            self.soft_update(self.eval_mix_net, self.target_mix_net, self.tau)
            self.soft_update(self.eval_q_star_net, self.target_q_star_net, self.tau) # 新增

        self.soft_update(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.tau)

        if self.args.use_lr_decay:
            self.qmix_lr_scheduler.step()
            self.role_lr_scheduler.step()

        # 温度衰减
        '''self.gumbel_temperature = max(
            self.args.gumbel_temp_min,
            self.gumbel_temperature * self.args.gumbel_temp_decay
        )'''

        return recl_loss, qmix_loss, role_decoder_loss

    def to_device(self, batch):
        """将batch数据移到GPU"""
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch

    def update_recl(self, batch_o, batch_last_a, batch_active, max_episode_len):
        """
        N = agent_num
        batch_o.shape = (batch_size, max_episode_len + 1, N,  obs_dim)
        batch_a.shape = (batch_size, max_episode_len, N,  action_dim)
        batch_active = (batch_size, max_episode_len, 1)
        """
        #print("====================更新角色嵌入=======================")
        batch_size = batch_o.shape[0]
        self.RECL.init_hidden(batch_size * self.N, self.device)
        total_loss = 0
        labels = np.zeros((batch_size, self.N))
        for t in range(max_episode_len): #  2 - max_episode_len
            # 获取智能体嵌入
            with torch.no_grad():
                agent_embedding = self.RECL.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                    batch_last_a[:, t].reshape(-1, self.action_dim),
                    detach=True) # agent_embedding.shape=(batch_size*N, agent_embed_dim)
                #print(f"智能体嵌入的形状{agent_embedding.shape}")
            # 计算角色嵌入
            role_embedding_query = (self.RECL.role_embedding_forward(agent_embedding, detach=False, ema=False)
                                    .reshape(-1, self.N, self.role_embedding_dim))
            role_embedding_key = (self.RECL.role_embedding_forward(agent_embedding, detach=True, ema=True)
                                  .reshape(-1, self.N, self.role_embedding_dim))

            # 计算logits
            W_expanded = self.RECL.W.unsqueeze(0).expand((role_embedding_query.shape[0], self.role_embedding_dim, self.role_embedding_dim))
            logits = torch.bmm(role_embedding_query, W_expanded)
            logits = torch.bmm(logits, role_embedding_key.transpose(1, 2))
            logits = logits - torch.max(logits, dim=-1)[0][:,:,None]
            exp_logits = torch.exp(logits)

            # K-means聚类
            agent_embedding_np = agent_embedding.reshape(batch_size, self.N, -1).cpu().numpy()

            for idx in range(batch_size):
                if batch_active[idx, t] > 0.5:
                    if t % self.args.multi_steps == 0:
                        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=5)  # 10
                        cluster_labels = kmeans.fit(agent_embedding_np[idx]).labels_
                        labels[idx] = copy.deepcopy(cluster_labels)
                    else:
                        cluster_labels = copy.deepcopy(labels[idx])

                    # InfoNCE损失
                    for cluster_id in range(self.num_clusters):
                        label_pos = [idx for idx, value in enumerate(cluster_labels) if value==cluster_id]
                        for anchor in label_pos:
                            pos_exp_sum = exp_logits[idx, anchor, label_pos].sum()
                            all_exp_sum = exp_logits[idx, anchor].sum()
                            total_loss += -torch.log((pos_exp_sum + 1e-8)/ (all_exp_sum + 1e-8))

        total_loss = total_loss / (batch_size * max_episode_len * self.N + 1e-8)

        if batch_active[idx, t] > 0.5:
            self.RECL_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.RECL_parameters, self.args.grad_clip)
            self.RECL_optimizer.step()
        #print(f"RECL损失：{total_loss}")
        return total_loss.item()

    def update_qmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_active, batch_dw, max_episode_len):
        batch_size = batch_o.shape[0]
        # 初始化隐藏状态
        #self.RECL.init_hidden(batch_size * self.N, self.device)
        self.eval_Q_net.init_hidden(batch_size * self.N, self.device)
        self.target_Q_net.init_hidden(batch_size * self.N, self.device)

        _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False)  # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
        inputs = torch.cat([inputs, role_embeddings], dim=-1)
        q_evals, q_targets = [], []

        self.eval_mix_net.state_gru_hidden = None
        # self.target_mix_net.state_gru_hidden = None
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s.reshape(-1, self.state_dim))).reshape(-1, max_episode_len + 1, self.state_dim)  # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.QMIX_input_dim))  # q_eval.shape=(batch_size*N,action_dim)
            q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.QMIX_input_dim))
            q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
            q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
                fc_batch_s[:, t].reshape(-1, self.state_dim),
                self.eval_mix_net.state_gru_hidden)  # shape=(batch, N*state_embed_dim)
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)


        self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, max_episode_len].reshape(-1, self.state_dim), self.eval_mix_net.state_gru_hidden)
        state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim) # shape=(batch*max_len+1, N,state_embed_dim)
        q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_targets = torch.stack(q_targets, dim=1)
        #q_targets = torch.stack(target_q_values, dim=1)

        with torch.no_grad():
            q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.QMIX_input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1)  # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
            a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze( -1)  # q_targets.shape=(batch_size, max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch_a.unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)

        role_embeddings = role_embeddings.reshape(-1, self.N, self.role_embedding_dim) # shape=((batch_size * max_episode_len+1), N, role_embed_dim)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)  # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        with torch.no_grad():
            att_target = self.target_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)  # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)

        #print(f"q_evals.shape={q_evals.shape}")
        #print(f"q_targets.shape={q_targets.shape}")
        #print(f"fc_batch_s.shape={fc_batch_s.shape}")
        #print(f"att_eval.shape={att_eval.shape}")
        #print(f"att_target.shape={att_target.shape}")
        #q_evals.shape=torch.Size([2, 30, 2, 1])
        #q_targets.shape=torch.Size([2, 30, 2, 1])
        #fc_batch_s.shape=torch.Size([2, 31, 15])
        #att_eval.shape=torch.Size([2, 31, 128])
        #att_target.shape=torch.Size([2, 31, 128])
        eval_q_total = self.eval_mix_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])
        target_q_total = self.target_mix_net(q_targets, fc_batch_s[:, 1:], att_target[:, 1:])
        targets = batch_r + self.gamma * (1 - batch_dw) * target_q_total
        td_error = (eval_q_total - targets.detach()) * batch_active
        loss = (td_error ** 2).sum() / (batch_active.sum() + 1e-8)

        self.qmix_optimizer.zero_grad()
        self.role_embedding_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.role_parameters, self.args.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_clip)
        self.qmix_optimizer.step()
        self.role_embedding_optimizer.step()

        #print(f"qmix更新损失:{loss}")
        return loss.item()

    def update_wqmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_active, batch_dw, max_episode_len):
        batch_size = batch_o.shape[0]
        # 初始化隐藏状态
        self.eval_Q_net.init_hidden(batch_size * self.N, self.device)
        self.target_Q_net.init_hidden(batch_size * self.N, self.device)

        _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len,
                                                                detach=False)  # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
        inputs = torch.cat([inputs, role_embeddings], dim=-1)
        q_evals, q_targets = [], []

        self.eval_mix_net.state_gru_hidden = None
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s.reshape(-1, self.state_dim))).reshape(-1,
                                                                                                     max_episode_len + 1,
                                                                                                     self.state_dim)  # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        for t in range(max_episode_len):
            q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.QMIX_input_dim))
            q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.QMIX_input_dim))
            q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))
            q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
                fc_batch_s[:, t].reshape(-1, self.state_dim),
                self.eval_mix_net.state_gru_hidden)
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
            fc_batch_s[:, max_episode_len].reshape(-1, self.state_dim), self.eval_mix_net.state_gru_hidden)
        state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        with torch.no_grad():
            q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.QMIX_input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1)
            a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)
        q_evals = torch.gather(q_evals, dim=-1, index=batch_a.unsqueeze(-1)).squeeze(-1)

        role_embeddings = role_embeddings.reshape(-1, self.N, self.role_embedding_dim)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)
        with torch.no_grad():
            att_target = self.target_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len + 1, self.N * self.att_out_dim)

        # ==================== WQMIX 核心更新部分 ====================
        # 1. 计算受限网络 Q_tot 与 无约束网络 Q_star 的当前值
        eval_q_total = self.eval_mix_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])
        eval_q_star = self.eval_q_star_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])

        with torch.no_grad():
            # 2. WQMIX 特性: 目标值(Target)由无约束的 target_q_star 生成，以提供准确的 Q 估计
            target_q_star = self.target_q_star_net(q_targets, fc_batch_s[:, 1:], att_target[:, 1:])
            targets = batch_r + self.gamma * (1 - batch_dw) * target_q_star

        # 3. 计算无约束网络 Q_star 的 Loss (标准的 MSE)
        td_error_star = (eval_q_star - targets) * batch_active
        loss_star = (td_error_star ** 2).sum() / (batch_active.sum() + 1e-8)

        # 4. 计算受限网络 Q_tot 的加权 Loss (WQMIX 精髓)
        td_error_tot = (eval_q_total - targets.detach()) * batch_active
        wqmix_alpha = getattr(self.args, 'wqmix_alpha', 0.8)  # 高估时的惩罚系数

        # 如果 td_error_tot < 0 (受限网络低估了真实值)，权重给 1.0
        # 如果 td_error_tot >= 0 (受限网络高估了真实值，通常因为受限网络的绝对值结构)，权重给 wqmix_alpha (打折惩罚)
        weight = torch.where(td_error_tot < 0, torch.ones_like(td_error_tot), wqmix_alpha * torch.ones_like(td_error_tot))
        loss_tot = (weight * (td_error_tot ** 2)).sum() / (batch_active.sum() + 1e-8)

        # 5. 总 Loss
        loss = loss_tot + loss_star
        # ============================================================

        self.qmix_optimizer.zero_grad()
        self.role_embedding_optimizer.zero_grad()
        loss.backward()

        # 确保 qmix_optimizer 包含 eval_q_star_net 的参数
        torch.nn.utils.clip_grad_norm_(self.role_parameters, self.args.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.qmix_parameters, self.args.grad_clip)

        self.qmix_optimizer.step()
        self.role_embedding_optimizer.step()

        return loss.item()

    def update_role_decoder(self, inputs, batch_o, batch_s, batch_last_a, batch_active, max_episode_len):
        """角色解码器更新"""
        batch_size = batch_o.shape[0]

        # 冻结 Q 网络参数，防止在更新策略时影响价值评估
        for p in self.eval_parameters:
            p.requires_grad = False

        self.RECL.init_hidden(batch_size * self.N, self.device)
        self.eval_Q_net.init_hidden(batch_size * self.N, self.device)
        self.eval_mix_net.init_hidden(batch_size, self.device)

        # 1. 重新生成带梯度的角色决策
        # 这里的 agent_embeddings 和 role_embeddings 必须带梯度
        agent_embeddings, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False)

        # 使用重参数化采样 (带梯度)
        role_z = role_embeddings[:, :max_episode_len].reshape(-1, self.role_embedding_dim)
        curr_roles, probs = self.RECL.role_decode(role_z, use_gumbel=self.use_gumbel)
        curr_roles = curr_roles.reshape(batch_size, max_episode_len, self.N, 1)

        # 2. 将新选出的角色喂给 Q 网络进行评价
        actor_q_input = torch.cat([inputs[:, :max_episode_len], role_embeddings[:, :max_episode_len], curr_roles],dim=-1)

        actor_qs = []
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s[:, :max_episode_len].reshape(-1, self.state_dim))).reshape(-1, max_episode_len, self.state_dim)  # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        for t in range(max_episode_len):
            q_t = self.eval_Q_net(actor_q_input[:, t].reshape(-1, actor_q_input.shape[-1]))
            actor_qs.append(q_t.reshape(batch_size, self.N))
            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(
                fc_batch_s[:, t].reshape(-1, self.state_dim),
                self.eval_mix_net.state_gru_hidden)  # shape=(batch, N*state_embed_dim)
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        #self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, max_episode_len].reshape(-1, self.state_dim), self.eval_mix_net.state_gru_hidden)
        #state_gru_outs.append(self.eval_mix_net.state_gru_hidden)
        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim)  # shape=(batch*max_len, N,state_embed_dim)
        role_embeddings = role_embeddings[:, :max_episode_len].reshape(-1, self.N, self.role_embedding_dim)
        actor_q_vals = torch.stack(actor_qs, dim=1)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len, self.N * self.att_out_dim)  # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)

        # 3. 通过混合网络计算全局总价值
        actor_q_total = self.eval_mix_net(actor_q_vals, batch_s[:, :max_episode_len], att_eval[:, :max_episode_len])

        # 4. 损失函数：最大化 Q 值（即最小化负 Q）
        mask = batch_active.reshape_as(actor_q_total)
        loss_actor = - (actor_q_total * mask).sum() / (mask.sum() + 1e-8)

        # 优化 Actor
        self.role_decoder_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.RECL.role_decoder.parameters(), self.args.grad_clip)
        self.role_decoder_optimizer.step()

        # 恢复 Q 网络参数的梯度状态
        for p in self.eval_parameters:
            p.requires_grad = True

        return loss_actor.item()


    '''def pretrain_agent_embedding(self, replay_buffer):
        #print(f"==========================预训练智能体嵌入=========================")
        """预训练智能体嵌入"""
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch = self.to_device(batch)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_size = batch_o.shape[0]

        self.RECL.init_hidden(batch_size * self.N, self.device)
        agent_embeddings = []
        for t in range(max_episode_len):
            agent_embedding = self.RECL.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim), batch_last_a[:,t].reshape(-1, self.action_dim), detach=False)
            agent_embeddings.append(agent_embedding.reshape(-1, self.N, self.agent_embedding_dim)) # (batch_size, N, agent_embedding_dim)

        agent_embeddings = torch.stack(agent_embeddings, dim=1)
        decoder_output = self.RECL.agent_embedding_decoder(
            agent_embeddings.reshape(-1, self.agent_embedding_dim)
        ).reshape(-1, max_episode_len, self.N, self.obs_dim)  # + self.N
        # 目标
        target_obs = batch_o[:, 1:max_episode_len + 1]
        #agent_id_onehot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_episode_len, 1, 1).to(self.device)
        #target = torch.cat([target_obs, agent_id_onehot], dim=-1)
        mask = batch_active.unsqueeze(-1).repeat(1, 1, self.N, self.obs_dim)  # +self.N
        loss = (((decoder_output - target_obs) * mask) ** 2).sum() / mask.sum()  # target

        self.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_decoder_optimizer.step()

        return loss.item()
        #print(f"==========================预训练智能体嵌入=========================")'''

    '''# 只预测一步
    def pretrain_agent_embedding(self, replay_buffer):
        # print(f"==========================预训练智能体嵌入=========================")
        """预训练智能体嵌入：联合重构空间特征与预测未来任务趋势"""
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch = self.to_device(batch)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_size = batch_o.shape[0]

        # 1. 确定维度的切分点
        window_size = self.args.history_window_size
        uav_obs_dim = self.obs_dim - window_size

        self.RECL.init_hidden(batch_size * self.N, self.device)
        agent_embeddings = []
        predicted_trends = []

        for t in range(max_episode_len):
            # 获取融合嵌入 e_t 以及 预测的趋势
            agent_embedding, predicted_trend = self.RECL.agent_embedding_forward(
                batch_o[:, t].reshape(-1, self.obs_dim),
                batch_last_a[:, t].reshape(-1, self.action_dim),
                detach=False,
                return_trend=True
            )
            agent_embeddings.append(agent_embedding.reshape(-1, self.N, self.agent_embedding_dim))
            predicted_trends.append(predicted_trend.reshape(-1, self.N, 1))

        agent_embeddings = torch.stack(agent_embeddings, dim=1)  # [batch, max_ep_len, N, embedding_dim]
        predicted_trends = torch.stack(predicted_trends, dim=1)  # [batch, max_ep_len, N, 1]

        # 2. 空间解码器输出：从 e_t 重构 uav_obs (不需要重建历史窗口)
        decoder_output = self.RECL.agent_embedding_decoder(
            agent_embeddings.reshape(-1, self.agent_embedding_dim)
        ).reshape(-1, max_episode_len, self.N, uav_obs_dim)

        # 3. 提取真实的 Target (用 t+1 时刻的真实数据作为目标)
        target_obs = batch_o[:, 1:max_episode_len + 1]

        # 目标一：目标 UAV 物理状态
        target_uav_obs = target_obs[:, :, :, window_size:]

        # 目标二：目标未来任务数 (t+1 时刻窗口的最后一个值，恰好是最新生成的任务数)
        target_trend = target_obs[:, :, :, window_size - 1].unsqueeze(-1)

        # 4. 分别计算 Loss
        # 空间重构 Loss
        mask_uav = batch_active.unsqueeze(-1).repeat(1, 1, self.N, uav_obs_dim)
        loss_recon = (((decoder_output - target_uav_obs) * mask_uav) ** 2).sum() / (mask_uav.sum() + 1e-8)

        # 时间预测 Loss
        mask_trend = batch_active.unsqueeze(-1).repeat(1, 1, self.N, 1)
        loss_trend = (((predicted_trends - target_trend) * mask_trend) ** 2).sum() / (mask_trend.sum() + 1e-8)

        # 5. 组合总 Loss (★ 实战提示：注意量级)
        # UAV 坐标和电量通常是归一化的 (0~1)，其 MSE 通常很小 (比如 0.01)
        # 任务数量可能是 10, 20 甚至更大，其 MSE 可能会非常大 (比如 100)
        # 为了防止预测任务 Loss 彻底吞噬重构 Loss，建议加一个权重系数 alpha
        alpha = 0.01  # 你可以根据打印出的 loss_recon 和 loss_trend 调整这个比例
        total_loss = loss_recon + alpha * loss_trend

        # 6. 反向传播
        self.encoder_decoder_optimizer.zero_grad()
        total_loss.backward()
        # 建议加梯度截断，防止初始阶段任务预测的误差导致梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.encoder_decoder_params, self.args.grad_clip)
        self.encoder_decoder_optimizer.step()

        # (可选) 打印出来看看两者的量级，方便你调参
        # print(f"Loss Recon: {loss_recon.item():.4f}, Loss Trend: {loss_trend.item():.4f}")

        return total_loss.item()'''

    def pretrain_agent_embedding(self, replay_buffer):
        """预训练智能体嵌入：联合重构空间特征与预测未来【多步】任务趋势"""
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch = self.to_device(batch)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_size = batch_o.shape[0]

        window_size = self.args.history_window_size
        uav_obs_dim = self.obs_dim - window_size
        future_steps = getattr(self.args, 'future_steps', 3)  # 默认预测未来3步

        self.RECL.init_hidden(batch_size * self.N, self.device)
        agent_embeddings = []
        predicted_trends = []

        for t in range(max_episode_len):
            agent_embedding, predicted_trend = self.RECL.agent_embedding_forward(
                batch_o[:, t].reshape(-1, self.obs_dim),
                batch_last_a[:, t].reshape(-1, self.action_dim),
                detach=False,
                return_trend=True
            )
            agent_embeddings.append(agent_embedding.reshape(-1, self.N, self.agent_embedding_dim))
            # 此时 predicted_trend 维度是 [batch*N, future_steps]
            predicted_trends.append(predicted_trend.reshape(-1, self.N, future_steps))

        agent_embeddings = torch.stack(agent_embeddings, dim=1)  # [batch, max_ep_len, N, embed_dim]
        predicted_trends = torch.stack(predicted_trends, dim=1)  # [batch, max_ep_len, N, future_steps]

        # 1. 空间解码器输出重构 uav_obs
        decoder_output = self.RECL.agent_embedding_decoder(
            agent_embeddings.reshape(-1, self.agent_embedding_dim)
        ).reshape(-1, max_episode_len, self.N, uav_obs_dim)

        target_obs = batch_o[:, 1:max_episode_len + 1]
        target_uav_obs = target_obs[:, :, :, window_size:]

        # ==================== 2. 多步目标提取与掩码核心逻辑 ====================
        # 提取整条轨迹所有的真实任务数量 (含最后溢出的1步)，形状: [batch, max_ep_len + 1, N]
        all_true_tasks = batch_o[:, :, :, window_size - 1]

        # 为了防止索引越界，在最后补充 future_steps 个全零垫片
        padding = torch.zeros(batch_size, future_steps, self.N).to(self.device)
        padded_tasks = torch.cat([all_true_tasks, padding], dim=1)  # [batch, max_ep_len + 1 + K, N]

        target_trend_list = []
        valid_mask_list = []

        for t in range(max_episode_len):
            # 获取 t 时刻向后的 K 步真实数据 (t+1 到 t+1+K)
            target_trend_list.append(padded_tasks[:, t + 1: t + 1 + future_steps, :])

            # 生成这一步的时间掩码 (如果越界了，就用 0 遮住它)
            step_mask = torch.ones(future_steps).to(self.device)
            for k in range(future_steps):
                if t + 1 + k > max_episode_len:  # 如果超出了真实的轨迹长度
                    step_mask[k] = 0.0
            valid_mask_list.append(step_mask)

        # target_trend: [batch, max_ep_len, future_steps, N] -> 换位 -> [batch, max_ep_len, N, future_steps]
        target_trend = torch.stack(target_trend_list, dim=1).transpose(2, 3)

        # valid_mask: [max_ep_len, future_steps] -> 广播 -> [batch, max_ep_len, N, future_steps]
        valid_mask = torch.stack(valid_mask_list, dim=0)
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.N, -1)

        # 结合原本的 active mask（智能体存活状态）
        active_mask = batch_active.unsqueeze(-1).expand(-1, -1, self.N, future_steps)
        final_trend_mask = active_mask * valid_mask
        # ======================================================================

        # 3. 计算 Loss
        # 空间重构 Loss
        mask_uav = batch_active.unsqueeze(-1).repeat(1, 1, self.N, uav_obs_dim)
        loss_recon = (((decoder_output - target_uav_obs) * mask_uav) ** 2).sum() / (mask_uav.sum() + 1e-8)

        # 时间多步预测 Loss (用上了 final_trend_mask 进行遮盖)
        loss_trend = (((predicted_trends - target_trend) * final_trend_mask) ** 2).sum() / (
                    final_trend_mask.sum() + 1e-8)

        # 组合总 Loss
        alpha = 0.01  # 防止任务数的MSE过大，按需调整
        total_loss = loss_recon + alpha * loss_trend

        # 反向传播
        self.encoder_decoder_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder_decoder_params, self.args.grad_clip)
        self.encoder_decoder_optimizer.step()

        return total_loss.item()

    def pretrain_recl(self, replay_buffer):
        """预训练对比学习"""
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch = self.to_device(batch)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)

        recl_loss = self.update_recl(batch_o, batch_last_a, batch_active, max_episode_len)
        #print(f"==========================预训练对比学习=========================")
        self.soft_update(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
        return recl_loss


    def soft_update(self, source_net, target_net, tau):
        """软更新"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def init_hidden_states(self):
        """初始化隐藏状态"""
        self.RECL.init_hidden(self.N, self.device)
        self.eval_Q_net.init_hidden(self.N, self.device)

    def save_models(self, path):
        """保存模型"""
        torch.save({
            'role_net': self.RECL.role_embedding_net,
            'agent_embedding_net': self.RECL.agent_embedding_net,
            'q_net': self.eval_Q_net,
            'mix_net': self.eval_mix_net,
        }, path)
        print(f"Models saved to {path}")

    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.RECL.load_state_dict(checkpoint['role_net'])
        self.eval_Q_net.load_state_dict(checkpoint['eval_Q_net'])
        self.eval_mix_net.load_state_dict(checkpoint['eval_mix_net'])

        self.target_Q_net.load_state_dict(self.eval_Q_net)
        self.target_mix_net.load_state_dict(self.eval_mix_net)
        self.RECL.role_embedding_target_net.load_state_dict(self.RECL.role_embedding_net)
        print(f"Models loaded from {path}")