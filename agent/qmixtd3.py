import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
from torch.optim.lr_scheduler import StepLR
from net.net import *
from net.attention import MultiHeadAttention

import torch.optim as optim
from common.nosie import Noise
from agent.actor import ServiceActor
from agent.critic import Critic
from agent.replay_buffer import ReplayBuffer
from agent.qmix_learner import VDN_QMIX
from agent.acorm_replay_buffer import AcormReplayBuffer
from utils.normalizer import ObservationNormalizer, ActionNormalizer


class MADDPG:
    """
        分层算法
        上层：角色选择（LSTM + DQN）-- 已修改
        下层：具体动作（TD3）
    """
    def __init__(self, args, device):
        self.args = args
        self.num_agents = args.num_uavs
        self.device = device
        self.noise = Noise(args)

        # ========== 上层：角色选择 (DQN) ==========
        self.window_size = args.history_window_size
        self.upper_state_dim = args.upper_state_dim

        self.upper_agent = VDN_QMIX(args)
        # 上层经验池
        self.upper_buffer = AcormReplayBuffer(args)

        # 上层探索参数 (Epsilon-Greedy)
        #self.upper_agent.epsilon = args.epsilon_start
        #self.upper_agent.epsilon_min = args.epsilon_min
        #self.upper_state_dim.epsilon_decay = args.epsilon_decay

        # ========== 下层 (TD3 - 保持不变) ==========
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.gamma = args.gamma

        # 添加归一化器
        self.obs_normalizer = ObservationNormalizer(args.obs_dim)
        self.state_normalizer = ObservationNormalizer(args.state_dim)
        self.action_normalizer = ActionNormalizer(args.action_dim, args.num_uavs)
        self.use_normalization = False

        self.service_actor = ServiceActor(self.obs_dim, self.action_dim, args.hidden_dim, args).to(device)
        self.service_actor_target = ServiceActor(self.obs_dim, self.action_dim, args.hidden_dim, args).to(device)
        self.service_optimizer = optim.Adam(self.service_actor.parameters(), lr=args.lr_actor)
        self.service_actor_target.load_state_dict(self.service_actor.state_dict())

        self.critic1 = Critic(self.state_dim, self.action_dim, self.num_agents, args.hidden_dim).to(self.device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim, self.num_agents, args.hidden_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim, self.num_agents, args.hidden_dim).to(self.device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim, self.num_agents, args.hidden_dim).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.lr_critic)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.lower_buffer = ReplayBuffer(args)
        self.use_per = args.use_per

        # 计数器
        self.upper_updates = 0
        self.lower_updates = 0

        # 大时间步轨迹追踪
        self.current_large_step_data = {
            'big_step': None,
            'initial_upper_state': None,
            'initial_upper_obs': None,
            'last_a': None,
            'a': None,
            'small_step_rewards': []
        }

    def select_roles(self, obs, last_a, explore=True):
        """
        上层：选择角色 (ACORM)
        """
        # 调用ACORM智能体
        with torch.no_grad():
            roles = self.upper_agent.choose_role(obs, last_a)

        return roles

    def select_actions(self, obs, role, explore=True):
        """
        下层动作选择保持不变
        """
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.zeros(self.num_agents, self.action_dim).to(self.device)
        raw_actions = torch.zeros(self.num_agents, self.action_dim).to(self.device)

        with torch.no_grad():
            for i in range(self.num_agents):
                actions[i, 0] = role[i]
                raw_actions[i, 0] = role[i]
                obs_i = obs[i].reshape(1, -1)
                if role[i] == 1:  # 服务角色
                    service_action, service_vh, service_dir_rad, service_dir_idx, service_offload = self.service_actor.get_action(obs_i, self.noise)
                    #print(service_action.shape)
                    service_action = service_action.squeeze()
                    raw_actions[i, 1] = service_action[0]
                    raw_actions[i, 2] = service_action[1]
                    raw_actions[i, 3:] = service_action[2:]
                    actions[i, 1] = service_vh.squeeze()
                    actions[i, 2] = service_dir_idx.float()
                    actions[i, 3:] = service_offload
                # 充电角色 (role==0) 动作保持为0

        if self.use_normalization:
            actions_norm = self.action_normalizer.normalize(actions)

        return actions.cpu().numpy(), raw_actions.cpu().numpy()

    def start_large_step(self, large_step, upper_state, upper_obs, last_role, roles):
        self.current_large_step_data = {
            'big_step': large_step,
            'initial_upper_state': upper_state.copy(),
            'initial_upper_obs': upper_obs.copy(),
            'last_a': last_role.copy(),
            'a': roles.copy(),
            'small_step_rewards': []
        }

    def store_small_step_reward(self, reward):
        self.current_large_step_data['small_step_rewards'].append(reward)

    def end_large_step(self, upper_reward, done):
        rewards = self.current_large_step_data['small_step_rewards']
        g_reward = sum((self.gamma ** i) * r for i, r in enumerate(rewards))
        #print(f"g_reward: {g_reward}")
        cumulative_reward = g_reward + upper_reward
        #print(f"cumulative_reward: {cumulative_reward}")

        self.upper_buffer.store_transition(
            self.current_large_step_data['big_step'],
            self.current_large_step_data['initial_upper_state'],
            self.current_large_step_data['initial_upper_obs'],
            self.current_large_step_data['last_a'],
            self.current_large_step_data['a'],
            cumulative_reward * 0.1,
            done
        )

        return cumulative_reward

    def store_lower_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        self.lower_buffer.add(obs, state, actions, reward, next_obs, next_state, done)

    def update_lower(self):
        if len(self.lower_buffer) < self.args.batch_size:
            return None, None, None, None

        if self.use_per:
            obs, states, actions, rewards, next_obs, next_states, dones, indices, is_weights = \
                self.lower_buffer.sample(self.args.batch_size, self.device)
        else:
            obs, states, actions, rewards, next_obs, next_states, dones = self.lower_buffer.sample(self.args.batch_size,
                                                                                                   self.device)
            is_weights = torch.ones(self.args.batch_size, 1).to(self.device)
            indices = None

        critic1_loss, critic2_loss = self.update_critic(obs, states, actions, rewards, next_obs, next_states, dones,
                                                        is_weights, indices)
        service_loss = torch.tensor(0.0).to(self.device)

        if self.lower_updates % self.args.policy_freq == 0:
            service_loss = self.update_actor(obs, states, actions)
            self.soft_update(self.critic1, self.target_critic1, self.args.tau)
            self.soft_update(self.critic2, self.target_critic2, self.args.tau)
            self.soft_update(self.service_actor, self.service_actor_target, self.args.tau)


        self.lower_updates += 1
        #return (critic1_loss.item(), critic2_loss.item(), service_loss.item())

        return (critic1_loss.item(), critic2_loss.item(),
                service_loss.item() if service_loss.item() != 0 else None)

    def update_critic(self, obs, states, actions, rewards, next_obs, next_states, dones, is_weights, indices):
        critic1_loss = torch.tensor(0.0).to(self.device)
        critic2_loss = torch.tensor(0.0).to(self.device)

        with torch.no_grad():
            batch_size = states.shape[0]
            next_obs = next_obs.reshape(-1, self.obs_dim)
            next_service_action = self.service_actor_target(next_obs)
            next_service_action = self.noise.GaussianNoise_target(action=next_service_action)
            next_service_action = torch.clamp(next_service_action, 0, 1)

            next_actions = torch.zeros(batch_size * self.num_agents, self.action_dim).to(self.device)
            roles = actions[:, :, 0].reshape(-1)
            service_mask = (roles == 1)

            next_actions[service_mask, 0] = 1
            #next_actions[service_mask, 1] = (next_service_action[service_mask, 0] * (self.args.vh_max - self.args.vh_min) + self.args.vh_min)
            next_actions[service_mask, 1] = next_service_action[service_mask, 0]
            #next_actions[service_mask, 2] = torch.round(next_service_action[service_mask, 1] * (2 * np.pi) / (2 * np.pi / self.args.num_directions))
            next_actions[service_mask, 2] = next_service_action[service_mask, 1]
            next_actions[service_mask, 3:] = next_service_action[service_mask, 2:]

            next_actions = next_actions.reshape(batch_size, self.num_agents, self.action_dim)

            q1_next_value = self.target_critic1(next_states, next_actions)
            q2_next_value = self.target_critic2(next_states, next_actions)
            q_next_value = torch.min(q1_next_value, q2_next_value)
            q_targets = rewards + self.gamma * (1 - dones) * q_next_value

        q1_values = self.critic1(states, actions)
        q2_values = self.critic2(states, actions)

        td_errors = (q_targets - q1_values).detach()
        critic1_loss = F.mse_loss(q1_values, q_targets)
        critic2_loss = F.mse_loss(q2_values, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        with torch.no_grad():
            pas = [p for p in self.critic1.parameters() if p.grad is not None]
            total_norm = torch.stack([p.grad.norm(2) for p in pas]).norm(2)

            clip_coef = self.args.max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for p in pas:
                    p.grad.mul_(clip_coef)

            # norm_after = torch.stack([p.grad.norm(2) for p in pas]).norm(2).item()
        #torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.args.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        with torch.no_grad():
            pas = [p for p in self.critic2.parameters() if p.grad is not None]
            total_norm = torch.stack([p.grad.norm(2) for p in pas]).norm(2)

            clip_coef = self.args.max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for p in pas:
                    p.grad.mul_(clip_coef)

            # norm_after = torch.stack([p.grad.norm(2) for p in pas]).norm(2).item()
        #torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.args.max_grad_norm)
        self.critic2_optimizer.step()

        if self.use_per and indices is not None:
            self.lower_buffer.update_priorities(indices, td_errors.cpu().numpy().flatten())

        return critic1_loss, critic2_loss

    def update_actor(self, obs, states, actions):
        batch_size = states.shape[0]
        obs = obs.reshape(-1, self.obs_dim)
        service_mask = (actions[:, :, 0] == 1).reshape(-1)
        service_loss = torch.tensor(0.0).to(self.device)

        if service_mask.sum() > 0:
            service_states = obs[service_mask]
            service_action = self.service_actor(service_states)

            new_service_actions = torch.zeros(service_mask.sum(), self.action_dim).to(self.device)
            new_service_actions[:, 0] = 1
            #new_service_actions[:, 1] = (service_action[:, 0] * (self.args.vh_max - self.args.vh_min) + self.args.vh_min)
            new_service_actions[:, 1] = service_action[:, 0]
            #new_service_actions[:, 2] = torch.round(service_action[:, 1] * (2 * np.pi) / (2 * np.pi / self.args.num_directions))
            new_service_actions[:, 2] = service_action[:, 1]
            new_service_actions[:, 3:] = service_action[:, 2:]

            new_actions = actions.clone().detach().reshape(-1, self.action_dim)
            new_actions[service_mask] = new_service_actions
            new_actions = new_actions.reshape(batch_size, self.num_agents, self.action_dim)
            q_service = self.critic1(states, new_actions)
            service_loss = -torch.mean(q_service)

            self.service_optimizer.zero_grad()
            service_loss.backward(retain_graph=True)
            with torch.no_grad():
                pas = [p for p in self.service_actor.parameters() if p.grad is not None]
                total_norm = torch.stack([p.grad.norm(2) for p in pas]).norm(2)

                clip_coef = self.args.max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in pas:
                        p.grad.mul_(clip_coef)

                #norm_after = torch.stack([p.grad.norm(2) for p in pas]).norm(2).item()
            #torch.nn.utils.clip_grad_norm_(self.service_actor.parameters(), self.args.max_grad_norm)
            self.service_optimizer.step()

        return service_loss

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filepath):
        """保存模型"""
        checkpoint = {
            #'upper_dqn': self.upper_dqn.state_dict(),
            'eval_Q': self.upper_agent.eval_Q_net.state_dict(),
            'target_Q': self.upper_agent.target_Q_net.state_dict(),
            'eval_mix': self.upper_agent.eval_mix_net.state_dict(),
            'target_mix': self.upper_agent.target_mix_net.state_dict(),
            'service_actor': self.service_actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")

    '''def save_model(self):
        # path to save the model
        for agent_id in range(self.UAVs):
            model_path = self.args.model_path + f'/agent{agent_id}'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.actor_cur_list[agent_id].state_dict(), model_path + '/actor.pkl')
            torch.save(self.critic_cur_list[agent_id].state_dict(), model_path + '/critic.pkl')'''

    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        #self.upper_dqn.load_state_dict(checkpoint['upper_dqn'])
        #self.upper_dqn_target.load_state_dict(checkpoint['upper_dqn'])  # 同时加载到 target
        self.service_actor.load_state_dict(checkpoint['service_actor'])
        self.service_actor_target.load_state_dict(checkpoint['service_actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.target_critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic2.load_state_dict(checkpoint['critic2'])
        # 2. 加载上层 ACORM (重要补充！)
        self.upper_agent.eval_Q_net.load_state_dict(checkpoint['eval_Q'])
        self.upper_agent.target_Q_net.load_state_dict(checkpoint['target_Q'])
        self.upper_agent.eval_mix_net.load_state_dict(checkpoint['eval_mix'])
        self.upper_agent.target_mix_net.load_state_dict(checkpoint['target_mix'])
        print(f"模型已加载: {filepath}")

    def load_lower_model(self, filepath):
        """专门用于加载预训练的下层 TD3 模型 (通常由阈值法训练得到)"""
        if not os.path.exists(filepath):
            print(f"[警告] 找不到下层预训练模型: {filepath}，将随机初始化下层网络！")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        # 加载 Actor 和 Critic
        self.service_actor.load_state_dict(checkpoint['service_actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])

        # 同步 Target 网络 (非常重要)
        self.service_actor_target.load_state_dict(checkpoint['service_actor'])
        self.target_critic1.load_state_dict(checkpoint['critic1'])
        self.target_critic2.load_state_dict(checkpoint['critic2'])

        print(f"成功加载下层预训练模型 (Actor & Critic): {filepath}")

