import numpy as np
import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{self.timestamp}.log')

        # 数据存储
        self.episode_rewards = []
        self.episode_completion_rates = []
        self.episode_delays = []
        self.episode_energies = []
        self.episode_fly_energies = []
        self.episode_uncovered_counts = []
        self.upper_critic_losses = []
        self.upper_actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.service_losses = []

        # --- 新增：用于保存最佳模型轨迹的变量 ---
        self.best_reward = -float('inf')
        self.best_episode_data = {
            'trajectory': None,  # 无人机路径
            'terminals': None,  # 终端位置
            'cs_pos': None  # 充电站位置
        }

    def log_episode(self, episode, upper_reward, reward, completion_rate, delay, energy,
                    fly_energy, uncovered_count, uav_locations,
                    terminal_pos, # <-- 传入环境位置
                    upper_actor_loss, upper_critic_loss, critic1_loss, critic2_loss, service_loss):

        # 记录数据
        self.episode_rewards.append(reward)
        self.episode_completion_rates.append(completion_rate)
        self.episode_delays.append(delay)
        self.episode_energies.append(energy)
        self.episode_fly_energies.append(fly_energy)
        self.episode_uncovered_counts.append(uncovered_count)

        # --- 新增：判断并保存最佳轨迹逻辑 ---
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode_data['trajectory'] = np.array(uav_locations)
            self.best_episode_data['terminals'] = np.array(terminal_pos)
            #self.best_episode_data['cs_pos'] = np.array(cs_pos)
            print(f"--- 探测到更优模型! Episode {episode}, Reward: {reward:.2f} ---")

        # 写入日志文件 (保持你原有的逻辑)
        log_msg = (f"Episode {episode}: Reward={reward:.2f}, Completion={completion_rate:.2%}\n")
        with open(self.log_file, 'a') as f:
            f.write(log_msg)

    def log_losses(self, critic1_loss, critic2_loss, service_loss):
        self.critic1_losses.append(critic1_loss)
        self.critic2_losses.append(critic2_loss)
        self.service_losses.append(service_loss)

    def log_upper_losses(self, critic_loss, actor_loss):
        self.upper_critic_losses.append(critic_loss)
        self.upper_actor_losses.append(actor_loss)

    def save_data(self):
        """保存所有数据，包括最佳轨迹"""
        data_dir = os.path.join(self.log_dir, f'data_{self.timestamp}')
        os.makedirs(data_dir, exist_ok=True)

        # 保存常规曲线
        np.save(os.path.join(data_dir, 'episode_rewards.npy'), np.array(self.episode_rewards))
        np.save(os.path.join(data_dir, 'critic1_losses.npy'), np.array(self.critic1_losses))
        np.save(os.path.join(data_dir, 'critic2_losses.npy'), np.array(self.critic2_losses))
        # 修正你原代码中的变量名错误 (role_losses -> upper_actor_losses)
        np.save(os.path.join(data_dir, 'upper_actor_losses.npy'), np.array(self.upper_actor_losses))
        np.save(os.path.join(data_dir, 'upper_critic_losses.npy'), np.array(self.upper_critic_losses))
        np.save(os.path.join(data_dir, 'service_losses.npy'), np.array(self.service_losses))

        # --- 新增：保存最佳轨迹数据为独立文件 ---
        if self.best_episode_data['trajectory'] is not None:
            np.save(os.path.join(data_dir, 'best_trajectory_info.npy'), self.best_episode_data)
            print(f"最佳轨迹数据已保存到: best_trajectory_info.npy")

        print(f"所有训练数据已保存到: {data_dir}")
        return data_dir