import numpy as np
import os
import json
from datetime import datetime
import csv


class Logger:
    """训练日志记录器"""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建时间戳
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

        self.recl_losses = []
        self.qmix_losses = []
        self.role_decoder_losses = []

        self.critic1_losses = []
        self.critic2_losses = []
        self.service_losses = []

        # UAV轨迹（每个episode保存一次）
        self.uav_trajectories = []
        self.best_trajectory = None
        self.best_terminals = None
        self.best_roles = None
        self.best_battery = None
        self.best_term = None

        # 写入表头
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write(f"Started at: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

        # 新增：定义 CSV 历史记录文件的路径
        self.history_csv_file = os.path.join(log_dir, f'role_battery_history_{self.timestamp}.csv')

    def log_episode(self, episode, upper_reward, reward, completion_rate, delay, energy,
                    fly_energy, uncovered_count,
                    upper_actor_loss, upper_critic_loss, role_decoder_loss,
                    critic1_loss, critic2_loss, service_loss):
        """记录episode数据"""
        self.episode_rewards.append(upper_reward)
        self.episode_rewards.append(reward)
        self.episode_completion_rates.append(completion_rate)
        self.episode_delays.append(delay)
        self.episode_energies.append(energy)
        self.episode_fly_energies.append(fly_energy)
        self.episode_uncovered_counts.append(uncovered_count)
        #self.uav_trajectories.append(uav_locations)

        # 写入日志文件
        log_msg = (f"Episode {episode}: "
                   f"Upper Reward={upper_reward:.2f}, "
                   f"Reward={reward:.2f}, "
                   f"Completion={completion_rate:.2%}, "
                   f"Delay={delay:.2f}s, "
                   f"Energy={energy:.2f}J, "
                   f"FlyEnergy={fly_energy:.2f}J, "
                   f"Uncovered={uncovered_count:.0f}, "
                   f"Upper_Actor_loss={upper_actor_loss:.4f}, "
                   f"Upper_Critic_loss={upper_critic_loss:.4f}, "
                   f"Role_Decoder_loss={role_decoder_loss:.4f}, "
                   f"Critic1_loss={critic1_loss:.4f}, "
                   f"Critic2_loss={critic2_loss:.4f}, "
                   f"Service_loss={service_loss:.4f} \n")

        with open(self.log_file, 'a') as f:
            f.write(log_msg)

        # 每10个episode打印一次
        if episode % 1 == 0:
            print(log_msg.strip())

    def log_episode_history(self, episode, roles_list, battery_list, term_list):
        """
        实时追加保存每个 Episode 的角色、电量和任务数
         roles_list: shape (steps, num_uavs)
         battery_list: shape (steps, num_uavs)
         term_list: shape (steps,)
        """
        file_exists = os.path.isfile(self.history_csv_file)

        # 以 'a' (追加) 模式打开，程序意外终止也不会丢失之前写入的数据
        with open(self.history_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # 如果文件刚创建，先写入表头
            if not file_exists:
                num_uavs = len(roles_list[0])
                header = ['Episode', 'LargeStep', 'TaskCount']
                for i in range(num_uavs):
                    header.extend([f'UAV{i}_Role', f'UAV{i}_Battery'])
                writer.writerow(header)

            # 遍历当前 Episode 的每一个大时间步，写入一行
            for step in range(len(term_list)):
                row = [episode, step, term_list[step]]
                for i in range(len(roles_list[step])):
                    row.append(roles_list[step][i])
                    # 保留4位小数，减小文件体积
                    row.append(round(battery_list[step][i], 4))
                writer.writerow(row)

    def log_losses(self, critic1_loss, critic2_loss, service_loss):
        """记录损失"""
        if critic1_loss is not None:
            self.critic1_losses.append(critic1_loss)
            self.critic2_losses.append(critic2_loss)
            self.service_losses.append(service_loss)

    def log_upper_losses(self, critic_loss, actor_loss):
        """记录损失"""
        if critic_loss is not None:
            self.upper_critic_losses.append(critic_loss)
            self.upper_actor_losses.append(actor_loss)

    def log_recl_losses(self, qmix_loss, recl_loss, role_decoder_loss):
        """记录损失"""
        if qmix_loss is not None:
            self.recl_losses.append(recl_loss)
            self.qmix_losses.append(qmix_loss)
            self.role_decoder_losses.append(role_decoder_loss)

    def save_data(self):
        """保存所有数据为npy文件"""
        data_dir = os.path.join(self.log_dir, f'data_{self.timestamp}')
        os.makedirs(data_dir, exist_ok=True)

        np.save(os.path.join(data_dir, 'episode_rewards.npy'),
                np.array(self.episode_rewards))
        np.save(os.path.join(data_dir, 'episode_completion_rates.npy'),
                np.array(self.episode_completion_rates))
        np.save(os.path.join(data_dir, 'episode_delays.npy'),
                np.array(self.episode_delays))
        np.save(os.path.join(data_dir, 'episode_energies.npy'),
                np.array(self.episode_energies))
        np.save(os.path.join(data_dir, 'episode_fly_energies.npy'),
                np.array(self.episode_fly_energies))
        np.save(os.path.join(data_dir, 'episode_uncovered_counts.npy'),
                np.array(self.episode_uncovered_counts))
        np.save(os.path.join(data_dir, 'critic1_losses.npy'),
                np.array(self.critic1_losses))
        np.save(os.path.join(data_dir, 'critic2_losses.npy'),
                np.array(self.critic2_losses))
        np.save(os.path.join(data_dir, 'upper_actor_losses.npy'),
                np.array(self.upper_actor_losses))
        np.save(os.path.join(data_dir, 'service_losses.npy'),
                np.array(self.service_losses))
        np.save(os.path.join(data_dir, 'upper_critic_losses.npy'),
                np.array(self.upper_critic_losses))
        np.save(os.path.join(data_dir, 'recl_losses.npy'),
                np.array(self.recl_losses))
        np.save(os.path.join(data_dir, 'qmix_losses.npy'),
                np.array(self.qmix_losses))
        np.save(os.path.join(data_dir, 'role_decoder_losses.npy'),
                np.array(self.role_decoder_losses))

        # 保存轨迹数据
        #np.save(os.path.join(data_dir, 'uav_trajectories.npy'),
        #        np.array(self.uav_trajectories, dtype=object), allow_pickle=True)

        print(f"\n数据已保存到: {data_dir}")

        return data_dir

    def save_traj(self):
        data_dir = os.path.join(self.log_dir, f'data_{self.timestamp}')
        os.makedirs(data_dir, exist_ok=True)

        np.save(os.path.join(data_dir, 'uav_trajectories.npy'),
                np.array(self.best_trajectory, dtype=object), allow_pickle=True)
        np.save(os.path.join(data_dir, 'terminal_pos.npy'),
                np.array(self.best_terminals, dtype=object), allow_pickle=True)

        #print(f"\n轨迹数据已保存到: {data_dir}")

    def save_role(self):
        data_dir = os.path.join(self.log_dir, f'data_{self.timestamp}')
        os.makedirs(data_dir, exist_ok=True)

        np.save(os.path.join(data_dir, 'best_roles.npy'),
                np.array(self.best_roles, dtype=object), allow_pickle=True)
        np.save(os.path.join(data_dir, 'best_battery.npy'),
                np.array(self.best_battery, dtype=object), allow_pickle=True)
        np.save(os.path.join(data_dir, 'best_term.npy'),
                np.array(self.best_term, dtype=object), allow_pickle=True)

        print(f"\n角色电量数据已保存到: {data_dir}")

    def get_statistics(self):
        """获取统计信息"""
        if len(self.episode_rewards) == 0:
            return {}

        # 计算最近100个episode的平均值
        #recent = min(100, len(self.episode_rewards))

        stats = {
            'avg_reward': np.mean(self.episode_rewards[-1]),
            'avg_completion_rate': np.mean(self.episode_completion_rates[-1]),
            'avg_delay': np.mean(self.episode_delays[-1]),
            'avg_energy': np.mean(self.episode_energies[-1]),
            'avg_fly_energy': np.mean(self.episode_fly_energies[-1]),
            'avg_uncovered': np.mean(self.episode_uncovered_counts[-1]),
        }

        return stats