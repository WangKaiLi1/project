import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.use('Agg')
from matplotlib.patches import Circle
import os


class Visualizer:
    """可视化工具"""
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"training_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)


        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_training_curves(self, data_dir, save_name='training_curves.png'):
        """绘制训练曲线"""
        # 加载数据
        rewards = np.load(os.path.join(data_dir, 'episode_rewards.npy'))
        completion_rates = np.load(os.path.join(data_dir, 'episode_completion_rates.npy'))
        delays = np.load(os.path.join(data_dir, 'episode_delays.npy'))
        energies = np.load(os.path.join(data_dir, 'episode_energies.npy'))
        fly_energies = np.load(os.path.join(data_dir, 'episode_fly_energies.npy'))
        uncovered = np.load(os.path.join(data_dir, 'episode_uncovered_counts.npy'))

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MADDPG Training Curves', fontsize=16, fontweight='bold')

        # 平滑函数
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window) / window, mode='valid')

        episodes = np.arange(len(rewards))

        # 1. 奖励曲线
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        axes[0, 0].plot(episodes[len(episodes) - len(smooth(rewards)):],
                        smooth(rewards), color='red', linewidth=2, label='Smoothed')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 任务完成率
        axes[0, 1].plot(episodes, completion_rates, alpha=0.3, color='green')
        axes[0, 1].plot(episodes[len(episodes) - len(smooth(completion_rates)):],
                        smooth(completion_rates), color='darkgreen', linewidth=2)
        axes[0, 1].set_title('Task Completion Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 平均延迟
        axes[1, 0].plot(episodes, delays, alpha=0.3, color='orange')
        axes[1, 0].plot(episodes[len(episodes) - len(smooth(delays)):],
                        smooth(delays), color='darkorange', linewidth=2)
        axes[1, 0].set_title('Average Task Delay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Delay (s)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 能耗对比
        axes[1, 1].plot(episodes, energies, alpha=0.3, color='purple', label='Compute Energy')
        axes[1, 1].plot(episodes, fly_energies, alpha=0.3, color='brown', label='Fly Energy')
        axes[1, 1].plot(episodes[len(episodes) - len(smooth(energies)):],
                        smooth(energies), color='darkviolet', linewidth=2)
        axes[1, 1].plot(episodes[len(episodes) - len(smooth(fly_energies)):],
                        smooth(fly_energies), color='saddlebrown', linewidth=2)
        axes[1, 1].set_title('Energy Consumption')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        '''# 5. 未覆盖终端数
        axes[2, 0].plot(episodes, uncovered, alpha=0.3, color='red')
        axes[2, 0].plot(episodes[len(episodes) - len(smooth(uncovered)):],
                        smooth(uncovered), color='darkred', linewidth=2)
        axes[2, 0].set_title('Uncovered Terminals Count')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].grid(True, alpha=0.3)

        # 6. 损失曲线
        try:
            critic_losses = np.load(os.path.join(data_dir, 'critic_losses.npy'))
            role_losses = np.load(os.path.join(data_dir, 'role_losses.npy'))

            steps = np.arange(len(critic_losses))
            axes[2, 1].plot(steps, critic_losses, alpha=0.5, label='Critic Loss')
            axes[2, 1].plot(steps, role_losses, alpha=0.5, label='Role Loss')
            axes[2, 1].set_title('Training Losses')
            axes[2, 1].set_xlabel('Update Step')
            axes[2, 1].set_ylabel('Loss')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        except:
            axes[2, 1].text(0.5, 0.5, 'Loss data not available',
                            ha='center', va='center', transform=axes[2, 1].transAxes)'''

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存到: {save_path}")

    def plot_uav_trajectories(self, uav_trajectories, episode_idx, args,
                              terminal_trajectories=None, save_name=None):
        """
        绘制 UAV 和终端的轨迹图

        Args:
            uav_trajectories: list or np.array, shape [steps, num_uavs, 3] (x,y,z)
            episode_idx: int or str, 当前 episode 的索引或名称（如 'best'）
            args: 参数对象，包含地图大小、充电站位置等
            terminal_trajectories: list or np.array, shape [steps, num_terminals, 2] (x,y)
                                   (可选，如果不传则不画终端)
            save_name: str, 自定义保存文件名 (可选)
        """
        # 1. 数据预处理：确保转为 numpy 数组
        uav_traj = np.array(uav_trajectories)

        # 2. 创建画布
        fig, ax = plt.subplots(figsize=(10, 10))

        # 设置坐标轴范围
        ax.set_xlim(0, args.area_length)
        ax.set_ylim(0, args.area_width)

        # 3. 绘制充电站 (红色加号 P)
        # 假设 args.charger_locations 是列表或数组
        charger_locs = np.array(args.charger_locations)
        ax.scatter(charger_locs[:, 0], charger_locs[:, 1],
                   c='red', marker='P', s=200, label='Charge Station', zorder=5)

        # 4. 绘制终端 (灰色点)
        if terminal_trajectories is not None:
            term_traj = np.array(terminal_trajectories)

            # 判断传入的是完整的历史轨迹还是单帧位置
            if term_traj.ndim == 3:
                # shape: [steps, num_terminals, 2] -> 取最后一步的位置
                final_term_pos = term_traj[-1]
            else:
                # shape: [num_terminals, 2] -> 本身就是位置
                final_term_pos = term_traj

            ax.scatter(final_term_pos[:, 0], final_term_pos[:, 1],
                       c='gray', marker='.', s=50, alpha=0.6, label='Terminals (End Pos)', zorder=2)

        # 5. 绘制 UAV 轨迹
        # uav_traj shape: [steps, num_uavs, 3]
        num_uavs = uav_traj.shape[1]
        # 生成不同的颜色
        colors = plt.cm.get_cmap('tab10', num_uavs)

        for i in range(num_uavs):
            # 提取单个 UAV 的 X, Y 路径
            path_x = uav_traj[:, i, 0]
            path_y = uav_traj[:, i, 1]
            color = colors(i)

            # 绘制路径线
            ax.plot(path_x, path_y, color=color, linewidth=2, alpha=0.8, label=f'UAV {i}')

            # 绘制起点 (圆点 o)
            ax.scatter(path_x[0], path_y[0], color=color, marker='o', s=100, edgecolors='black', zorder=4)

            # 绘制终点 (叉号 X)
            ax.scatter(path_x[-1], path_y[-1], color=color, marker='X', s=150, edgecolors='black', zorder=4)

        # 6. 图表装饰
        title_str = f'Trajectory - {episode_idx}' if isinstance(episode_idx,
                                                                str) else f'Trajectory - Episode {episode_idx}'
        ax.set_title(title_str, fontsize=16)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 图例放外面一点防止遮挡
        ax.grid(True, linestyle='--', alpha=0.5)

        # 7. 保存图片
        if save_name is None:
            save_name = f'traj_{episode_idx}.png'

        save_path = os.path.join(self.save_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  轨迹图已保存: {save_path}")

    def plot_multiple_trajectories(self, data_dir, args, world, episodes_to_plot=[0, 50, 100, 200]):
        """绘制多个episode的轨迹对比"""
        trajectories = np.load(os.path.join(data_dir, 'uav_trajectories.npy'),allow_pickle=True)

        terminal_positions = [[t.location.x, t.location.y] for t in world.terminals]

        for ep_idx in episodes_to_plot:
            if ep_idx < len(trajectories):
                self.plot_uav_trajectories(
                    trajectories[ep_idx],
                    ep_idx,
                    args,
                    save_name=f'trajectory_episode_{ep_idx}.png'
                )

    def plot_best_trajectory(npy_path):
        """
        读取npy文件并绘制轨迹图
        npy_path: 'logs/data_xxx/best_trajectory_info.npy' 的路径
        """
        # 加载数据
        data = np.load(npy_path, allow_pickle=True).item()

        traj = data['trajectory']  # Shape: (Steps, Num_UAV, 2)
        terminals = data['terminals']  # Shape: (Num_Terminals, 2)
        cs_pos = data['cs_pos']  # Shape: (Num_CS, 2)

        plt.figure(figsize=(8, 8))

        # 1. 绘制终端位置 (灰色点)
        plt.scatter(terminals[:, 0], terminals[:, 1], c='gray', alpha=0.6, label='Terminals', s=20)

        # 2. 绘制充电站位置 (红色十字)
        plt.scatter(cs_pos[:, 0], cs_pos[:, 1], marker='+', c='red', s=150, linewidths=3, label='Charge Station')

        # 3. 绘制每架无人机的轨迹
        num_uavs = traj.shape[1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 选几个好看的颜色

        for i in range(num_uavs):
            x = traj[:, i, 0]
            y = traj[:, i, 1]

            # 画轨迹线
            plt.plot(x, y, color=colors[i % len(colors)], label=f'UAV {i}', alpha=0.8)

            # 画起点 (圆圈) 和 终点 (叉)
            plt.scatter(x[0], y[0], color=colors[i % len(colors)], marker='o', s=100, edgecolors='k')
            plt.scatter(x[-1], y[-1], color=colors[i % len(colors)], marker='x', s=100, linewidths=2)

        plt.xlim(0, 1000)  # 根据你的环境范围调整
        plt.ylim(0, 1000)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Best Episode Trajectory Analysis')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
