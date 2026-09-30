from collections import deque
import numpy as np
import math
import torch
from sklearn.cluster import KMeans

from env.entities.TaskDistribution import TaskDistribution
from env.entities.UAV import UAV
from env.entities.Terminal import Terminal
from env.entities.ChargeStation import ChargeStation
from common.convert import cover_MB_to_b, cover_dBm_to_W


class MECWorld(object):
    def __init__(self, args):
        self.args = args
        self.current_step = 0   #  当前小时间步
        self.current_large_step = 0  # 当前大时间步
        self.uavs = []
        self.terminals = []
        self.charge_stations = []

        self.total_tasks_generated = 0
        self.total_tasks_completed = 0
        self.terminal_local_energy = []  # 终端本地计算能耗[1....N]
        self.terminal_trans_energy = []  # 终端传输能耗
        self.uav_compute_energy = []  # UAV计算能耗
        self.uav_fly_energy = []  # UAV飞行能耗[1...U]
        self.average_delay = []  # 平均任务完成延迟
        self.average_energy = []
        self.task_completion_rate = []  # 任务完成率
        self.uncovered_terminals_count = []  # 每个时间步无覆盖的终端数

        self.selected_terminals = []
        self.large_tasks = []

        # ========== 新增：任务生成相关 ==========
        # 滑动窗口：记录当前大时间步内每个小时间槽的任务生成数
        #self.task_window = deque(maxlen=args.steps_per_large)
        # 任务生成分布参数
        self.task_distribution = TaskDistribution(args)
        # 滑动窗口：记录前N个大时间步的任务数
        self.task_history_window = deque(maxlen=args.history_window_size)
        # 当前大时间步的目标任务数
        self.current_large_step_tasks = 0

    def initialize_entities(self):
        cluster_center = self.initialize_terminals()
        #[[225.29020185 524.60937848][607.52784948 381.90603221]]
        # 初始化无人机,网格均匀分布
        if self.args.num_uavs == 2:
            uav_positions = [
                [250, 250],  # 左下
                [750, 750],  # 右上
            ]
        else:
            # 网格分布，避免边界
            grid_size = int(np.ceil(np.sqrt(self.args.num_uavs)))
            uav_positions = []
            margin = 100  # 边界留白

            for i in range(self.args.num_uavs):
                row = i // grid_size
                col = i % grid_size
                x = margin + (col + 0.5) * (self.args.area_length - 2 * margin) / grid_size
                y = margin + (row + 0.5) * (self.args.area_width - 2 * margin) / grid_size
                uav_positions.append([x, y])
        '''
        uav_positions = np.array([
            cluser_center[0],
            cluser_center[1],
        ])'''
        initial_battery_ratios = [0.8, 0.6, 0.4, 0.7]
        for i, pos in enumerate(uav_positions):
            #height = np.random.choice(self.args.height, self.args.charge_height)
            uav = UAV(i, self.args, [pos[0], pos[1], self.args.height])
            if i < len(initial_battery_ratios):
                initial_ratio = initial_battery_ratios[i]
            else:
                initial_ratio = np.random.uniform(0.5, 1.0)
            
            uav.battery = uav.battery_capacity * initial_ratio
            #uav.battery = uav.battery_capacity * 0.8
            self.uavs.append(uav)

        # 初始化充电站
        for i, loc in enumerate(self.args.charger_locations):
            station = ChargeStation(i, self.args, loc)
            self.charge_stations.append(station)

    def initialize_terminals(self):
        # 随机生成所有终端位置
        terminal_positions = []
        for i in range(self.args.num_terminals):
            x = np.random.uniform(150, self.args.area_length - 150)
            y = np.random.uniform(150, self.args.area_width - 150)
            terminal_positions.append([x, y])

        terminal_positions = np.array(terminal_positions)

        # 使用K-means聚类
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(terminal_positions)
        cluster_centers = kmeans.cluster_centers_

        #print(f"平衡前簇大小分布: {np.bincount(cluster_labels)}")

        # 均衡簇大小（确保每个簇用户数相等）
        cluster_label, cluster_centers = self.balance_clusters(
            terminal_positions,
            cluster_labels,
            cluster_centers,
            num_clusters
        )
        #print(f"平衡后簇大小分布: {np.bincount(cluster_label)}")
        #print(cluster_centers)

        # 创建终端并分配簇
        for i in range(self.args.num_terminals):
            x, y = terminal_positions[i]
            cluster_id = cluster_label[i]

            terminal = Terminal(i, self.args, [x, y], cluster_id=cluster_id)
            terminal.cluster_center = cluster_centers[cluster_id]
            #terminal.generate_task()
            self.terminals.append(terminal)
            #self.total_tasks_generated += 1

        return cluster_centers

    def balance_clusters(self, positions, labels, centers, num_clusters):
        """均衡簇大小，确保每个簇用户数相等"""
        target_size = len(positions) / num_clusters
        # 统计每个簇的当前大小
        cluster_sizes = np.bincount(labels, minlength=num_clusters)
        # 找出过大和过小的簇
        while not np.all(cluster_sizes == target_size):
            # 找到最大和最小的簇
            max_cluster = np.argmax(cluster_sizes)
            min_cluster = np.argmin(cluster_sizes)

            if cluster_sizes[max_cluster] <= target_size:
                break

            # 从最大簇中找到距离最小簇中心最近的用户
            max_cluster_users = np.where(labels == max_cluster)[0]
            distances = np.linalg.norm(
                positions[max_cluster_users] - centers[min_cluster],
                axis=1
            )

            # 将最近的用户转移到最小簇
            transfer_user = max_cluster_users[np.argmin(distances)]
            labels[transfer_user] = min_cluster

            # 更新簇大小
            cluster_sizes = np.bincount(labels, minlength=num_clusters)

        """重新计算簇中心"""
        centers = np.zeros((num_clusters, 2))
        for i in range(num_clusters):
            cluster_positions = positions[labels == i]
            if len(cluster_positions) > 0:
                centers[i] = np.mean(cluster_positions, axis=0)

        return labels, centers

    def reset_for_large_step(self):
        """
        每个大时间步开始时调用
        """
        # 1. 将上一个大时间步的任务数加入滑动窗口
        if self.current_large_step > 0:
            self.task_history_window.append(self.current_large_step_tasks)

        # 2. 计算当前大时间步的目标任务数（带噪声）
        base_tasks = self.task_distribution.get_task_count(self.current_large_step)
        #self.current_large_step_tasks = self.task_distribution.get_task_count(self.current_large_step)
        #print(f"大时间步{self.current_large_step + 1}任务数{self.current_large_step_tasks}")
        noise = np.random.normal(0, self.args.task_noise_std)
        self.current_large_step_tasks = int(np.clip(
            base_tasks + noise,
            self.args.min_tasks_per_large_step,
            self.args.max_tasks_per_large_step
        ))

        # 3. 重置小时间步计数
        self.current_step = 0
        self.total_tasks_generated = 0
        self.total_tasks_completed = 0
        self.terminal_local_energy = []
        self.terminal_trans_energy = []
        self.uav_compute_energy = []
        self.uav_fly_energy = []
        self.average_delay = []
        self.average_energy = []
        self.task_completion_rate = []
        self.uncovered_terminals_count = []

        self.selected_terminals = []
        self.large_tasks = []
        # 动态用户关联
        cluster_labels, cluster_centers = self.dynamic_user_association()

        # 4. 重置所有终端的任务状态
        for terminal in self.terminals:
            terminal.is_covered = False
            terminal.has_task = False
            terminal.current_task = None

        #3.6新增
        #for uav in self.uavs:
        #    uav.is_charging = False

        # 2. 生成当前时间槽的新任务
        self.generate_tasks_for_slot()

    def generate_tasks_for_slot(self):
        """
        为当前小时间槽生成任务
        每个小时间槽生成的任务数 = 当前大时间步目标任务数
        """
        '''# 清空上一个时间槽的任务标记
        for terminal in self.terminals:
            terminal.has_task = False'''

        if self.current_step % 100 == 0:
            # 随机选择 current_large_step_tasks 个终端生成任务
            noise = np.random.normal(0, self.args.task_noise_std/5)
            num_task = int(np.clip(
                self.current_large_step_tasks + noise,
                self.args.min_tasks_per_large_step,
                self.args.max_tasks_per_large_step
            ))
            num_tasks = min(num_task, self.args.num_terminals)
            self.large_tasks.append(num_tasks)

            self.selected_terminals = np.random.choice(
                self.terminals,
                size=num_tasks,
                replace=False
            )

            #print(f"{self.current_large_step + 1}中第{self.current_step + 1}小步的任务数为{num_tasks}")

        # 为选中的终端生成任务
        for terminal in self.selected_terminals:
            terminal.generate_task()
            terminal.has_task = True
            self.total_tasks_generated += 1

    def step(self, actions):
        """
        执行物理状态更新
        actions: {uav_id: action_dict}
        """
        step_fly_energy = 0  # 当前时间步的总飞行能耗
        for uav in self.uavs:
            action = actions[uav.uav_id]
            uav.role = action[0]
            #print(f"UAV {uav.uav_id} 动作: {action}")
            #print(f"UAV {uav.uav_id} 角色: {uav.role}")
            fly_energy = self.execute_uav_action(uav, action)
            step_fly_energy += fly_energy

        self.uav_fly_energy.append(step_fly_energy)
        self.handle_tasks_globally(actions)
        self.process_uncovered_terminals()
        completed_rate = self.record_statistics()
        self.current_step += 1
        if self.current_step >= self.args.steps_per_large:   # ==
            self.current_large_step_tasks = np.mean(self.large_tasks)
        return completed_rate

    def execute_uav_action(self, uav, action):
        if uav.role == 0:  # 充电角色
            fly_energy = self.execute_charge_action(uav, action)
        else:  # 服务角色
            fly_energy = self.execute_service_action(uav, action)

        return fly_energy

    def execute_charge_action(self, uav, action):
        # 找到最近的充电站
        target_station = uav.find_nearest_charge_station(self.charge_stations)
        if target_station is None:
            print(f"UAV {uav.uav_id} 未找到充电站")
            return

        fly_energy = uav.charge_move(target_station)

        return fly_energy

    def execute_service_action(self, uav, action):
        v = action[1]
        direction_idx = action[2]
        direction = direction_idx * (2 * np.pi / self.args.num_directions)
        offload_ratios = action[3:]
        fly_energy = uav.service_move(v, direction)

        '''print(f"UAV {uav.uav_id} 服务移动: "
              f"{uav.location.toString()},"
              f"飞行能耗{fly_energy:.2f}J,"
              f"电量{uav.get_battery_ratio():.2%}")'''

        '''# 任务卸载
        covered_terminals = self.get_covered_terminals(uav)
        # 只处理有任务的终端
        task_terminals = [t for t in covered_terminals if t.has_task and t.current_task]
        # 匹配卸载比例（按max_terminals_per_slot对齐）
        for i, terminal in enumerate(task_terminals):
            if i < len(offload_ratios):
                offload_ratio = float(offload_ratios[i])
                compute_per_user = uav.compute_resource / len(task_terminals)
                self.process_offload(uav, terminal, offload_ratio, compute_per_user)'''

        #self.process_offload(uav, offload_ratios)
        return fly_energy

    def handle_tasks_globally(self, actions):
        """
        全局判定：每个有任务的终端选择距离最近的服务型无人机
        """
        # 1. 筛选当前处于“服务角色”且未坠毁的无人机
        active_service_uavs = [uav for uav in self.uavs if uav.role == 1 and not uav.is_crashed]

        # 2. 遍历所有有任务的终端
        for terminal in self.terminals:
            if not (terminal.has_task and terminal.current_task):
                continue

            # 寻找覆盖该终端的所有 UAV
            candidate_uavs = []
            for uav in active_service_uavs:
                dist = self.compute_distance_2d(
                    uav.location.get_position()[:2],
                    terminal.location.get_position()[:2]
                )
                if dist <= self.args.coverage_radius:
                    candidate_uavs.append((uav, dist))

            if not candidate_uavs:
                continue  # 没人覆盖，留给 process_uncovered_terminals 处理

            # 3. 就近原则：选择距离最近的 UAV
            # 按距离排序，取第一个
            candidate_uavs.sort(key=lambda x: x[1])
            best_uav, min_dist = candidate_uavs[0]

            # 标记终端已被覆盖
            terminal.is_covered = True

            # 4. 确定该 UAV 对应的卸载比例
            # 注意：actions[uav_id] 的格式是 [role, v, dir, ratio0, ratio1, ... ratioN]
            # 这里需要某种逻辑将终端映射到 ratio 索引，通常按终端 ID 或 覆盖顺序
            # 假设我们依然采用原本的逻辑：该 UAV 覆盖的所有终端中的索引
            uav_covered_list = self.get_uav_covered_list(best_uav)
            try:
                task_index = uav_covered_list.index(terminal)
                # action 从索引 3 开始是卸载比例
                offload_ratio = float(actions[best_uav.uav_id][3 + task_index])
                #print(offload_ratio)
                #offload_ratio = np.clip(offload_ratio, 0.9, 1)
            except (ValueError, IndexError):
                offload_ratio = 1.0  # 默认全卸载或处理异常

            # 5. 执行卸载计算
            # 计算该 UAV 此时总共负担了多少个终端的任务，用于平分计算资源
            total_task_count = len([t for t in uav_covered_list if t.has_task])
            compute_per_user = best_uav.compute_resource / max(1, total_task_count)

            self.process_offload(best_uav, terminal, offload_ratio, compute_per_user)

    def get_uav_covered_list(self, uav):
        """辅助函数：获取某无人机覆盖的所有终端（按 ID 排序保证索引稳定）"""
        covered = []
        for t in self.terminals:
            dist = self.compute_distance_2d(uav.location.get_position()[:2], t.location.get_position()[:2])
            if dist <= self.args.coverage_radius:
                covered.append(t)
        return covered

    def get_covered_terminals(self, uav):
        """获取UAV覆盖范围内的终端"""
        covered = []
        for terminal in self.terminals:
            distance = self.compute_distance_2d(uav.location.get_position()[:2], terminal.location.get_position()[:2])
            if distance <= self.args.coverage_radius and terminal.is_covered == False:
                terminal.is_covered = True
                covered.append(terminal)
        return covered

    def process_offload(self, uav, terminal, offload_ratio, compute_per_user):
        if not terminal.current_task:
            return

        task = terminal.current_task

        if offload_ratio < 0 or offload_ratio > 1:
            return

        #data_sizes[terminal_id] = terminal.current_task.data_size
        #cycles[terminal_id] = terminal.current_task.cycles
        #delay_thresholds[terminal_id] = terminal.current_task.delay_threshold
        #print(f"终端{terminal.terminal_id} 卸载比例: {ratio:.2f}")
        #print(f"终端{terminal.terminal_id} 任务大小: {terminal.current_task.data_size:.2f} MB")

        local_delay = task.local_delay * (1 - offload_ratio)
        local_energy = task.local_energy * (1 - offload_ratio)
        task.local_delay = round(local_delay, 3)
        task.local_energy = local_energy
        #print(f"终端{terminal.terminal_id} 本地延迟: {local_delay:.2f} s")
        #print(f"终端{terminal.terminal_id} 本地能耗: {local_energy:.4f} J")

        trans_rate = self.compute_transmission_rate(uav, terminal)
        #print(f"终端{terminal.terminal_id}和uav{uav.uav_id} 传输速率: {trans_rate / 1e6:.2f} Mbps")
        task_bits = cover_MB_to_b(terminal.current_task.data_size * offload_ratio)
        trans_delay = task_bits / trans_rate if trans_rate > 0 else float('inf')
        terminal.current_task.trans_delay = round(trans_delay, 3)
        #print(f"终端{terminal.terminal_id}和uav{uav.uav_id} 传输延迟: {trans_delay:.3f} s")
        trans_energy = terminal.trans_power * trans_delay
        terminal.current_task.trans_energy = trans_energy
        #print(f"终端{terminal.terminal_id}和uav{uav.uav_id} 传输能耗: {trans_energy:.4f} J")

        total_cycles = task_bits * task.cycles_per_bit
        uav_exe_delay, uav_exe_energy = uav.compute_task(total_cycles, compute_per_user)
        task.uav_exe_delay = uav_exe_delay
        task.uav_exe_energy = uav_exe_energy
        #print(f"终端{terminal.terminal_id}任务在uav{uav.uav_id} 执行延迟: {uav_exe_delay:.2f} s")
        #print(f"终端{terminal.terminal_id}任务在uav{uav.uav_id} 执行能耗: {uav_exe_energy:.4f} J")

        total_delay = max(local_delay, trans_delay + uav_exe_delay)
        task.completed_delay = round(total_delay, 3)
        #print(f"终端{terminal.terminal_id} 总延迟: {total_delay:.2f} s (阈值: {terminal.current_task.delay_threshold:.2f} s)")

        # 判断成功
        if total_delay <= task.delay_threshold:
            task.offload_success = True
            self.total_tasks_completed += 1

    def process_uncovered_terminals(self):
        uncovered_count = 0
        for terminal in self.terminals:
            if terminal.has_task and terminal.current_task and not terminal.is_covered:
                uncovered_count += 1
                #本地计算
                #terminal.current_task.local_delay = terminal.current_task.local_delay
                #terminal.current_task.local_energy = terminal.current_task.local_energy
                terminal.current_task.trans_delay = 0
                terminal.current_task.trans_energy = 0
                terminal.current_task.uav_exe_delay = 0
                terminal.current_task.uav_exe_energy = 0
                terminal.current_task.completed_delay = terminal.current_task.local_delay
                #print(f"终端{terminal.terminal_id} 完全本地延迟: {terminal.current_task.local_delay:.2f} s")
                #print(f"终端{terminal.terminal_id} 完全本地能耗: {terminal.current_task.local_energy:.4f} J")
                #print(f"终端{terminal.terminal_id} 延迟阈值: {terminal.current_task.delay_threshold:.2f} s")

                # 判断是否成功
                if terminal.current_task.local_delay <= terminal.current_task.delay_threshold:
                    terminal.current_task.offload_success = True
                    self.total_tasks_completed += 1

        # 记录统计
        self.uncovered_terminals_count.append(uncovered_count)
        #print(f"\n 时间步{self.current_step}无覆盖终端总数: {uncovered_count}")

    def compute_return_energy(self, uav):
        """
        计算返回最近充电站所需的能量
        """
        # 找到最近的充电站
        target_station = uav.find_nearest_charge_station(self.charge_stations)
        if target_station is None:
            # 没有充电站，返回无穷大（极端情况）
            return float('inf'), None, float('inf')

        distance = target_station.compute_distance(uav.location)
        # 计算飞行时间,使用平均水平速度和最大垂直速度
        avg_horizontal_speed = (uav.max_v_horizontal + uav.min_v_horizontal) / 2
        avg_vertical_speed = (uav.max_v_vertical + uav.min_v_vertical) / 2
        speed = math.sqrt(avg_horizontal_speed ** 2 + avg_vertical_speed ** 2)
        fly_time = distance / speed if speed > 0 else float('inf')
        # 计算飞行能量
        fly_energy = uav.compute_fly_energy(avg_horizontal_speed, avg_vertical_speed, fly_time)
        # 添加安全裕度
        return_energy = fly_energy * uav.safety_margin
        return return_energy

    def check_energy_constraint(self, uav):
        """
        检查能量约束是否满足
        Returns:
            is_safe: bool，是否安全
            energy_shortage: 能量缺口
        """
        return_energy = self.compute_return_energy(uav)
        if return_energy == float('inf'):
            return False, -float('inf')

        energy_margin = uav.battery - return_energy
        if energy_margin >= 0:
            is_safe = True
        else:
            is_safe = False
        energy_shortage = max(0, return_energy - uav.battery)
        return is_safe, energy_shortage

    def dynamic_user_association(self):
        """动态更新用户聚类"""
        # 获取所有终端的当前位置
        terminal_positions = np.array([
            t.location.get_position()[:2] for t in self.terminals
        ])

        # K-means聚类
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, random_state=None)
        cluster_labels = kmeans.fit_predict(terminal_positions)
        cluster_centers = kmeans.cluster_centers_

        # 均衡簇大小
        cluster_labels, cluster_centers = self.balance_clusters(
            terminal_positions,
            cluster_labels,
            cluster_centers,
            num_clusters
        )

        # 更新终端的簇信息
        for i, terminal in enumerate(self.terminals):
            terminal.cluster_id = cluster_labels[i]
            terminal.cluster_center = cluster_centers[cluster_labels[i]]

        # 返回关联信息（用于奖励计算）
        return cluster_labels, cluster_centers

    def record_statistics(self):
        """记录统计信息"""
        total_local_energy = 0
        total_trans_energy = 0
        total_uav_energy = 0
        total_delay = 0
        completed_tasks = 0

        for terminal in self.terminals:
            if terminal.current_task:
                total_local_energy += terminal.current_task.local_energy
                total_trans_energy += terminal.current_task.trans_energy
                total_uav_energy += terminal.current_task.uav_exe_energy
                total_delay += terminal.current_task.completed_delay
                #print(f"任务传输延迟{terminal.current_task.trans_delay}")
                #print(f"任务本地延迟{terminal.current_task.local_delay}")
                #print(f"任务无人机延迟{terminal.current_task.uav_exe_delay}")
                if terminal.current_task.offload_success:
                    completed_tasks += 1

        self.terminal_local_energy.append(total_local_energy)
        self.terminal_trans_energy.append(total_trans_energy)
        #print(f"任务传输延迟{self.terminal_trans_energy}")
        self.uav_compute_energy.append(total_uav_energy)
        total_energy = total_local_energy + total_trans_energy + total_uav_energy
        avg_energy = round(total_energy / self.args.num_terminals, 3)
        self.average_energy.append(avg_energy)
        avg_delay = round(total_delay / self.args.num_terminals, 3)
        self.average_delay.append(avg_delay)
        # 任务完成率
        #print(f"完成任务数{completed_tasks}")
        completion_rate = completed_tasks / self.current_large_step_tasks
        #print(f"任务完成率: {completion_rate:.2%}")
        self.task_completion_rate.append(round(completion_rate, 2))
        return completion_rate

    def get_task_history_features(self):
        """
        获取滑动窗口特征（用于上层状态）
        返回：归一化的任务数序列 [history_window_size]
        """
        if len(self.task_history_window) == 0:
            return np.zeros(self.args.history_window_size, dtype=np.float32)

        # 归一化到 [0, 1]
        window_array = np.array(list(self.task_history_window), dtype=np.float32)
        normalized = window_array / self.args.max_tasks_per_large_step

        # 填充到固定长度
        if len(normalized) < self.args.history_window_size:
            padding = np.zeros(self.args.history_window_size - len(normalized), dtype=np.float32)
            normalized = np.concatenate([padding, normalized])

        return normalized

    def compute_distance_2d(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def compute_transmission_rate(self, uav, terminal):
        """计算传输速率"""
        distance = uav.location.distance_to(terminal.location)
        elevation_angle = math.degrees(math.asin(uav.location.z / distance))
        # LoS 概率
        c = (1 + self.args.wa * math.exp(-self.args.wb * (elevation_angle - self.args.wa)))
        p_los = 1 / c
        # 路径损耗
        lfs = 20 * math.log10(4 * math.pi * distance * self.args.carrier_freq / self.args.light_speed)
        path_loss = p_los * (lfs + self.args.eta_LoS) + (1 - p_los) * (lfs + self.args.eta_NLoS)
        # SINR
        N0 = cover_dBm_to_W(self.args.N0)
        received_power = terminal.trans_power / (10 ** (path_loss / 10))
        sinr = received_power / N0
        # 传输速率（bps）
        trans_rate = self.args.bandwidth * math.log2(1 + sinr)

        return trans_rate