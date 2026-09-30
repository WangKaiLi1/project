from collections import deque

from args.Charge_args import get_Charge_args
from args.Terminal_args import get_user_args
from args.UAV_args import get_UAV_args
from args.argument import get_args
from env.mec_world import MECWorld
import numpy as np

class Scenario():
    def make_world(self, args):
        world = MECWorld(args)
        self.reset_world(world, args)
        return world

    def reset_world(self, world, args):
        world.uavs = []
        world.terminals = []
        world.charge_stations = []
        world.initialize_entities()

        #3.10修改
        world.current_large_step = 0  # 当前大时间步
        world.task_history_window = deque(maxlen=args.history_window_size)
        world.current_large_step_tasks = 0

        #world.current_step = 0
        #world.reset_for_large_step()
        '''world.total_tasks_generated = 0
        world.total_tasks_completed = 0
        world.terminal_local_energy = []
        world.terminal_trans_energy = []
        world.uav_compute_energy = []
        world.uav_fly_energy = []
        world.average_delay = []
        world.average_energy = []
        world.task_completion_rate = []
        world.uncovered_terminals_count = []'''

    def reset_role(self, world):
        '''for uav in world.uavs:
            uav.reset()'''
        #world.current_step = 0
        world.reset_for_large_step()
        world.total_tasks_generated = 0
        world.total_tasks_completed = 0
        world.terminal_local_energy = []
        world.terminal_trans_energy = []
        world.uav_compute_energy = []
        world.uav_fly_energy = []
        world.average_delay = []
        world.average_energy = []
        world.task_completion_rate = []
        world.uncovered_terminals_count = []
        # 动态用户关联
        cluster_labels, cluster_centers = world.dynamic_user_association()

    '''def get_obs(self, uav, world):
        obs = []
        # UAV自身状态（归一化到[0,1]）
        obs.append(uav.location.x / world.args.area_length)
        obs.append(uav.location.y / world.args.area_width)
        obs.append(uav.location.z / world.args.height)  # 假设最大高度200m
        obs.append(uav.get_battery_ratio())

        # 最近充电站的相对距离
        nearest_station = uav.find_nearest_charge_station(world.charge_stations)
        if nearest_station:
            # 计算向量
            dx = (nearest_station.location.x - uav.location.x) / world.args.area_length
            dy = (nearest_station.location.y - uav.location.y) / world.args.area_width
            dz = (nearest_station.location.z - uav.location.z) / world.args.height
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # 必须包含方向信息！
            obs.extend([dx, dy, dz, dist])
        else:
            obs.extend([0, 0, 0, 0])

        terminal_obs = []
        for terminal in world.terminals:
            distance = np.linalg.norm(np.array(uav.location.get_position()[:2])
                                      - np.array(terminal.location.get_position()[:2]))
            if distance <= world.args.coverage_radius:
                terminal_obs.append(distance / world.args.area_length)  # 归一化
                if terminal.current_task:
                    terminal_obs.append(terminal.current_task.data_size)
                    terminal_obs.append(terminal.current_task.cycles_per_bit/10.0)
                    terminal_obs.append(terminal.current_task.delay_threshold)
                else:
                    terminal_obs.extend([0, 0, 0])
            else:
                terminal_obs.extend([0, 0, 0, 0])

        # 填充到固定长度
        max_terminal_obs = world.args.num_terminals * 4
        while len(terminal_obs) < max_terminal_obs:
            terminal_obs.append(0)

        obs.extend(terminal_obs[:max_terminal_obs])

        # 其他UAV的相对位置（协作信息）
        for other_uav in world.uavs:
            if other_uav.uav_id != uav.uav_id:
                dx = (other_uav.location.x - uav.location.x) / world.args.area_length
                dy = (other_uav.location.y - uav.location.y) / world.args.area_width
                dz = (other_uav.location.z - uav.location.z) / world.args.height
                obs.extend([dx, dy, dz, other_uav.get_battery_ratio()])

        return np.array(obs, dtype=np.float32)'''

    def get_obs(self, uav, world):
        # --- 1. 提取静态参数，避免重复查找 ---
        L, W, H = world.args.area_length, world.args.area_width, world.args.height

        # 获取唯一的簇中心（去重）
        #cluster_centers = list(set(tuple(t.cluster_center / L) for t in world.terminals))
        #cluster_centers = np.array(cluster_centers)  # 转换为numpy数组
        # 修改 mec_scenery.py
        unique_centers = set(tuple(t.cluster_center / L) for t in world.terminals)
        # 按坐标排序，保证输入顺序固定
        cluster_centers = sorted(list(unique_centers), key=lambda x: (x[0], x[1]))
        cluster_centers = np.array(cluster_centers)

        # --- 3. 开始构建 Obs (全部向量化) ---
        obs_list = []
        # [修改点]：在 obs_list 最前面或最后面加入 One-Hot 身份编码
        # 假设 num_uavs = 2
        #uav_id_onehot = [0.0] * world.args.num_uavs
        #uav_id_onehot[uav.uav_id] = 1.0  # [1, 0] 或 [0, 1]

        #obs_list.extend(uav_id_onehot)  # 加入身份信息
        # [Self Info]: Pos(3) + Battery(1)
        obs_list.extend([
            uav.location.x / L,
            uav.location.y / W,
            uav.location.z / H,
            uav.get_battery_ratio()
        ])
        '''# [Charge Station Info]: 增加向量信息 (dx, dy, dz, dist)
        nearest_station = uav.find_nearest_charge_station(world.charge_stations)
        if nearest_station:
            dx = (nearest_station.location.x - uav.location.x) / L
            dy = (nearest_station.location.y - uav.location.y) / W
            dz = (nearest_station.location.z - uav.location.z) / H
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # 向量化计算
            obs_list.extend([dx, dy, dz, dist])
        else:
            obs_list.extend([0, 0, 0, 0])'''

        # ========== 3. 有任务的终端信息 ==========
        max_terminals = world.args.max_terminals_per_slot
        uav_pos_2d = np.array([uav.location.x, uav.location.y])

        # 收集有任务的终端
        task_terminals = [t for t in world.terminals if t.has_task and t.current_task]

        terminal_features = []
        for terminal in task_terminals:
            # 计算距离
            term_pos = np.array([terminal.location.x, terminal.location.y])
            dist = np.linalg.norm(term_pos - uav_pos_2d)
            # 判断是否在覆盖范围内
            in_range = dist <= world.args.coverage_radius
            # 特征：[归一化距离, 任务大小, 是否在范围内]
            if in_range:
                # 添加任务信息
                features = [(terminal.location.x - uav.location.x) / world.args.coverage_radius,
                            (terminal.location.y - uav.location.y) / world.args.coverage_radius,
                            terminal.current_task.data_size
                            # terminal.current_task.cycles_per_bit / 10.0,
                            # terminal.current_task.delay_threshold
                            ]

                terminal_features.append(features)

        # 填充到固定长度
        while len(terminal_features) < max_terminals:
            terminal_features.append([0.0, 0.0, 0.0])

        # 只取前max_terminals个
        terminal_features = terminal_features[:max_terminals]

        '''
        cover_radius = world.args.coverage_radius

        task_terminals = []  # 有任务且在感知范围内的
        not_task_terminals = []  # 无任务但在感知范围内的

        for term in world.terminals:
            # 计算相对坐标和物理距离
            dx = term.location.x - uav.location.x
            dy = term.location.y - uav.location.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # 严格局部观测
            if dist <= cover_radius:
                # 是否在覆盖半径内
                can_serve = 1.0

                # 基础特征提取
                is_task = 1.0 if (term.has_task and term.current_task) else 0.0
                data_size = term.current_task.data_size if is_task else 0.0

                # 归一化特征 [相对x, 相对y, 任务大小, 是否可服务]
                feat = [dx / cover_radius, dy / cover_radius, data_size, can_serve]

                if is_task:
                    task_terminals.append({'dist': dist, 'feat': feat})
                else:
                    not_task_terminals.append({'dist': dist, 'feat': feat})

        # 有任务的按距离排序 (最优先)
        task_terminals.sort(key=lambda x: x['dist'])
        # 无任务的也按距离排序 (次优先)
        not_task_terminals.sort(key=lambda x: x['dist'])

        # 按优先级组装最终向量
        terminal_features = []

        # 首先填充有任务的终端
        for item in task_terminals:
            if len(terminal_features) < max_terminals:
                terminal_features.append(item['feat'])

        # 填充无任务的终端 (作为环境背景/潜在热点)
        for item in not_task_terminals:
            if len(terminal_features) < max_terminals:
                terminal_features.append(item['feat'])

        # 填充
        # 0,0,0,0 代表"此处无感知信息"
        while len(terminal_features) < max_terminals:
            terminal_features.append([0.0, 0.0, 0.0, 0.0])

        # 只取前max_terminals个
        terminal_features = terminal_features[:max_terminals]'''

        # 展平并加入 obs          obs_list.extend(terminal_feats.flatten())
        # 展平并添加到观测
        for features in terminal_features:
            obs_list.extend(features)

        # 加入终端的簇心
        for center in cluster_centers:
            # 归一化簇中心坐标
            cx = center[0] / L if len(center) > 0 else 0
            cy = center[1] / W if len(center) > 1 else 0
            obs_list.extend([cx, cy])

        # [Other UAVs Info]
        for other_uav in world.uavs:
            if other_uav.uav_id != uav.uav_id:
                dx = (other_uav.location.x - uav.location.x) / L
                dy = (other_uav.location.y - uav.location.y) / W
                dz = (other_uav.location.z - uav.location.z) / H
                obs_list.extend([dx, dy, dz, other_uav.get_battery_ratio()])

        return np.array(obs_list, dtype=np.float32)

    '''def get_reward(self, world):
        hard_penalty = 0
        # 碰撞惩罚， 在碰撞距离内惩罚10
        for i in range(world.args.num_uavs):
            for j in range(i + 1, world.args.num_uavs):
                distance = world.uavs[i].location.distance_to(world.uavs[j].location)
                if distance < world.args.collision_radius:
                    #hard_penalty -= (world.args.collision_radius - distance) * 0.4
                    hard_penalty -= 0.5


        #任务完成率，分层奖励
        #completion_rate = world.task_completion_rate[-1] if world.task_completion_rate else 0
        #if completion_rate >= 0.9:
        #    completion_reward = 70.0 + (completion_rate - 0.9) * 200.0
        #elif completion_rate >= 0.7:
        #    completion_reward = 50.0 + (completion_rate - 0.7) * 100.0
        #else:
        #    completion_reward = completion_rate * 60.0
        # 任务奖励：指数型，但连续
        # rate 0.5 -> 7.4
        # rate 0.7 -> 20.0
        # rate 0.9 -> 54.5
        # rate 1.0 -> 100.0
        # 这样每一小步提升都有明显的梯度
        completion_rate = world.task_completion_rate[-1] if world.task_completion_rate else 0
        #completion_reward = 100.0 * (np.exp(2 * completion_rate) - 1) / (np.exp(2) - 1)
        #completion_reward = completion_rate * 200.0    # 1/2
        #completion_reward = completion_rate * 250.0
        # 指数奖励：0->0, 0.5->12.5, 0.7->24.5, 0.9->40.5, 1.0->50
        #completion_reward = 50 * (completion_rate ** 1.5)  # 指数函数，鼓励高完成率
        completion_reward = 10 * (completion_rate ** 1.5)  # 5

        # 电量惩罚 (只罚不奖！)
        # 只有当电量低于阈值才开始扣分，迫使它去充电
        energy_penalty = 0
        for uav in world.uavs:
            bat = uav.get_battery_ratio()
            if bat < 0.2:
                # 越低罚得越重: 0.1 -> -10, 0.0 -> -20
                energy_penalty -= (0.2 - bat) * 10.0  # 50

            # 严重惩罚耗尽 (Crashed)
            #if uav.is_crashed:
            #    energy_penalty -= 40.0
        #覆盖率奖励
        covered_count = sum(1 for t in world.terminals if t.is_covered)
        coverage_reward = (covered_count / world.args.num_terminals) * 5.0   # 50

        # 4.6 + 2.5 - 0.5 * ？ - 2 * 2，

        reward = hard_penalty + completion_reward + energy_penalty + coverage_reward

        return reward'''


    '''def get_reward(self, world):
        reward = 0

        # 1. 任务完成奖励 (加个基准线，把它变成压力)
        # 修改：引入基准线。如果覆盖率低于 30%，即使活着也是负分。
        # 逻辑：强迫智能体必须“努力工作”才能把分数变成正的。
        completion_rate = world.task_completion_rate[-1] if hasattr(world, 'task_completion_rate') else 0

        # 设定一个“及格线”，比如 0.2 (20%)
        # 如果 rate 是 0.1，奖励是 (0.1 - 0.2) * 10 = -1.0 (惩罚)
        # 如果 rate 是 0.8，奖励是 (0.8 - 0.2) * 10 = +6.0 (奖励)
        reward += (completion_rate - 0.6) * 10.0

        # 2. 碰撞惩罚 (改为平滑的梯度惩罚)
        # 你的原来逻辑是撞上才扣分(-5)。
        # 现在：进入危险区域就开始扣分，越近扣越多。
        collision_radius = world.args.collision_radius
        sense_radius = collision_radius * 2.5  # 警戒范围

        for i in range(world.args.num_uavs):
            for j in range(i + 1, world.args.num_uavs):
                dist = world.uavs[i].location.distance_to(world.uavs[j].location)

                if dist < collision_radius:
                    reward -= 0.5  # 撞上了，重罚
                elif dist < sense_radius:
                    # 没撞上，但在警戒范围内，根据距离给一个 0 到 -2 的惩罚
                    # 距离越近，惩罚越大。这就给了网络一个梯度方向：离远点！
                    penalty_factor = (sense_radius - dist) / (sense_radius - collision_radius)
                    reward -= penalty_factor * 0.5

        # 3. 能量惩罚 (保持原样或微调)
        for uav in world.uavs:
            bat = uav.get_battery_ratio()
            if bat < 0.15:
                reward -= (0.15 - bat) * 5.0  # 稍微降低权重，前期先学任务

        # ==========================================
        # 6. 时间/步数惩罚 (最关键的一点)
        # ==========================================
        # 每走一步都扣一点点分。
        # 这告诉智能体：“尽快完成覆盖，不要磨蹭，不要在低覆盖率状态下混日子”
        reward -= 0.1

        return reward'''


    '''def get_reward(self, world):
        reward = 0

        # 任务奖励
        # 原代码: reward += (completion_rate - 0.6) * 10.0
        # 修改为: 只要有完成就是正奖励，完成率越高奖励指数级上升
        completion_rate = world.task_completion_rate[-1] if hasattr(world, 'task_completion_rate') else 0
        # 线性奖励，鼓励任何微小的进步
        reward += completion_rate * 20.0

        # 增加距离引导奖励 (Heuristic Reward)
        # 这对于训练初期至关重要，告诉UAV“靠近簇中心是好的”
        for uav in world.uavs:
            if uav.role == 1:  # 服务角色
                # 找到最近的簇中心
                min_dist = float('inf')
                # 假设 world 里保存了 cluster_centers，或者从 terminal 获取
                # 这里简化逻辑：找到最近的有任务的终端
                target_terminals = [t for t in world.terminals if t.has_task]
                if target_terminals:
                    dists = [uav.location.distance_to(t.location) for t in target_terminals]
                    min_dist = min(dists)
                    # 距离越近，惩罚越小（或者是正奖励）
                    # 归一化距离奖励: 越近分越高
                    reward += (1.0 - min(min_dist, 500.0) / 500.0) * 0.5

        # 碰撞和出界
        # 增加出界惩罚
        for uav in world.uavs:
            if uav.location.x <= 0 or uav.location.x >= world.args.area_length or \
                    uav.location.y <= 0 or uav.location.y >= world.args.area_width:
                reward -= 5.0  # 强惩罚

        # 时间步惩罚 (保持)
        reward -= 0.1

        return reward'''

    def get_reward(self, world):
        reward = 0

        # 任务完成奖励
        completion_rate = world.task_completion_rate[-1] if world.task_completion_rate else 0
        reward += completion_rate * 20.0  # 鼓励完成任务 15-20-15-20

        #energy = world.average_energy[-1] if world.average_energy else 0
        #reward -= energy * 0.001 * 2   # 0.2-1.2，乘以权重5后1.0-6.0，  5-2

        # 公共惩罚：碰撞与出界
        # 服务奖励

        for uav in world.uavs:
            # 出界惩罚
            if uav.location.x <= 0 or uav.location.x >= world.args.area_length or \
                    uav.location.y <= 0 or uav.location.y >= world.args.area_width:
                reward -= 4.0  # 稍微降低，防止不敢动   4.0-3.0

            # 碰撞惩罚
            for other in world.uavs:
                if uav.uav_id != other.uav_id:
                    dist = uav.location.distance_to(other.location)
                    if dist < world.args.collision_radius:
                        reward -= 2.0    #  2.0-1.5

            # 务必加回距离引导奖励！这是破局关键
            if uav.role == 1:  # 服务角色
                target_terminals = [t for t in world.terminals if t.has_task]
                if target_terminals:
                    # 寻找最近的有任务的终端距离
                    dists = [uav.location.distance_to(t.location) for t in target_terminals]
                    min_dist = np.mean(dists)
                    # 距离越近，给一个小的正向引导（最大给 2.0 分）
                    guidance = (1.0 - min(min_dist, 1000.0) / 1000.0) * 2.0
                    #reward += guidance

            '''if uav.role == 1:  # 服务角色
                # 距离引导奖励：靠近有任务的终端
                target_terminals = [t for t in world.terminals if t.has_task]
                if target_terminals:
                    dists = [uav.location.distance_to(t.location) for t in target_terminals]
                    min_dist = np.mean(dists)  # min
                    # reward += (1.0 - min(min_dist, 400.0) / 400.0) * 0.5
                    reward += (1.0 - min(min_dist, 1000.0) / 1000.0) * 3  # 0.5-4-2-3
                if target_terminals:
                    # 1. 计算所有距离并排序
                    dists = [uav.location.distance_to(t.location) for t in target_terminals]
                    dists.sort()

                    # 2. 取最近的 K 个
                    # 如果总任务数少于 K，就取全部
                    K = min(len(dists), 8)
                    avg_min_dist = sum(dists[:K]) / K

                    # 3. 归一化引导奖励
                    # 分母改为 1000 (你的地图边长)
                    # 这样即便无人机在地图对角线，也能感受到微弱的引力
                    # 权重保持 5.0，作为“寻找任务”的动力
                    guidance = (1.0 - min(avg_min_dist, 1000.0) / 1000.0) * 5.0
                    reward += guidance'''


        # 3. 时间步惩罚 (防止磨蹭)
        reward -= 0.1
        return reward


def main():
    args = get_args()
    args = get_UAV_args(args)
    args = get_user_args(args)
    args = get_Charge_args(args)
    scenario = Scenario()
    world = scenario.make_world(args)
    print(world)


if __name__ == '__main__':
    main()