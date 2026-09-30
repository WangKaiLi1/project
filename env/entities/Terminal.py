import math
import random
import numpy as np
from env.entities.core import Location
from env.entities.core import Task
from common.convert import cover_MB_to_b


class Terminal(object):
    def __init__(self, terminal_id, args, initial_location, cluster_id=None):
        self.args = args
        self.terminal_id = terminal_id
        self.location = Location(initial_location[0], initial_location[1], 0, args)
        self.step_distance = args.terminal_step_distance
        self.trans_power = args.trans_power
        self.compute_resource = np.random.uniform(args.terminal_compute_min, args.terminal_compute_max)
        self.is_covered = False
        self.current_task = None
        self.task_counter = 0
        self.cluster_id = cluster_id  # 所属簇ID
        self.cluster_center = None  # 簇中心
        self.cluster_radius = args.cluster_radius  # 簇活动半径
        # 新增：标记当前小时间槽是否已生成任务
        self.has_task = False

        # 软约束参数
        self.attraction_threshold = args.cluster_radius * 1.1   # 开始吸引的距离
        self.max_attraction = 0.90  # 最大吸引力系数
        # 任务Task
        #self.task = None
        #self.generate_task()

    def move(self):
        """高斯随机游走 + 簇中心软约束"""
        # 1. 基础高斯随机游走
        dx = np.random.randn() * self.step_distance
        dy = np.random.randn() * self.step_distance

        # 2. 如果有簇中心，添加软约束
        if self.cluster_center is not None:
            distance_to_center = self.get_distance()

            # 只有超过阈值才施加吸引力
            if distance_to_center > self.attraction_threshold:
                # 计算吸引力强度（距离越远，吸引力越强）
                excess_distance = distance_to_center - self.attraction_threshold
                attraction_strength = min(
                    excess_distance / self.attraction_threshold,
                    self.max_attraction
                )

                # 计算向簇中心的方向
                angle_to_center = math.atan2(
                    self.cluster_center[1] - self.location.y,
                    self.cluster_center[0] - self.location.x
                )

                # 添加向心分量
                dx += attraction_strength * self.step_distance * math.cos(angle_to_center)
                dy += attraction_strength * self.step_distance * math.sin(angle_to_center)

        # 3. 更新位置（只限制区域边界）
        new_x = self.location.x + dx
        new_y = self.location.y + dy

        new_x = np.clip(new_x, 0, self.args.area_length)
        new_y = np.clip(new_y, 0, self.args.area_width)

        self.location.set_position(x=new_x, y=new_y, z=0)

    def get_distance(self):
        """获取到簇中心的距离（用于统计）"""
        if self.cluster_center is None:
            return None
        return math.sqrt(
            (self.location.x - self.cluster_center[0]) ** 2 +
            (self.location.y - self.cluster_center[1]) ** 2
        )

    def generate_task(self):
        """生成新任务"""
        # 将任务数据大小四舍五入到小数点后两位
        data_size = round(np.random.uniform(self.args.task_minimum_data_size, self.args.task_maximum_data_size), 2)
        cycles_per_bit = np.random.randint(self.args.cycles_per_bit_min, self.args.cycles_per_bit_max)
        delay_threshold = round(np.random.uniform(self.args.task_delay_min, self.args.task_delay_max), 3)

        self.current_task = Task(self.task_counter, data_size, cycles_per_bit, delay_threshold)
        self.task_counter += 1
        self.has_task = True
        # task_index = 0 if self.task is None else self.task.task_index + 1
        # 创建任务
        # self.task = Task(task_index, task_bit, cycles_per_bit, threshold)  # 任务Task
        # 本地计算的时延和能耗
        self.current_task.local_delay, self.current_task.local_energy = self.compute_local_execution()
        self.current_task.completed_delay = self.current_task.local_delay

    def compute_local_execution(self):
        """计算本地执行的延迟和能耗"""
        if self.current_task is None:
            return 0, 0

        total_bits = cover_MB_to_b(self.current_task.data_size)
        total_cycles = total_bits * self.current_task.cycles_per_bit
        delay = total_cycles / self.compute_resource
        energy = self.args.K * (self.compute_resource ** 3) * delay
        return round(delay, 3), energy

    #def get_position(self):
    #    return [self.location.x, self.location.y, self.location.z]
