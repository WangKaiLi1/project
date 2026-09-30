import math
import numpy as np
from env.entities import ChargeStation
from env.entities.core import Location


class UAV(object):
    def __init__(self, uav_id, args, initial_location):
        self.args = args
        self.uav_id = uav_id
        self.location = Location(initial_location[0], initial_location[1], args.height, args)
        self.role = 1
        self.battery = self.args.battery
        self.battery_capacity = self.args.battery
        self.battery_threshold = self.args.battery_threshold

        self.weight = self.args.weight
        self.g = self.args.g
        self.rho = self.args.Rho
        self.A = self.args.A
        self.CD0 = self.args.CD0
        self.sigma_A = self.args.blade_area_ratio * self.args.A
        self.Ut = self.args.Ut
        self.K = self.args.K
        self.max_v_horizontal = self.args.vh_max
        self.min_v_horizontal = self.args.vh_min
        self.max_v_vertical = self.args.vt_max
        self.min_v_vertical = self.args.vt_min
        self.compute_resource = np.random.uniform(args.uav_compute_min, args.uav_compute_max)
        self.coverage_radius = self.args.coverage_radius
        self.service_height = self.args.height
        self.charge_height = self.args.charge_height
        self.N0 = self.args.N0  # 噪声  dBm
        self.safety_margin = args.safety_margin  # 安全裕度
        self.is_crashed = False
        self.is_charging = False
        '''# 能量约束违反标志
        self.energy_constraint_violated = False
        self.charge_step = -1
        self.arrive_station_step = -1
        self.charged_battery_ratio = 0
        self.arrive_station = False'''

        # 充电状态
        self.charge_step_counter = 0  # 当前充电步数
        self.charge_total_steps = args.steps_per_large  # 总充电步数

        self.service_counter = 0

    def reset(self):
        self.is_charging = False
        '''self.energy_constraint_violated = False
        self.charge_step = -1
        self.arrive_station_step = -1
        self.charged_battery_ratio = 0
        self.arrive_station = False'''

    def get_battery_ratio(self):
        """获取电池电量比例"""
        return self.battery / self.battery_capacity

    def need_charge(self):
        """判断是否需要强制充电"""
        return self.get_battery_ratio() < self.args.battery_threshold

    def need_service(self):
        return self.get_battery_ratio() > self.args.service_threshold

    def set_role(self, role):
        self.role = role

    def find_nearest_charge_station(self, charge_stations):
        """找到最近的充电站"""
        if len(charge_stations) == 0:
            return None

        min_distance = float('inf')
        target_station = None
        for station in charge_stations:
            distance = station.compute_distance(self.location)
            if distance < min_distance:
                min_distance = distance
                target_station = station

        return target_station

    def charge_move(self, target_station):
        if self.is_crashed:
            return 0.0  # 坠毁了就不动，也不扣能耗

        """更新充电状态"""
        # 阶段1：移动到充电站
        if not self.is_charging:
            # 计算虚拟飞行能耗
            #arrival_energy = self.calculate_arrival_cost(target_station)
            distance = self.location.distance_to(target_station.location)
            arrival_energy = self.calculate_arrival_cost(distance)
            # 扣除能量
            self.consume_energy(arrival_energy)
            # 到达充电站
            self.location.set_position(
                x=target_station.location.x,
                y=target_station.location.y,
                z=self.args.charge_height
            )
            self.is_charging = True
            self.charge_step_counter = 0  # 重置计数器
            return arrival_energy
        # 阶段2：充电中
        else:
            self.charge_step_counter += 1
            power = target_station.compute_received_power(self.location)
            #print(f"充电功率{power}")
            self.charge(power)

            # 充电完成
            if self.charge_step_counter >= (self.charge_total_steps - 1):
                self.is_charging = False
                self.charge_step_counter = 0
                #print("充电完成")

            return 0.0

    def service_move(self, v, direction):
        if self.is_crashed:
            return 0.0  # 坠毁了就不动，也不扣能耗

        if self.service_counter == 0:
            self.service_counter += 1
            energy = self.calculate_arrival_cost(self.service_height - self.location.z)
            self.location.set_position(
                x=self.location.x,
                y=self.location.y,
                z=self.service_height
            )

            return energy
        else:
            #print(f"{v_horizontal} {v_vertical}")  #17.697708129882812 4.364423751831055
            v = np.clip(v, self.min_v_horizontal, self.max_v_horizontal)
            direction = np.clip(direction, 0, 2 * np.pi)
            # 获取新位置
            delta_x = v * np.cos(direction) * self.args.small_timestep
            delta_y = v * np.sin(direction) * self.args.small_timestep
            #print(f"{delta_x} {delta_y}")  # -37.3069953918457 15.453056335449219
                                            # 36.049957275390625 -14.932380676269531
            new_x = np.clip(self.location.x + delta_x, 0, self.args.area_width)
            new_y = np.clip(self.location.y + delta_y, 0, self.args.area_width)
            energy = self.compute_fly_energy(v, 0, self.args.small_timestep)

            self.location.set_position(x=new_x, y=new_y, z=self.service_height)
            self.consume_energy(energy)
            if self.service_counter >= self.args.steps_per_large or self.is_crashed:
                self.service_counter = 0

            return energy

    def calculate_arrival_cost(self, distance):
        """计算瞬时到达的虚拟能耗"""
        #distance = self.location.distance_to(target_station.location)
        avg_horizontal_speed = (self.max_v_horizontal + self.min_v_horizontal) / 2
        avg_vertical_speed = (self.max_v_vertical + self.min_v_vertical) / 2
        speed = math.sqrt(avg_horizontal_speed ** 2 + avg_vertical_speed ** 2)

        # 估算飞行时间 = 距离 / 平均速度
        estimated_time = distance / speed
        # 估算能耗 = 功率 * 时间
        fly_energy = self.compute_fly_energy(avg_horizontal_speed, avg_vertical_speed, estimated_time)

        return fly_energy

    def compute_fly_energy(self, v_horizontal, v_vertical, fly_time):
        # 水平飞行功率
        gravity = self.weight * self.g
        V_h = math.sqrt(gravity / (2 * self.rho * self.A))
        a = (gravity ** 2) / (math.sqrt(2) * self.rho * self.A)
        P_horizontal = a * (1 / math.sqrt(v_horizontal ** 2 + math.sqrt(v_horizontal ** 4 + 4 * V_h ** 4)))

        # 垂直飞行功率
        P_vertical = gravity * abs(v_vertical) / 0.7

        # 叶片阻力型线功率
        P_blade = 0.125 * self.CD0 * self.sigma_A * self.rho * (self.Ut ** 3)

        P_total = P_horizontal + P_vertical + P_blade
        energy = P_total * fly_time
        return energy

    def charge(self, received_power):
        """接收充电功率"""
        energy_gain = received_power * self.args.small_timestep
        self.battery = min(self.battery + energy_gain, self.battery_capacity)

    def compute_task(self, total_cycles, compute_per_user):
        """计算任务的延迟/能耗"""
        uav_exe_delay = total_cycles / compute_per_user
        uav_exe_delay = round(uav_exe_delay, 3)
        uav_exe_energy = self.args.K * (compute_per_user ** 3) * uav_exe_delay
        self.consume_energy(uav_exe_energy)
        return uav_exe_delay, uav_exe_energy

    def consume_energy(self, energy):
        """消耗能量"""
        self.battery -= energy
        if self.battery < 0:
            self.battery = 0
            self.is_crashed = True
            #print(f"UAV{self.uav_id}: 电量耗尽!!!")
        #return self.battery
