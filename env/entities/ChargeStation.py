import numpy as np
import math
from env.entities.core import Location


class ChargeStation(object):
    def __init__(self, station_id, args, location):
        self.station_id = station_id
        self.args = args
        self.location = Location(location[0], location[1], location[2], args)  # [x, y, z]
        self.charge_height = args.charge_height

        # 激光充电参数
        self.laser_power = args.laser_power  # W
        self.beam_size = args.size  # m
        self.angular_spread = args.degree
        self.attenuation_factor = args.attenuation_factor
        self.receiver_area = args.area  # m²
        self.receiver_efficiency = args.receiver_efficiency
        self.energy_efficiency = args.energy_efficiency
        self.radius = args.charge_radius  # m

    def get_charge_height(self):
        return self.charge_height

    def get_radius(self):
        return self.radius

    #def get_position(self):
    #    return [self.location[0], self.location[1], self.location[2]]

    def can_charge(self, uav_location):
        """判断无人机是否在充电范围内"""
        distance = self.compute_distance(uav_location)
        return distance <= self.radius

    def compute_received_power(self, uav_location):
        """计算无人机接收到的充电功率"""
        distance = self.compute_distance(uav_location)
        if distance > self.radius:
            return 0.0

        # 激光束扩展半径
        beam_radius = self.beam_size + self.angular_spread * distance
        # 光强衰减
        attenuation = np.exp(-self.attenuation_factor * distance)
        # 激光功率密度
        intensity = (self.laser_power * attenuation) / (beam_radius ** 2)
        # 接收功率
        received_power = intensity * self.receiver_area * self.receiver_efficiency * self.energy_efficiency
        return received_power

    def compute_distance(self, uav_location):
        """计算到无人机的距离"""
        return self.location.distance_to(uav_location)