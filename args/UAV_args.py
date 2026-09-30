# UAV的参数

# def get_UAV_args(args):
#     args.height = 150  # 飞行高度
#     args.battery = 800000  # 电池 J
#     args.A = 0.503  # Rotor disc area 平方米       出自 PDDQNNLP
#     args.Ut = 12  # Tip speed of the rotor blade m/s  这里两篇文献分别是12 和 120    出自 PDDQNNLP
#     args.v0 = 4  # Mean rotor induced velocity in hover m/s   出自 Semi-Distributed
#     args.s = 0.05  # Rotor solidity 无单位 出自 PDDQNNLP 出自 Semi-Distributed
#     args.F0 = 0.6  # Fuselage drag ratio 无单位  出自 PDDQNNLP   出自 Semi-Distributed
#     args.P1 = 80  # Blade profile power w       出自 Semi-Distributed
#     args.P2 = 90  # Induced power w             出自 Semi-Distributed
#     args.N0 = -70  # 噪声  dBm       dBm=10*log (W)
#
#     args.uav_compute_resource = 1e10  # 1e8 CPU cycles/s 出自 Semi-Distributed
#     args.weight = 9.65  # kg        出自 PDDQNNLP
#     args.Rho = 1.225  # air density kg/m3  出自 PDDQNNLP
#     # TODO  工作频率  飞行速度 CPU有关的计算系数
#     args.fc = 1e7  # TODO
#     args.vt = 20  # TODO
#     args.uav_compute_coefficient = 1.5e-8  # TODO
#
#     return args


def get_UAV_args(args):
    args.height = 100  # 飞行高度
    args.battery = 100000  # 电池 J
    args.battery_threshold = 0.27  # 强制充电阈值
    args.service_threshold = 0.85

    args.arrival_steps = 2
    #args.charge_step = args.steps_per_large - args.arrival_steps

    # 物理参数
    args.weight = 5  # kg
    args.g = 9.8
    args.A = 0.503  # m² the total area of the UAV rotor disks
    args.Rho = 1.225  # kg/m3 air density
    args.CD0 = 0.012  # the profile drag coefficient
    args.blade_area_ratio = 0.056  # *A=the total blade area
    args.Ut = 150  # m/s Tip speed of the rotor blade

    args.N0 = -70  # 噪声  -70dBm       dBm=10*log (W)
    args.K = 1e-28

    # 速度限制
    args.vh_max = 20   # 15-20
    args.vh_min = 5    # 3-5
    args.vt_max = 6
    args.vt_min = 1

    # 计算资源
    args.uav_compute_min = 20e9  # 10 GHz- 15 - 20 - 40 - 90
    args.uav_compute_max = 30e9  # 15 GHz- 18 - 25 - 50 - 100
    args.uav_compute_coefficient = 1.0  # J/Cycle

    args.coverage_radius = 250  # m
    args.collision_radius = 30

    args.safety_margin = 1.5

    return args
