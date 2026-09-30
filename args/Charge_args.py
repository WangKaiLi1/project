#充电站参数
def get_Charge_args(args):
    # 充电站位置（建筑物顶部，可以设置多个）
    args.charger_locations = [
        [250, 250, 50],  # 充电站1
        [250, 750, 50],  # 充电站2
        [750, 750, 50],  # 充电站3
        [750, 250, 50],  # 充电站4
    ]
    args.charge_height = 50   # m 建筑物顶部高度
    args.laser_power = 12000  # W 激光发射功率
    args.energy_efficiency = 0.9  # UAV energy conversion factor 能量转换效率
    args.receiver_efficiency = 0.2  # receiver optical efficiency 光效率
    args.area = 0.01  # m^2 接收面积
    args.size = 0.1  # m size of the emitted laser beam 发射激光束尺寸
    args.degree = 3.4e-5  # angular spreading degree 角度扩散系数
    args.attenuation_factor = 1e-6  # /m attenuation factor 衰减因子
    args.charge_radius = 100   # m 充电半径
    return args
