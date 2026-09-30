# Terminal的参数
# def get_user_args(args):
#     # 0.1-1 MB  对应本地计算时间 0.8 -8 s
#     args.step_distance = 10
#     args.trans_power = 0.1  # W 出自 PDDQNNLP   出自 Semi-Distributed
#     args.user_compute_resource = 1e8  # 计算资源 CPU cycles/s
#     args.task_minimum_data_size = 0.5  # MB
#     args.task_maximum_data_size = 1
#     args.cycles_per_bit = 100
#     args.random_threshold = False
#     if args.random_threshold:
#         args.min_threshold = 3  # s
#         args.max_threshold = 5  # s
#     else:
#         args.threshold = 4  # s
#     args.terminal_compute_coefficient = 3e-9
#     return args
def get_user_args(args):
    # 0.5-2 MB  对应本地计算时间 0.25 - 5 s
    args.terminal_step_distance = 2  # m 每小时间步移动距离
    args.trans_power = 0.2  # W 发射功率

    # 任务参数
    args.task_minimum_data_size = 0.5  # MB
    args.task_maximum_data_size = 2.0  # MB
    args.cycles_per_bit_min = 90  # CPU cycles/bit
    args.cycles_per_bit_max = 100
    # 延迟约束
    args.task_delay_min = 2.0  # s（500ms） 1.0 - 1.5 - 2.0 - 5.0
    args.task_delay_max = 3.5  # s（1.5s）  3.5 _ 7.5 - 7.0

    # 本地计算资源
    args.terminal_compute_min = 3e8  # 1 GHz -- 3e8
    args.terminal_compute_max = 5e8  # 2 GHz -- 5e8
    args.terminal_compute_coefficient = 1.0  # J/Cycle

    args.cluster_radius = 150
    return args

