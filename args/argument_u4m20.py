import argparse
import torch
from args.UAV_args import get_UAV_args
from args.Terminal_args import get_user_args
from args.Charge_args import get_Charge_args
from args.get_time import file_save_time
#from args.alg_args import get_HMA_args, get_HMAO_args, get_MADDPG_args, get_HDQN_args, get_h_random_ddpg_args, \get_h_dqn_random_args

def get_args():
    parser = argparse.ArgumentParser("MEC Environment for Multi-UAV System")

    # ========== 环境参数 ==========
    parser.add_argument('--num_uavs', type=int, default=4, help='UAV数量')
    parser.add_argument('--num_terminals', type=int, default=20, help='终端数量')
    parser.add_argument("--num_charge_stations", type=int, default=4)
    parser.add_argument('--area_length', type=float, default=1000, help='区域长度(m)')
    parser.add_argument('--area_width', type=float, default=1000, help='区域宽度(m)')
    parser.add_argument('--bandwidth', type=float, default=2e6, help='带宽(Hz)')
    parser.add_argument('--carrier_freq', type=float, default=5e7, help='载波频率(Hz)')
    parser.add_argument('--light_speed', type=float, default=3e8, help='光速(m/s)')
    #parser.add_argument('--noise_power', type=float, default=-70, help='噪声功率(dBm)')
    #parser.add_argument('--N0', type=float, default=-174, help='噪声功率谱密度(dBm/Hz)')
    parser.add_argument('--wa', type=float, default=9.61, help='LoS参数a')
    parser.add_argument('--wb', type=float, default=0.16, help='LoS参数b')
    parser.add_argument('--eta_LoS', type=float, default=1.6, help='LoS额外损耗(dB)')
    parser.add_argument('--eta_NLoS', type=float, default=23.0, help='NLoS额外损耗(dB)')

    #parser.add_argument("--obs_dim", type=int, default=92)
    #parser.add_argument("--state_dim", type=int, default=108)
    #parser.add_argument("--action_dim", type=int, default=24)
    #parser.add_argument("--upper_obs_dim", type=float, default=5)
    # ========== 任务生成相关参数 ==========
    parser.add_argument('--task_generation_mode', type=str, default='sinusoidal', choices=['sinusoidal', 'random_walk', 'step_function', 'random', 'uniform'], help='任务生成模式')
    parser.add_argument('--task_period', type=int, default=13, help='任务生成周期（大时间步）')
    parser.add_argument('--min_tasks_per_large_step', type=float, default=3, help='最小任务生成数')
    parser.add_argument('--max_tasks_per_large_step', type=float, default=18, help='最大任务生成数')
    parser.add_argument('--max_terminals_per_slot', type=float, default=18, help='每个时间槽的最大终端数量')
    parser.add_argument('--task_noise_std', type=float, default=1.0, help='任务生成噪声标准差')

    # ========== 分层算法参数 ==========
    parser.add_argument('--history_window_size', type=int, default=5, help='滑动窗口大小（记录前N个大时间步的任务数）')

    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=5000, help='训练episodes数')
    parser.add_argument('--test_episodes', type=int, default=100, help='测试episodes数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")

    #  DQN
    parser.add_argument('--lr_upper', type=float, default=5e-5, help='上层critic学习率')
    parser.add_argument('--upper_buffer_size', type=int, default=10000, help='上层经验池大小')
    parser.add_argument('--lr_role', type=float, default=5e-5, help='上层actor学习率')

    # TD3
    parser.add_argument('--lr_actor', type=float, default=1e-5, help='Actor学习率')
    parser.add_argument('--lr_critic', type=float, default=5e-5, help='Critic学习率')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.001, help='软更新系数')
    parser.add_argument('--buffer_size', type=int, default=150000, help='经验池容量')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='初始探索率')  # 0.95
    parser.add_argument('--epsilon_decay', type=float, default=0.998, help='探索率衰减')
    parser.add_argument('--epsilon_decay_steps', type=float, default=6000, help='探索率衰减步数')
    parser.add_argument('--epsilon_min', type=float, default=0.02, help='最小探索率')  # 0.1
    #parser.add_argument('--noise_scale', type=float, default=0.1, help='噪声尺度')
    #parser.add_argument('--noise_decay', type=float, default=0.999, help='噪声衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--policy_freq', type=int, default=3, help='更新频率')
    parser.add_argument('--num_directions', type=int, default=16, help='离散方向数')

    parser.add_argument('--use_per', type=bool, default=False, help='是否使用优先级经验回放')
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER优先级指数 (0=均匀, 1=完全优先级)')
    parser.add_argument('--per_beta', type=float, default=0.4, help='PER重要性采样初始值')
    parser.add_argument('--per_beta_increment', type=float, default=0.001, help='PER beta增长率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')   # acorm中使用了123， 42//123//0//3407//
    parser.add_argument('--use_cuda', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--save_interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--print_interval', type=int, default=5, help='打印间隔')
    parser.add_argument("--model_dir", type=str, default="./result/model/")
    #parser.add_argument("--result_dir", type=str, default="./result/")
    parser.add_argument("--log_dir", type=str, default="./logs/")

    args = parser.parse_args()

    args.upper_state_dim = args.history_window_size + 5 * args.num_uavs  # [x, y, z, battery, dist_to_station]
    args.upper_obs_dim = 1 + 3 * args.num_uavs + args.history_window_size
    args.obs_dim = 4 * args.num_uavs + args.max_terminals_per_slot * 3 + 4    # 8+80+4=92
    args.state_dim = args.num_uavs * 5 + args.num_terminals * 3   #8+60+8=76
    args.action_dim = 1 + 2 + args.max_terminals_per_slot  # role, vh, direction, offload_ratios...
    args.upper_action_dim = 1

    args.epsilon_decay = (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps


    # 读取各模块参数
    args = get_train_args(args)
    args = get_UAV_args(args)
    args = get_user_args(args)
    args = get_Charge_args(args)
    args = get_ACORM_args(args)
    args = get_qv_args(args)
    #args = get_env_args(args)

    return args


def get_train_args(args):
    """训练相关参数"""
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 双时间尺度设置
    args.large_timestep = 120  # 120秒（角色分配周期）
    args.small_timestep = 2  # 2秒（服务决策周期）
    args.steps_per_large = args.large_timestep // args.small_timestep  # 60步
    args.episode_length = 30  # 30
    # 最小训练的buffer大小
    args.min_batch_size = 1000

    return args

def get_ACORM_args(args):
    """ACORM算法参数配置"""
    args.episode_limit = args.episode_length  # 每个episode的最大步数

    # ========== ACORM原有参数 ==========
    # Agent Embedding
    args.agent_embedding_dim = 128
    args.agent_embedding_lr = 1e-3

    # Role Embedding (RECL)
    args.role_embedding_dim = 64
    args.use_ln = False
    args.cluster_num = 2
    args.recl_lr = 8e-4
    args.train_recl_freq = 2   # 200 - 3 - 2 - 1 - 3 - 2
    args.role_tau = 0.005
    args.multi_steps = 1

    # Attention
    args.att_dim = 128
    args.att_out_dim = 64
    args.n_heads = 4
    args.soft_temperature = 1.0
    args.state_embed_dim = 64

    # QMIX
    args.qmix_hidden_dim = 32
    args.hyper_hidden_dim = 64
    args.hyper_layers_num = 2
    args.rnn_hidden_dim = 64
    args.mlp_hidden_dim = 64

    # 角色解码器
    args.role_decoder_hidden_dim = 128

    # ========== 训练参数 ==========
    # 角色嵌入网络/QMIX/角色解码器的学习率
    args.lr = 5e-4
    args.qmix_lr = 1e-4
    args.decoder_lr = 1e-4
    args.upper_gamma = 0.99
    args.upper_tau = 0.005    # 0.005-0.001-0.003-0.001-0.005

    # 学习率调度器
    args.use_hard_update = False
    args.use_lr_decay = True
    args.lr_decay_steps = 500  # 500
    args.lr_decay_rate = 0.98

    args.use_grad_clip = True   # 无用
    args.grad_clip = 10.0   # 更新qmix时

    args.use_layer_norm = True   # 层归一化，原有的ACORM中是use_ln=False
    args.use_gumbel = True
    #args.gumbel_temperature = 1.0
    args.gumbel_temperature = 1.0  # 初始温度
    args.gumbel_temp_min = 0.5  # 最小温度
    args.gumbel_temp_decay = 0.9995  # 衰减率

    args.target_update_freq = 2

    # ========== 探索参数 ==========
    #args.upper_episode = 0.9
    #args.episode_decay = 0.999
    #args.episode_min = 0.01
    #args.action_noise_std = 0.2  # 初始噪声标准差
    #args.noise_decay = 0.9995  # 噪声衰减率
    #args.min_noise_std = 0.01  # 最小噪声

    # ========== Buffer参数 ==========
    args.upper_buffer_size = 2000   # 5000
    args.upper_batch_size = 32   # 32
    args.upper_min_buffer_size = 500  # 500

    # ========== 预训练参数 ==========
    args.agent_embed_pretrain_epochs = 120   # 120-100-120
    args.recl_pretrain_epochs = 100         # 100-80-100

    # ========== 其他参数 ==========
    args.add_last_action = True  # MEC环境不需要last_action
    args.add_agent_id = True
    args.future_steps = 3

    return args

def get_qv_args(args):
    args.use_rnn = True
    args.use_gpu = True
    args.use_double_q = True
    args.use_RMS = False

    return args

def get_env_args(args):
    """环境基本参数"""

    # 算法
    # args.alg = 'HMAO'
    # args.alg = 'HMA'
    # args.alg = 'MADDPG'
    # args.alg = 'HDQN'
    # args.alg='HRandomDDPG'
    # args.alg='HDQNRandom'
    # args.alg = 'Local'
    # if args.alg == 'HMAO':
    #     get_HMAO_args(args)
    # elif args.alg == 'HMA':
    #     args.option_length = 5 * 2
    #     get_HMA_args(args)
    # elif args.alg == 'MADDPG':
    #     get_MADDPG_args(args)
    # elif args.alg == 'HDQN':
    #     get_HDQN_args(args)
    # elif args.alg == 'HRandomDDPG':
    #     get_h_random_ddpg_args(args)
    # elif args.alg == 'HDQNRandom':
    #     get_h_dqn_random_args(args)
    # 训练时间
    # args.train_time = file_save_time()
    # args.base_path = './result/' + args.alg + '/train_time_' + args.train_time
    # 图片保存路径
    # args.plt_path = args.base_path + '/plt'
    # args.model_path = args.base_path + '/model'
    # args.log_path = args.base_path + '/log'
    # args.yaml_path = args.base_path + '/args/'

    return args



