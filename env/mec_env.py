import numpy as np
from env.mec_scenery import Scenario


class MECEnv(object):
    """
    MEC 环境封装
    职责：
    1. 提供标准的 RL 接口（reset/step）
    2. 调用 Scenario 获取观测和奖励
    3. Episode 管理和统计
    """
    def __init__(self, args, world, reset_callback=None, observation_callback=None, reward_callback=None):
        self.args = args
        self.world = world
        #self.scenario = Scenario()
        #self.world = self.scenario.make_world(args)
        self.reset_callback = reset_callback
        self.observation_callback = observation_callback
        self.reward_callback = reward_callback
        self.num_uavs = args.num_uavs
        self.num_terminals = args.num_terminals

        self.uav_locations = [[uav.location.get_position() for uav in self.world.uavs]]
        #self.episode_rewards = []
        self.episode_steps = 0

    def reset(self):
        self.reset_callback(self.world, self.args)
        #self.episode_rewards = []
        #self.episode_steps = 0
        #obs = self.get_obs()
        #state = self.get_state()
        # 获取上层状态
        #upper_state = self.get_upper_state()
        # 记录info
        self.uav_locations = [[uav.location.get_position() for uav in self.world.uavs]]
        self.episode_steps = 0

        #return obs, state, upper_state

    def start_large_step(self):
        """
        开始新的大时间步
        """
        self.world.reset_for_large_step()
        self.world.current_large_step += 1
        # 获取上层状态
        upper_state = self.get_upper_state()
        upper_obs = self.get_upper_obs()

        return upper_state, upper_obs

    def start_small_step(self):
        obs = self.get_obs()
        state = self.get_state()

        return obs, state

    def step(self, actions):
        completed_rate = self.world.step(actions)
        reward = self.reward_callback(self.world)
        done = self.world.current_step >= self.args.steps_per_large
        if not done:  # 只有在未达到最大步数时才检查电量
            # 检查是否所有无人机都电量耗尽
            all_uavs_dead = all(uav.is_crashed == True for uav in self.world.uavs)
            if all_uavs_dead:
                done = True
                #print(f"所有无人机电量耗尽，提前结束")

        uav_location = []
        for uav in self.world.uavs:
            uav_location.append(uav.location.get_position())
        self.uav_locations.append(uav_location)
        infos = {
            'task_completion_rate': completed_rate,
            'current_step': self.world.current_step,
            'current_large_step': self.world.current_large_step,
            'target_tasks': self.world.current_large_step_tasks
        }

        # 1. 终端移动
        for terminal in self.world.terminals:
            terminal.move()
            # 重置标志
            terminal.is_covered = False
            terminal.has_task = False
            terminal.current_task = None
        # 2. 生成当前时间槽的新任务
        self.world.generate_tasks_for_slot()

        obs = self.get_obs()
        state = self.get_state()
        return obs, state, reward, done, infos

    def get_upper_state(self):
        """
        获取上层状态（用于角色选择）
        包含：滑动窗口 + 所有UAV状态
        """
        upper_state = []
        # 1. 滑动窗口特征
        task_window = self.world.get_task_history_features()  # [history_window_size]
        upper_state.extend(task_window)
        # 2. 所有UAV的状态---num_uavs * 5
        uav_states = []
        for uav in self.world.uavs:
            uav_state = [
                uav.location.x / self.args.area_length,
                uav.location.y / self.args.area_width,
                uav.location.z / self.args.height,
                uav.get_battery_ratio()
            ]

            # 到最近充电站的距离
            nearest_station = uav.find_nearest_charge_station(self.world.charge_stations)
            if nearest_station:
                dist = uav.location.distance_to(nearest_station.location)
                uav_state.append(dist / self.args.area_length)
            else:
                uav_state.append(1.0)

            uav_states.extend(uav_state)

        # 3. 拼接
        #upper_state = np.concatenate([task_window, uav_states])
        upper_state.extend(uav_states)

        #return upper_state.astype(np.float32)
        return np.array(upper_state, dtype=np.float32)

    def get_upper_obs(self):
        L, W, H = self.world.args.area_length, self.world.args.area_width, self.world.args.height
        obs = []
        uav_id_onehot = [0.0] * self.world.args.num_uavs
        # obs_list.extend(uav_id_onehot)  # 加入身份信息

        # 获取任务滑动窗口（包含过去几个大时间步的任务数及当前任务数）
        task_window = self.world.get_task_history_features()
        for uav in self.world.uavs:
            uav_obs = []
            uav_obs.extend(task_window)
            uav_id_onehot[uav.uav_id] = 1.0  # [1, 0] 或 [0, 1]
            uav_obs.extend(uav_id_onehot)
            #uav_obs.append(self.world.current_large_step_tasks)
            # 加入任务滑动窗口序列，而不是仅加入当前任务数
            uav_obs.extend([
                # uav.location.x / L,
                # uav.location.y / W,
                # uav.location.z / H,
                uav.role,
                uav.get_battery_ratio()
            ])

            # 到最近充电站的距离
            nearest_station = uav.find_nearest_charge_station(self.world.charge_stations)
            if nearest_station:
                dist = uav.location.distance_to(nearest_station.location)
                uav_obs.append(dist / self.args.area_length)
            else:
                uav_obs.append(1.0)

            for other_uav in self.world.uavs:
                if other_uav.uav_id != uav.uav_id:
                    # uav_obs.append(other_uav.get_battery_ratio())
                    uav_obs.extend([
                        other_uav.role,
                        other_uav.get_battery_ratio()
                    ])

            '''uav_obs.extend([
                uav.location.x / L,
                uav.location.y / W,
                uav.location.z / H,
                uav.get_battery_ratio()
            ])

            for other_uav in self.world.uavs:
                if other_uav.uav_id != uav.uav_id:
                    uav_obs.append(other_uav.get_battery_ratio())'''

            obs.append(uav_obs)

        return np.array(obs, dtype=np.float32)

    def get_obs(self):
        obs = []
        for uav in self.world.uavs:
            obs.append(self.observation_callback(uav, self.world))
        return np.array(obs, dtype=np.float32)

    # 这部分下层状态是否存入角色？而不是在动作中？那这样下层选择动作和训练部分就需要修改。
    def get_state(self):
        state = []
        L, W, H = self.args.area_length, self.args.area_width, self.args.height
        for uav in self.world.uavs:
            '''state.extend(uav.location.get_position())  # [x, y, z]
            state.append(uav.get_battery_ratio())  # 电量比例'''
            state.extend([
                uav.location.x / L,
                uav.location.y / W,
                uav.location.z / H,
                uav.get_battery_ratio(),
                uav.role
            ])

        # 2. 有任务的终端信息
        #max_terminals = self.world.args.max_terminals_per_slot
        terminal_obs = []
        for terminal in self.world.terminals:
            #terminal_obs.extend(terminal.location.get_position()[:2])
            terminal_obs.extend([
                terminal.location.x / L,
                terminal.location.y / W
            ])
            if terminal.has_task and terminal.current_task:
                terminal_obs.append(terminal.current_task.data_size)
                #terminal_obs.append(terminal.current_task.cycles_per_bit / 10.0)
                #terminal_obs.append(terminal.current_task.delay_threshold)
            else:
                terminal_obs.append(0)

        # 填充到固定长度
        max_terminal_obs = self.world.args.num_terminals * 3
        while len(terminal_obs) < max_terminal_obs:
            terminal_obs.append(0)

        state.extend(terminal_obs[:max_terminal_obs])

        return np.array(state, dtype=np.float32)

    def render(self):
        """渲染"""
        print(f"\n{'=' * 80}")
        print(f"时间步: {self.world.current_step} | "
              f"任务完成率: {self.world.record_statistics():.2%}")
        print(f"{'=' * 80}")

        for uav in self.world.uavs:
            print(f"UAV {uav.uav_id}: "
                  f"{uav.location.toString()}, "
                  f"电量{uav.get_battery_ratio():.2%}, "
                  f"角色{'充电' if uav.role == 0 else '服务'}")

