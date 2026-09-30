import numpy as np


class TaskDistribution:
    """
    任务生成分布模型
    根据当前时间槽动态计算任务生成数
    """

    def __init__(self, args):
        self.args = args
        self.mode = args.task_generation_mode

        # 分布参数
        self.period = args.task_period  # 周期（大时间步）
        self.min_tasks = args.min_tasks_per_large_step
        self.max_tasks = args.max_tasks_per_large_step

        # 随机游走参数
        self.current_tasks = (self.min_tasks + self.max_tasks) // 2
        self.step_size = (self.max_tasks - self.min_tasks) // 10

        # ================= 新增：马尔可夫阵发参数 =================
        # 状态：0 = 平稳期 (Low), 1 = 高峰期 (Burst)
        self.markov_state = 0

        # 状态转移概率
        # P(平稳->平稳)=0.85, P(平稳->高峰)=0.15 (突发高峰)
        # P(高峰->高峰)=0.70, P(高峰->平稳)=0.30 (高峰持续几步后结束)
        self.trans_prob_0_to_1 = 0.15  # 0.15-0.25-0.15
        self.trans_prob_1_to_0 = 0.25  # 0.30-0.45-0.25
        # ==========================================================

        # 新增：用于平滑的真实底层任务数
        self.smoothed_tasks = float((self.min_tasks + self.max_tasks) / 2)

    def get_task_count(self, large_step):
        """
        根据当前大时间步计算任务生成数
        """
        if self.mode == 'sinusoidal':
            phase = (large_step % self.period) / self.period * 2 * np.pi - np.pi / 2
            tasks = (self.min_tasks + self.max_tasks) / 2 + \
                    (self.max_tasks - self.min_tasks) / 2 * np.sin(phase)
            return int(tasks)

        elif self.mode == 'random_walk':
            change = np.random.randint(-self.step_size, self.step_size + 1)
            self.current_tasks = np.clip(
                self.current_tasks + self.step_size,  # 原代码有小bug，这里修正了
                self.min_tasks,
                self.max_tasks
            )
            return self.current_tasks

        elif self.mode == 'random':
            return np.random.randint(self.min_tasks, self.max_tasks + 1)

        elif self.mode == 'step_function':
            if (large_step // self.period) % 2 == 0:
                return self.max_tasks
            else:
                return self.min_tasks

        # ================= 新增：马尔可夫阵发模式 =================
        elif self.mode == 'markov_burst':
            # 1. 状态转移
            rand_val = np.random.random()
            if self.markov_state == 0:  # 当前是平稳期
                if rand_val < self.trans_prob_0_to_1:
                    self.markov_state = 1  # 爆发！
            else:  # 当前是高峰期
                if rand_val < self.trans_prob_1_to_0:
                    self.markov_state = 0  # 结束爆发

            '''# 2. 根据状态生成任务数
            mid_point = self.min_tasks + (self.max_tasks - self.min_tasks) // 3  
            if self.markov_state == 0:
                # 平稳期：生成较少的任务 (min 到 1/3 处)
                return np.random.randint(self.min_tasks, mid_point + 1)  # 3-9
            else:
                # 高峰期：生成极端的任务 (接近 max)
                return np.random.randint(self.max_tasks - 4, self.max_tasks + 1)  # 18-23'''
            # 2. 确定当前状态的“目标引力点”
            if self.markov_state == 0:
                # 平稳期目标：3 ~ 6
                target_tasks = np.random.randint(self.min_tasks, self.min_tasks + 4)
            else:
                # 高峰期目标：18 ~ 22
                target_tasks = np.random.randint(self.max_tasks - 4, self.max_tasks + 1)

            # 3. EMA 平滑 (核心秘诀：让曲线变柔和，呈现趋势)
            # alpha 越小，曲线越平滑，LSTM 越容易抓到趋势
            alpha = 0.4
            self.smoothed_tasks = alpha * target_tasks + (1 - alpha) * self.smoothed_tasks

            # 最终返回四舍五入的整数
            return int(np.clip(self.smoothed_tasks, self.min_tasks, self.max_tasks))
        # ==========================================================

        else:  # 'uniform'
            return (self.min_tasks + self.max_tasks) // 2