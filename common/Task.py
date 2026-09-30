import numpy as np


class Task(object):
    # 任务类型常量
    TASK_TYPE_DELAY_SENSITIVE = 0  # 延迟敏感型
    TASK_TYPE_COMPUTE_INTENSIVE = 1  # 计算密集型
    TASK_TYPE_BALANCED = 2  # 均衡型

    def __init__(self, task_id, data_size, cycles_per_bit, delay_threshold, task_type=None):
        self.task_id = task_id
        self.data_size = data_size
        self.cycles_per_bit = cycles_per_bit
        self.delay_threshold = delay_threshold
        self.task_type = task_type  # 任务类型

        # ... 原有代码 ...

    def generate_task(self):
        """根据任务类型生成任务"""
        # 随机选择任务类型
        task_type = np.random.choice([
            Task.TASK_TYPE_DELAY_SENSITIVE,
            Task.TASK_TYPE_COMPUTE_INTENSIVE,
            Task.TASK_TYPE_BALANCED
        ], p=[0.3, 0.3, 0.4])  # 可调整概率分布

        if task_type == Task.TASK_TYPE_DELAY_SENSITIVE:
            # 延迟敏感：小数据量、低计算需求、严格延迟
            data_size = round(np.random.uniform(0.5, 2.0), 2)
            cycles_per_bit = np.random.randint(500, 1000)
            delay_threshold = round(np.random.uniform(0.1, 0.3), 3)

        elif task_type == Task.TASK_TYPE_COMPUTE_INTENSIVE:
            # 计算密集：大数据量、高计算需求、宽松延迟
            data_size = round(np.random.uniform(3.0, 5.0), 2)
            cycles_per_bit = np.random.randint(1500, 2500)
            delay_threshold = round(np.random.uniform(0.8, 1.5), 3)

        else:  # TASK_TYPE_BALANCED
            # 均衡型：使用原有参数范围
            data_size = round(np.random.uniform(
                self.args.task_minimum_data_size,
                self.args.task_maximum_data_size
            ), 2)
            cycles_per_bit = np.random.randint(
                self.args.cycles_per_bit_min,
                self.args.cycles_per_bit_max
            )
            delay_threshold = round(np.random.uniform(
                self.args.task_delay_min,
                self.args.task_delay_max
            ), 3)

        self.current_task = Task(
            self.task_counter,
            data_size,
            cycles_per_bit,
            delay_threshold,
            task_type=task_type
        )
        self.task_counter += 1
        self.current_task.local_delay, self.current_task.local_energy = \
            self.compute_local_execution()