from args.argument import get_args

class Location(object):
    """环境中节点的位置"""
    def __init__(self, x, y, z, args):
        self.args = args
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def get_position(self):
        return [self.x, self.y, self.z]

    def set_position(self, x=None, y=None, z=None):
        """更新位置，自动限制在区域内"""
        if x is not None:
            self._x = max(0, min(x, self.args.area_length))
        if y is not None:
            self._y = max(0, min(y, self.args.area_width))
        if z is not None:
            self._z = max(0, z)  # 高度不设上限

    def distance_to(self, other_location):
        """计算到另一个位置的3D距离"""
        import math
        return math.sqrt(
            (self._x - other_location.x) ** 2 +
            (self._y - other_location.y) ** 2 +
            (self._z - other_location.z) ** 2
        )

    def toString(self):
        return f"位置[x:{self._x:.1f}, y:{self._y:.1f}, z:{self._z:.1f}]"


# 任务类
class Task(object):
    def __init__(self, task_id, data_size, cycles_per_bit, delay_threshold):
        """
        任务初始化方法
        :param task_id: 任务号
        :param data_size: 任务的大小
        :param cycles_per_bit: 计算所需CPU转数
        :param delay_threshold: 最大能容忍延迟
        """
        self.task_id = task_id
        self.data_size = data_size  # MB
        self.cycles_per_bit = cycles_per_bit
        self.delay_threshold = delay_threshold  # s

        # 本地 传输 UAV上的能耗和时延
        self.local_delay = 0
        self.local_energy = 0
        self.trans_delay = 0
        self.trans_energy = 0
        self.uav_exe_delay = 0
        self.uav_exe_energy = 0
        self.completed_delay = 0
        self.offload_success = False


if __name__ == '__main__':
    args = get_args()
    location = Location(199, 199, 20, args)
    # flag = location.setx(101)
    # flag = location.setx(201)
    # flag = location.setx(99)
    location.set_position(101, 201, 99)
    print(location.get_position())
