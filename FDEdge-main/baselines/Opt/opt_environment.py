import numpy as np
import collections
from scipy import stats


class OffloadEnvironment:
    """
    最优 (Opt) 方法的任务卸载环境
    通过穷举所有可能的 ES 选择，为每个任务找到延迟最小的最优分配方案
    作为性能上界参考（理想情况，实际中不可行）
    """

    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # ===== 基本参数 =====
        self.n_tasks = num_tasks       # 每个时间槽最大任务到达数量
        self.n_BSs = num_BSs           # 基站/边缘服务器数量
        self.time_slots = time_slots_  # 总时间槽数量
        self.duration = 1              # 时间槽时长（秒）
        self.ES_capacities = es_capacities  # 各 ES 计算能力（GHz）

        # 各 ES 传输速率
        np.random.seed(5)
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])

        # 各任务计算密度
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])

        # ===== 任务数据 =====
        self.tasks_bit = []
        self.min_bit = bit_range[0]
        self.max_bit = bit_range[1]

        # ===== 队列状态 =====
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])

    def reset(self, tasks_bit):
        """重置环境状态"""
        self.tasks_bit = tasks_bit
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])

    def step(self, t, b, n):
        """
        按基站分组模式的最优调度
        穷举所有 ES，选择总延迟最小的 ES 作为最优动作
        """
        opt_action = []
        min_service_delay = 1000  # 初始化为一个较大值
        for i in range(self.n_BSs):
            action = i
            # 计算排队等待延迟
            wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]

            if action == b:
                # 本地处理：无传输延迟
                tran_comp_delays = (self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])
            else:
                # 远程处理：传输延迟 + 计算延迟
                tran_comp_delays = (self.tasks_bit[t][b][n] / self.tran_rate_BSs[action] +
                                    self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])

            service_delay = tran_comp_delays + wait_delay
            # 记录延迟最小的 ES
            if service_delay < min_service_delay:
                min_service_delay = service_delay
                opt_action = action

        # 更新最优 ES 的累积负载
        self.proc_queue_bef[t][opt_action] = (self.proc_queue_bef[t][opt_action] +
                                              self.comp_density[n] * self.tasks_bit[t][b][n])

        return min_service_delay

    def step_(self, t, n):
        """
        简化模式的最优调度
        穷举所有 ES，选择总延迟最小的 ES
        """
        opt_action = []
        min_service_delay = 1000
        for i in range(self.n_BSs):
            action = i
            wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]
            # 传输延迟 + 计算延迟
            tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                                self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])

            service_delay = tran_comp_delays + wait_delay
            if service_delay < min_service_delay:
                min_service_delay = service_delay
                opt_action = action

        # 更新最优 ES 的累积负载
        self.proc_queue_bef[t][opt_action] = (self.proc_queue_bef[t][opt_action] +
                                              self.comp_density[n] * self.tasks_bit[t][n])

        return min_service_delay

    def update_proc_queues(self, t):
        """更新所有 ES 的处理队列到下一时间槽"""
        for b in range(self.n_BSs):
            self.proc_queue_len[t + 1][b] = np.max(
                [self.proc_queue_len[t][b] + self.proc_queue_bef[t][b] - self.ES_capacities[b] * self.duration, 0])
