import numpy as np


class OffloadEnvironment:
    """
    DQN 基线的任务卸载环境
    与 FDEdge 主环境类似，但支持两种 step 模式：
    - step(): 按基站分组的任务卸载（区分本地和远程传输）
    - step_(): 简化版任务卸载（统一计算传输延迟）
    """

    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # ===== 基本参数 =====
        self.n_tasks = num_tasks       # 每个时间槽最大任务到达数量
        self.n_BSs = num_BSs           # 基站/边缘服务器数量
        self.time_slots = time_slots_  # 总时间槽数量
        self.state_dim = 2 + self.n_BSs  # 状态维度 = 任务大小(1) + 计算需求(1) + 各ES队列长度(n_BSs)
        self.action_dim = num_BSs      # 动作空间维度 = ES 数量
        self.duration = 1              # 每个时间槽时长（秒）
        self.ES_capacities = es_capacities  # 各 ES 计算能力（GHz）

        # 各 ES 的传输速率，范围 [400, 500] Mbits/s
        np.random.seed(5)
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])

        # 各任务的计算密度，范围 [0.1024, 0.3072] Gigacycles/Mbit
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])

        # ===== 任务数据 =====
        self.tasks_bit = []             # 所有时间槽的任务数据量
        self.min_bit = bit_range[0]     # 任务最小数据量（Mbits）
        self.max_bit = bit_range[1]     # 任务最大数据量（Mbits）

        # ===== 队列状态 =====
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # 各 ES 队列负载（Gigacycles）
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # 本时间槽内已累积负载（Gigacycles）
        self.wait_delay = 0  # 当前任务排队等待延迟

    def reset_env(self, tasks_bit):
        """重置环境状态"""
        self.tasks_bit = tasks_bit
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])
        self.wait_delay = 0

    def step(self, t, b, n, action):
        """
        执行任务卸载（按基站分组模式）
        参数：
          - t: 当前时间槽
          - b: 任务所属的源基站
          - n: 任务在该基站中的序号
          - action: 选择处理任务的目标 ES
        返回：(下一状态, 奖励, 延迟)
        """
        # 计算排队等待延迟
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]

        if action == b:
            # 任务在本地基站处理，无传输延迟，只有计算延迟
            tran_comp_delays = (self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])
        else:
            # 任务卸载到其他基站，需要传输延迟 + 计算延迟
            tran_comp_delays = (self.tasks_bit[t][b][n] / self.tran_rate_BSs[action] +
                                self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])

        # 总服务延迟 = 传输计算延迟 + 排队等待延迟
        delay = tran_comp_delays + self.wait_delay
        reward = - delay  # 奖励 = 负延迟

        # 更新该 ES 本时间槽内的累积负载
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][b][n]

        # 获取下一个状态
        if n == len(self.tasks_bit[t][b]) - 1:
            # 当前基站的最后一个任务 → 看下一个时间槽或下一个基站
            next_state = np.hstack([self.tasks_bit[t + 1][b][0],
                                    self.comp_density[0] * self.tasks_bit[t + 1][b][0],
                                    self.proc_queue_len[t + 1]])
        else:
            # 当前基站的下一个任务
            next_state = np.hstack([self.tasks_bit[t][b][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][b][n + 1],
                                    self.proc_queue_len[t]])

        return next_state, reward, delay

    def step_(self, t, n, action):
        """
        执行任务卸载（简化模式，统一计算传输延迟）
        参数：
          - t: 当前时间槽
          - n: 任务序号
          - action: 选择处理任务的目标 ES
        返回：(下一状态, 奖励, 延迟)
        """
        # 计算排队等待延迟
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]

        # 传输延迟 + 计算延迟
        tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                            self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])

        # 总服务延迟
        delay = tran_comp_delays + self.wait_delay
        reward = - delay

        # 更新该 ES 本时间槽内的累积负载
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][n]

        # 获取下一个状态
        if n == len(self.tasks_bit[t]) - 1:
            # 当前时间槽最后一个任务 → 下一时间槽第一个任务
            next_state = np.hstack([self.tasks_bit[t + 1][0],
                                    self.comp_density[0] * self.tasks_bit[t + 1][0],
                                    self.proc_queue_len[t + 1]])
        else:
            # 当前时间槽下一个任务
            next_state = np.hstack([self.tasks_bit[t][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][n + 1],
                                    self.proc_queue_len[t]])

        return next_state, reward, delay

    def update_proc_queues(self, t):
        """
        更新所有 ES 的处理队列到下一时间槽
        公式：Q_{t+1} = max(Q_t + 新增负载 - ES处理能力 * 时间槽时长, 0)
        """
        for b in range(self.n_BSs):
            self.proc_queue_len[t + 1][b] = np.max(
                [self.proc_queue_len[t][b] + self.proc_queue_bef[t][b] - self.ES_capacities[b] * self.duration, 0])
