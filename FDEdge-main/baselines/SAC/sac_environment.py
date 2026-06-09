import numpy as np
import collections

import torch
from scipy import stats


class OffloadEnvironment:
    """
    SAC 基线的任务卸载环境
    与 DQN 环境结构一致，支持按基站分组 (step) 和简化 (step_) 两种模式
    """

    def __init__(self, num_tasks, bit_range, num_BSs, time_slots_, es_capacities):
        # ===== 基本参数 =====
        self.n_tasks = num_tasks       # 每个时间槽最大任务到达数量
        self.n_BSs = num_BSs           # 基站/边缘服务器数量
        self.time_slots = time_slots_  # 总时间槽数量
        self.state_dim = 2 + self.n_BSs  # 状态维度
        self.action_dim = num_BSs      # 动作空间维度
        self.duration = 1              # 时间槽时长（秒）
        self.ES_capacities = es_capacities  # 各 ES 计算能力（GHz）

        # 各 ES 传输速率，范围 [400, 500] Mbits/s
        np.random.seed(5)
        self.tran_rate_BSs = np.random.randint(400, 501, size=[self.n_BSs])

        # 各任务计算密度，范围 [0.1024, 0.3072] Gigacycles/Mbit
        np.random.seed(1)
        self.comp_density = np.random.uniform(0.1024, 0.3072, size=[self.n_tasks])

        # ===== 任务数据 =====
        self.tasks_bit = []
        self.min_bit = bit_range[0]
        self.max_bit = bit_range[1]

        # ===== 队列状态 =====
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])  # 各 ES 队列负载
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])  # 本时间槽内累积负载
        self.wait_delay = 0

    def reset_env(self, tasks_bit):
        """重置环境状态"""
        self.tasks_bit = tasks_bit
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.n_BSs])
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.n_BSs])
        self.wait_delay = 0

    def step(self, t, b, n, action):
        """
        按基站分组模式的任务卸载
        本地处理（action == b）无传输延迟，远程处理需要传输延迟
        """
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]

        if action == b:
            # 本地处理：只有计算延迟
            tran_comp_delays = (self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])
        else:
            # 远程处理：传输延迟 + 计算延迟
            tran_comp_delays = (self.tasks_bit[t][b][n] / self.tran_rate_BSs[action] +
                                self.comp_density[n] * self.tasks_bit[t][b][n] / self.ES_capacities[action])

        delay = tran_comp_delays + self.wait_delay
        reward = - delay

        # 更新累积负载
        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][b][n]

        # 获取下一状态
        if n == len(self.tasks_bit[t][b]) - 1:
            if b == (self.n_BSs - 1):
                # 最后一个基站的最后一个任务 → 下一时间槽
                next_state = np.hstack([self.tasks_bit[t + 1][b][0],
                                        self.comp_density[0] * self.tasks_bit[t + 1][b][0],
                                        self.proc_queue_len[t+1]])
            else:
                # 当前时间槽下一个基站的第一个任务
                next_state = np.hstack([self.tasks_bit[t][b+1][0],
                                        self.comp_density[0] * self.tasks_bit[t][b+1][0],
                                        self.proc_queue_len[t]])
        else:
            # 当前基站的下一个任务
            next_state = np.hstack([self.tasks_bit[t][b][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][b][n + 1],
                                    self.proc_queue_len[t]])

        return next_state, reward, delay

    def step_(self, t, n, action):
        """简化模式的任务卸载，统一计算传输延迟"""
        self.wait_delay = (self.proc_queue_len[t][action] + self.proc_queue_bef[t][action]) / self.ES_capacities[action]

        # 传输延迟 + 计算延迟
        tran_comp_delays = (self.tasks_bit[t][n] / self.tran_rate_BSs[action] +
                            self.comp_density[n] * self.tasks_bit[t][n] / self.ES_capacities[action])

        delay = tran_comp_delays + self.wait_delay
        reward = - delay

        self.proc_queue_bef[t][action] = self.proc_queue_bef[t][action] + self.comp_density[n] * self.tasks_bit[t][n]

        # 获取下一状态
        if n == len(self.tasks_bit[t]) - 1:
            next_state = np.hstack([self.tasks_bit[t + 1][0],
                                    self.comp_density[0] * self.tasks_bit[t + 1][0],
                                    self.proc_queue_len[t+1]])
        else:
            next_state = np.hstack([self.tasks_bit[t][n + 1],
                                    self.comp_density[n + 1] * self.tasks_bit[t][n + 1],
                                    self.proc_queue_len[t]])

        return next_state, reward, delay

    def update_proc_queues(self, t):
        """更新所有 ES 的处理队列到下一时间槽"""
        for b in range(self.n_BSs):
            self.proc_queue_len[t + 1][b] = np.max(
                [self.proc_queue_len[t][b] + self.proc_queue_bef[t][b] - self.ES_capacities[b] * self.duration, 0])
