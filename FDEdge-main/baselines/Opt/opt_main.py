from opt_environment import OffloadEnvironment
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ==================== 初始化环境参数 ====================
    NUM_BSs = 10             # 基站/边缘服务器数量
    NUM_TASKS_max = 100      # 每个时间槽最大任务数
    BIT_RANGE = [10, 40]     # 任务数据量范围 [10, 40] Mbits
    NUM_TIME_SLOTS = 100     # 总时间槽数量
    ES_capacity_max = 50     # ES 最大计算能力（GHz）
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])
    episodes = 100           # 实验轮数

    # 创建环境
    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)

    # ======== 最优任务调度（穷举法，作为性能上界）========
    average_delays = []  # 记录每轮平均延迟
    for i_episode in range(episodes):
        # 生成本轮到达任务
        arrival_tasks = []
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))

        env.reset(arrival_tasks)  # 重置环境
        episode_delays = []  # 本轮所有任务延迟
        for t in range(env.time_slots - 1):
            # ===== 最优调度：为每个任务穷举所有 ES，选延迟最小的 =====
            task_set_len = len(env.tasks_bit[t])
            for n in range(task_set_len):
                n_delay = env.step_(t, n)  # 自动选择最优 ES
                episode_delays.append(n_delay)
            env.update_proc_queues(t)  # 更新队列

        average_delays.append(np.mean(episode_delays))
        print({'Episode': '%d' % (i_episode + 1), 'average delay': '%.4f' % average_delays[-1]})

    print('============ 所有任务卸载完成（Opt 最优方法）==========')

    # ===== 保存和绘制结果 =====
    episodes_list = list(range(len(average_delays)))
    np.savetxt('../results/AveDelay_Opt_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average service delay')
    plt.xlabel('Episode')
    plt.savefig('../results/AveDelay_Opt_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_episode' + str(episodes) + '.png')
    plt.close()
