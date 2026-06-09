from sac_environment import OffloadEnvironment
from sac_model import *
import matplotlib.pyplot as plt


def SACScheduling_algorithm():
    # ==================== 初始化环境参数 ====================
    NUM_BSs = 10             # 基站/边缘服务器数量
    NUM_TASKS_max = 100      # 每个时间槽最大任务数
    BIT_RANGE = [10, 40]     # 任务数据量范围 [10, 40] Mbits
    NUM_TIME_SLOTS = 100     # 总时间槽数量
    ES_capacity_max = 50     # ES 最大计算能力（GHz）
    np.random.seed(2)
    ES_capacity = np.random.randint(10, ES_capacity_max + 1, size=[NUM_BSs])

    # ==================== 初始化 DRL 模型参数 ====================
    actor_lr = 1e-4          # Actor 网络学习率
    critic_lr = 1e-3         # Critic 网络学习率
    alpha = 0.01             # 熵正则化温度系数初始值
    alpha_lr = 1e-3          # 温度系数学习率
    episodes = 100           # 训练总轮数
    hidden_dim = 128         # 隐藏层神经元数
    gamma = 0.9              # 折扣因子
    tau = 0.005              # 目标网络软更新系数
    train_buffer_size = 10000  # 经验池容量
    batch_size = 64          # 训练批量大小
    target_entropy = -1      # 目标熵值
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ==================== 创建环境、智能体和经验池 ====================
    env = OffloadEnvironment(NUM_TASKS_max, BIT_RANGE, NUM_BSs, NUM_TIME_SLOTS, ES_capacity)
    agent = SAC(env.state_dim, hidden_dim, env.action_dim, actor_lr, critic_lr, alpha, alpha_lr,
                target_entropy, tau, gamma, device)
    train_buffer = ReplayBuffer(train_buffer_size)

    # ======== 基于 SAC 的在线任务调度与并行网络训练 ========
    average_delays = []  # 记录每轮平均延迟
    exe_count = 0        # 任务执行计数
    train_count = 0      # 训练次数计数
    for i_episode in range(episodes):
        # 生成本轮到达任务
        arrival_tasks = []
        for i in range(env.time_slots):
            task_dim = np.random.randint(1, env.n_tasks + 1)
            arrival_tasks.append(np.random.uniform(env.min_bit, env.max_bit, size=[task_dim]))

        env.reset_env(arrival_tasks)  # 重置环境
        episode_delays = []  # 本轮所有任务延迟
        for t in range(env.time_slots - 1):
            # ===== 在线任务调度 =====
            task_set_len = len(env.tasks_bit[t])
            for n in range(task_set_len):
                # 构建状态：[任务大小, 计算需求, 各ES队列长度]
                state = np.hstack([env.tasks_bit[t][n],
                                   env.comp_density[n] * env.tasks_bit[t][n],
                                   env.proc_queue_len[t]])
                action = agent.take_action(state)  # 按策略概率采样动作
                next_state, reward, delay = env.step_(t, n, action)  # 执行卸载
                train_buffer.add(state, action, reward, next_state)  # 存入经验池
                episode_delays.append(delay)
                exe_count = exe_count + 1
            env.update_proc_queues(t)  # 更新队列

            # ===== 网络训练 =====
            if train_buffer.size() > 300:  # 经验池达到最小要求后开始训练
                b_s, b_a, b_r, b_ns = train_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r}
                agent.update(transition_dict)  # 更新网络参数
                train_count = train_count + 1

        average_delays.append(np.mean(episode_delays))
        print({'Episode': '%d' % (i_episode+1), 'average delay': '%.4f' % average_delays[-1]})

    print('============ 所有任务卸载和模型训练完成（SAC 方法）==========')

    # ===== 保存和绘制结果 =====
    episodes_list = list(range(len(average_delays)))
    np.savetxt('../results/AveDelay_sac_BS' + str(NUM_BSs) +
               '_tasks' + str(NUM_TASKS_max) +
               '_f' + str(ES_capacity_max) +
               '_episode' + str(episodes) + '.csv', average_delays, delimiter=',', fmt='%.4f')
    plt.figure(1)
    plt.plot(episodes_list, average_delays)
    plt.ylabel('Average make-span')
    plt.xlabel('Episode')
    plt.savefig('../results/AveDelay_sac_BS' + str(NUM_BSs) +
                '_tasks' + str(NUM_TASKS_max) +
                '_f' + str(ES_capacity_max) +
                '_episode' + str(episodes) + '.png')
    plt.close()


if __name__ == '__main__':
    SACScheduling_algorithm()
