import random
import numpy as np
import collections
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """经验回放池，存储 (状态, 动作, 奖励, 下一状态) 元组"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 固定容量的先进先出队列

    def add(self, state, action, reward, next_state):
        """添加一条经验记录"""
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """随机采样 batch_size 条记录"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        """返回当前经验池大小"""
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """
    Q 网络：输入状态，输出各动作的 Q 值
    包含两个隐藏层，使用 ReLU 激活函数
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 输入层 → 隐藏层1
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)   # 隐藏层1 → 隐藏层2
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)   # 隐藏层2 → 输出层（各动作Q值）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """
    DQN (Deep Q-Network) 算法
    使用 epsilon-贪婪策略进行动作选择，通过目标网络稳定训练
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 e_greedy, target_update, device, e_greedy_increment=None):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)          # Q 网络（主网络）
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)   # 目标 Q 网络（延迟更新）
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # Adam 优化器
        self.gamma = gamma                # 折扣因子
        self.epsilon_max = e_greedy       # epsilon 最大值（最终贪婪概率）
        self.epsilon_increment = e_greedy_increment  # epsilon 递增步长（用于逐步提高贪婪概率）
        # 如果设置了递增，则从 0 开始；否则直接使用最大值
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.target_update = target_update  # 目标网络更新频率（每多少步硬更新一次）
        self.count = 0       # 训练步数计数器
        self.device = device
        self.history_loss = []  # 历史损失记录

    def take_action(self, state):
        """
        epsilon-贪婪策略选择动作
        以 epsilon 概率选择 Q 值最大的动作（利用），以 1-epsilon 概率随机选择（探索）
        """
        if np.random.random() < self.epsilon:
            # 利用：选择 Q 值最大的动作
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        else:
            # 探索：随机选择动作
            action = np.random.randint(0, self.action_dim)
        return action

    def update(self, transition_dict):
        """
        更新网络参数
        1. 每 target_update 步将主网络参数硬拷贝到目标网络
        2. 计算 TD 目标：Q_target = r + γ * max_a' Q_target(s', a')
        3. 最小化 MSE(Q(s,a), Q_target) 更新主网络
        """
        # 定期硬更新目标网络（直接复制主网络参数）
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        # 准备训练数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        # 计算当前 Q 值：Q(s, a)
        q_values = self.q_net(states).gather(1, actions)
        # 计算目标 Q 值：r + γ * max_a' Q_target(s', a')
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values

        # 计算 MSE 损失并反向传播
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        self.history_loss.append(dqn_loss.item())

        # 逐步增大 epsilon（从探索向利用过渡）
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
