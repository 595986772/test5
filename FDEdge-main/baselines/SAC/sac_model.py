import numpy as np
import torch
import torch.nn.functional as F
import collections
import random


class ReplayBuffer:
    """经验回放池，存储 (状态, 动作, 奖励, 下一状态) 元组"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        """添加经验记录"""
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """随机采样"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class PolicyNet(torch.nn.Module):
    """
    策略网络（Actor）
    输入系统状态，输出各动作的概率分布（经 softmax 归一化）
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)  # 输出动作概率分布


class QValueNet(torch.nn.Module):
    """Q 值网络（Critic），输入状态，输出各动作的 Q 值"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC:
    """
    SAC (Soft Actor-Critic) 算法 - 离散动作版本
    包含：
      - Actor: 策略网络，输出动作概率分布
      - Critic: 双 Q 网络 (Q1, Q2)
      - Target: 双目标网络
      - Alpha: 自动调节的温度参数
    与 FDSAC 的区别：Actor 使用普通 MLP 而非扩散模型
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha, alpha_lr, target_entropy, tau, gamma, device):
        # ===== Actor 网络（普通 MLP 策略网络）=====
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        # ===== 双 Critic 网络 =====
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        # ===== 双目标网络 =====
        self.target_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数与 Critic 相同
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_2.load_state_dict(self.critic_2.state_dict())

        # ===== 优化器 =====
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # ===== 温度参数 alpha（使用 log 值训练更稳定）=====
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 允许自动调节
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵值
        self.gamma = gamma  # 折扣因子
        self.tau = tau       # 软更新系数
        self.device = device
        self.history_loss = []

    def take_action(self, state):
        """根据策略网络输出的概率分布采样动作"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)  # 构建分类分布
        action = action_dist.sample()  # 按概率采样动作
        return action.item()

    def calc_target_q(self, rewards, next_states):
        """
        计算目标 Q 值
        Q_target = r + γ * (Σ π(a|s') * min(Q1_target, Q2_target) + α * H(π))
        """
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)

        # 策略熵 H = -Σ p * log(p)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

        # 双目标网络取最小 Q 值
        target1_q1_value = self.target_1(next_states)
        target2_q2_value = self.target_2(next_states)

        min_q_value = torch.sum(next_probs * torch.min(target1_q1_value, target2_q2_value), dim=1, keepdim=True)
        next_value = min_q_value + self.log_alpha.exp() * entropy
        q_target = rewards + self.gamma * next_value
        return q_target

    def soft_update(self, net, target_net):
        """软更新目标网络：θ_target = (1-τ)*θ_target + τ*θ"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """
        一步训练：更新 Critic、Actor、Alpha 和目标网络
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        # ===== 1. 更新 Critic 网络 =====
        target_q_values = self.calc_target_q(rewards, next_states)

        # 更新 Critic 1
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, target_q_values.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # 更新 Critic 2
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, target_q_values.detach()))
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ===== 2. 更新 Actor 网络 =====
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # 策略熵
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        # 最大化 (期望Q值 + α * 熵)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== 3. 更新温度参数 alpha =====
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ===== 4. 软更新目标网络 =====
        self.soft_update(self.critic_1, self.target_1)
        self.soft_update(self.critic_2, self.target_2)
