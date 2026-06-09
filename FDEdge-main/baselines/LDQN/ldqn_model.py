"""
LDQN (LSTM-augmented Deep Q-Network) baseline for MOFD.
Ref: FDEdge paper, which cites LDQN as 'DQN + LSTM over task arrival history'.

实现细节:
  - 用 LSTMCell 作为状态编码器: emb(state) -> LSTMCell -> Q-head
  - 推理期: hidden (h, c) 在一个 episode 内持续, 每个 episode 重置
  - 训练期: batch 采样互不连续, hidden 置零 (常见近似, 论文 repo 也是如此)
  - 多目标: 奖励直接用 omega 线性标量化, omega 拼进状态 (由 MOFDEnvironment 提供)
  - mask: 用 -1e18 填充无效动作位的 Q, 保证 argmax 落在有效 ES
"""
import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, r, s_next, mask_next):
        self.buffer.append((s, a, r, s_next, mask_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sn, mn = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(sn, dtype=np.float32), np.array(mn, dtype=np.float32))

    def size(self):
        return len(self.buffer)


class LQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, emb_dim=64):
        super().__init__()
        self.emb = nn.Linear(state_dim, emb_dim)
        self.lstm = nn.LSTMCell(emb_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.hidden_dim = hidden_dim

    def zero_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, s, h=None):
        e = F.relu(self.emb(s))
        if h is None:
            h = self.zero_hidden(s.size(0), s.device)
        h_new, c_new = self.lstm(e, h)
        q = self.head(h_new)
        return q, (h_new, c_new)


class LDQN:
    def __init__(self, state_dim, action_dim, hidden_dim=128, emb_dim=64,
                 learning_rate=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 target_update=100, device=torch.device('cpu')):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.count = 0
        self.device = device

        self.q_net = LQNet(state_dim, hidden_dim, action_dim, emb_dim).to(device)
        self.target_net = LQNet(state_dim, hidden_dim, action_dim, emb_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self._eval_hidden = None

    def reset_hidden(self):
        """每个 episode 开始前调用, 清空持久隐状态."""
        self._eval_hidden = None

    def take_action(self, state_np, mask_np, stochastic=True):
        s = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
        with torch.no_grad():
            q, self._eval_hidden = self.q_net(s, self._eval_hidden)
            q_np = q.cpu().numpy()[0]
        q_np = np.where(mask_np > 0.5, q_np, -1e18)
        if stochastic and np.random.rand() < self.epsilon:
            valid = np.where(mask_np > 0.5)[0]
            return int(np.random.choice(valid)) if len(valid) else 0
        return int(np.argmax(q_np))

    def update(self, batch):
        s, a, r, sn, mn = batch
        s_t = torch.tensor(s, device=self.device)
        a_t = torch.tensor(a, device=self.device).view(-1, 1)
        r_t = torch.tensor(r, device=self.device).view(-1, 1)
        sn_t = torch.tensor(sn, device=self.device)
        mn_t = torch.tensor(mn, device=self.device)

        q_all, _ = self.q_net(s_t, None)
        q = q_all.gather(1, a_t)
        with torch.no_grad():
            qn_all, _ = self.target_net(sn_t, None)
            qn_all = qn_all.masked_fill(mn_t < 0.5, -1e18)
            qn_max = qn_all.max(1, keepdim=True)[0]
            target = r_t + self.gamma * qn_max
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return dict(loss=float(loss.detach()), eps=self.epsilon)
