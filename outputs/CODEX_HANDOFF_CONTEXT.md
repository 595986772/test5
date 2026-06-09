# FDEdge / MOFD V5 Codex Handoff Context

这个文件用于把当前项目和这轮对话的上下文迁移到另一个 Codex 账号。  
新账号打开本项目后，可以先让 Codex 阅读本文件，再继续工作。

建议给新 Codex 的第一句话：

```text
请先阅读 outputs/CODEX_HANDOFF_CONTEXT.md，然后继续这个 FDEdge / MOFD V5 项目。后续回答请保持术语一致：PC-FDN、H-MCSS、Vector-Q、COR、PopArt-lite、偏好条件化、delay-energy 多目标边缘卸载。解释公式时按参数逐项解释，尽量用普通文本公式，不要默认使用 LaTeX 渲染。
```

## 1. 项目位置与主要目录

当前工作区：

```text
D:\python_project\实验版5
```

主要目录：

```text
FDEdge-main/       主项目代码、实验脚本、论文/中期材料
dataset/           OpenB / trace 数据
outputs/           已生成输出文件，包括 PPT 与本上下文文件
.agents/skills/    项目本地写作/审稿/翻译技能
.claude/skills/    另一套本地技能定义
```

已生成的中期 PPT：

```text
outputs/019e6924-c217-70d1-9072-cd17a53b7708/presentations/pc-fdn-midterm/output/pc-fdn-midterm-report.pptx
```

## 2. 项目主题

研究主题：

```text
面向多目标边缘卸载的偏好条件化反馈扩散策略研究
```

核心问题：

```text
1. 偏好泛化：希望一次训练后适配不同用户偏好，而不是每个偏好单独训练一个模型。
2. 梯度失衡：delay 与 energy 奖励量级差异较大，delay 容易主导 critic loss。
3. 目标扩展：研究计划中还包括 delay-energy-accuracy 三目标扩展，但当前 V5 代码主要是 delay-energy 二目标。
```

核心方法：

```text
PC-FDN + H-MCSS + Vector-Q + COR + PopArt-lite
```

一句话概括：

```text
V5 用偏好条件化反馈扩散 Actor 生成卸载动作概率，用 H-MCSS 在历史、先验、随机三类候选中择优，用 Vector-Q 分别评价 delay 和 energy，再通过 PopArt-lite 与 COR 解决通道量级失衡和偏好梯度冲突。
```

## 3. 关键代码文件

```text
FDEdge-main/mofd_v5.py
当前 V5 模型主体：PC-FDN Actor、Vector-Q Critic、COR、PopArt-lite、SAC 更新。

FDEdge-main/mofd_main.py
训练主入口：偏好网格、episode loop、Replay Buffer、Pareto / HV 评估。

FDEdge-main/mofd_environment.py
V1 纯边缘环境。

FDEdge-main/mofd_environment_v2.py
V2 云 + 边缘环境，action 0 是 cloud，action 1..E 是 edge。

FDEdge-main/mofd_model.py
PCPolicyNet，反馈扩散 Actor 内部的策略网络骨干。

FDEdge-main/feedback_diffusion.py
DDPM / feedback diffusion 的基础实现。

FDEdge-main/helpers.py
VP beta schedule、preference set、Pareto front、2D hypervolume 等工具。

FDEdge-main/mofd_v2_main.py
在 cloud + edge V2 环境上训练的入口。

FDEdge-main/mofd_v2_dense_main.py
V2 dense workload 变体入口。
```

注意：

```text
mofd_main.py 当前默认配置里 use_v7=True。
如果要跑纯 V5，需要设置：

use_v7 = False
use_v6 = False
use_v5 = True
use_cor = True
use_popart = True
use_envelope = False
```

当前讨论中确认过的重要训练配置：

```text
delay_scale = 0.05
energy_scale = 0.25
alpha_T = 1.0
alpha_E = 1.0
target_entropy = 0.5
gamma = 0.95
tau = 0.005
```

## 4. V5 整体架构

整体流程：

```text
任务到达 / 环境采样
→ 构造状态 s = [任务特征, 偏好 omega, 服务器特征]
→ PC-FDN Actor 生成动作概率
→ H-MCSS 构造 feedback / prior / random 三路候选
→ Vector-Q 双 Critic 对候选进行评价
→ 按当前 omega 标量化 Q 值并选择动作
→ 环境执行卸载，得到 delay / energy
→ 构造向量奖励 r_vec = [r_T, r_E]
→ 写入 Replay Buffer
→ 采样 batch 更新 Critic、Actor、温度 alpha_H、Target Critic
```

状态结构：

```text
state[0]  = 归一化任务数据量
state[1]  = 归一化任务计算量 / 计算密度
state[2]  = omega_T
state[3]  = omega_E
state[4:] = 每个服务器的特征
```

每个服务器特征通常包含：

```text
[CPU 频率, 队列长度, valid flag, 信道增益]
```

奖励结构：

```text
r_T = -delay  * delay_scale
r_E = -energy * energy_scale
r_vec = [r_T, r_E]
```

V5 Replay Buffer 保存的是向量奖励，不是提前压成标量的 reward。

## 5. PC-FDN Actor

PC-FDN Actor 的输入：

```text
当前状态 s
当前扩散 latent x_k
扩散步 k
偏好 omega，已经写在 state[2:4]
```

PCPolicyNet 内部拆成三类信息：

```text
任务-偏好编码：
[task_size, workload, omega_T, omega_E]

服务器编码：
每个服务器 [f, queue, valid, channel]

扩散步编码：
sinusoidal timestep embedding
```

然后拼接：

```text
h = [server_embedding, context_embedding, timestep_embedding, latent_action]
```

输出动作概率向量。

## 6. H-MCSS 三源候选

推理 / 训练交互时，V5 构造三路候选起点：

```text
feedback ：当前 episode 中历史动作概率 latent
prior    ：omega-buffer 或 uniform 给出的 episode 级先验
random   ：随机高斯噪声
```

三路分别经过 Actor 去噪，得到三个动作概率分布。  
然后用双 Critic 的最小 Vector-Q 进行评价：

```text
Q_eff(s,a;omega) = (omega * alpha)^T Q_vec(s,a)
```

对每个候选分布计算期望 Q：

```text
expected_q_i = sum_a pi_i(a|s) * Q_eff(s,a;omega)
```

选择 `expected_q_i` 最大的候选。

## 7. Vector-Q Critic

传统 scalar critic：

```text
Q(s) in R^A
```

V5 Vector-Q critic：

```text
Q(s) in R^(A x 2)
Q(s,a) = [Q_T(s,a), Q_E(s,a)]
```

含义：

```text
Q_T ：动作 a 对 delay 目标的价值估计
Q_E ：动作 a 对 energy 目标的价值估计
```

推理或 Actor 更新时再按偏好标量化：

```text
Q_eff(s,a;omega) = omega_T * alpha_T * Q_T(s,a)
                 + omega_E * alpha_E * Q_E(s,a)
```

当前推荐配置中：

```text
alpha_T = 1.0
alpha_E = 1.0
```

所以基本就是：

```text
Q_eff = omega_T * Q_T + omega_E * Q_E
```

## 8. Critic 更新

V5 先计算下一状态动作分布：

```text
pi_next = Actor(s_next, latent_next, prior, mask_next)
```

再用双 target critic：

```text
min_Q_vec = min(Q_target1, Q_target2)
```

向量状态价值：

```text
V_vec = sum_a pi_next(a) * min_Q_vec(s_next,a)
```

代码里的 vector Bellman target：

```text
target_vec = r_vec + gamma * (V_vec + alpha_H * entropy_next / N_OBJ)
```

其中：

```text
N_OBJ = 2
alpha_H = exp(log_alpha)
entropy_next = -sum_a pi_next(a) * log(pi_next(a))
```

## 9. PopArt-lite

PopArt-lite 维护每个目标通道的运行 RMS：

```text
sigma_k = running RMS of target_vec[:, k]
```

Critic loss 中会除以 `sigma_k^2`：

```text
loss_k = (omega_k * alpha_k / sigma_k^2) * (Q_k - target_k)^2
```

作用：

```text
平衡 delay 和 energy 的数值量级，避免 delay 通道因为数值大而主导 Critic 更新。
```

注意：

```text
当前代码里的 PopArt-lite 只作用于 critic loss，不对 Q 输出做完整 PopArt 反归一。
```

## 10. COR 冲突目标正则化

COR 用于处理不同偏好之间的梯度冲突。

步骤：

```text
1. 从 Dirichlet 分布采样一个辅助偏好 omega_sample。
2. 把 batch 状态中的 state[2:4] 替换为 omega_sample，得到 s_dagger。
3. 分别计算原偏好损失 L(omega) 和辅助偏好损失 L(omega_sample)。
4. 计算两个损失对 Critic 参数梯度的余弦相似度 rho。
5. 如果 rho < c，说明梯度方向冲突，加入 COR penalty。
```

COR 惩罚：

```text
cor_weight = max(c - rho, 0) * lambda_COR

L_COR = cor_weight *
        E[sum_k omega_sample_k *
          (Q_current_k(s_dagger,a) - Q_target_k(s_dagger,a))^2]
```

默认：

```text
c = 0.0
lambda_COR = 0.1
```

直观含义：

```text
如果重 delay 的梯度和重 energy 的梯度方向相反，COR 会约束 Q 网络在 relabeled 状态上不要剧烈偏离 target Q，从而平滑偏好轴上的 Q 函数。
```

## 11. SAC 熵项

策略目标中有：

```text
alpha_H * H(pi_theta)
```

其中：

```text
H(pi_theta) ：策略熵，表示动作概率分布的随机性
alpha_H     ：SAC 熵温度系数，控制模型有多重视探索
```

例子：

```text
pi = [0.98, 0.01, 0.01]
熵很低，策略几乎只选一个服务器。

pi = [0.34, 0.33, 0.33]
熵较高，多个服务器都有机会被尝试。
```

`alpha_H` 的作用：

```text
alpha_H 大 → 更鼓励探索，动作概率更分散
alpha_H 小 → 更重视当前高 Q 动作，动作概率更集中
```

代码中：

```text
alpha_H = exp(log_alpha)
target_entropy = 0.5
```

如果策略熵太低，`alpha_H` 会增大，鼓励探索。  
如果策略熵太高，`alpha_H` 会减小，让策略更关注高 Q 动作。

## 12. 公式解释偏好

用户喜欢这样的解释格式：

```text
π        ：策略，也就是模型学到的任务卸载决策规则
x ~ π    ：卸载动作 x 是由策略 π 产生的
x_{m,e}  ：任务 m 是否卸载到服务器 e，1 表示选择，0 表示不选择
γ^m      ：折扣因子，越靠后的任务影响越小
T_m      ：任务 m 的时延
E_m      ：任务 m 的能耗
(T_m,E_m)^T ：二目标代价向量
```

用户还明确要求：

```text
公式推导解释尽量不要用 LaTeX 渲染，改成直接显示的普通文本公式。
```

## 13. 已解释过的关键公式

### 13.1 链路速率

普通含义：

```text
C_{u,e} = W * log2(1 + p_off * |h_{u,e}|^2 / sigma^2)
```

参数：

```text
C_{u,e}        ：用户 u 到服务器 e 的传输速率
W              ：无线信道带宽
p_off          ：任务卸载时的发射功率
h_{u,e}        ：用户 u 到服务器 e 的信道系数
|h_{u,e}|^2    ：信道增益
sigma^2        ：噪声功率
```

代码注意：

```text
当前代码没有完整实现 Shannon 公式，而是用：
v = base_tran_rate * channel_gain
```

### 13.2 任务时延

含义：

```text
T_m = 传输时延 + 计算时延 + 排队等待时延
```

代码对应：

```text
rho_d = comp_density[n] * d_n
tran_delay = d_n / v
comp_delay = rho_d / f_b
wait_delay = (queue_len + queue_bef) / f_b
delay = tran_delay + comp_delay + wait_delay
```

### 13.3 任务能耗

含义：

```text
E_m = 卸载传输能耗 + 服务器执行能耗
```

代码对应：

```text
e_off = p_off * tran_delay
e_exe = kappa * f_b^2 * rho_d
energy = e_off + e_exe
```

V2 环境中：

```text
action == 0 → cloud，使用 cloud_kappa
action > 0  → edge，使用 edge kappa
```

### 13.4 多目标优化目标

普通文本：

```text
min over pi:
E_{x ~ pi} [
  sum over m in M:
    gamma^m * (T_m, E_m)^T
]

subject to:
x_{m,e} in {0,1}
sum_e x_{m,e} = 1
```

含义：

```text
π              ：策略，也就是模型学到的任务卸载决策规则
x ~ π          ：卸载动作 x 是由策略 π 产生的
x_{m,e}        ：任务 m 是否卸载到服务器 e，1 表示选择，0 表示不选择
γ^m            ：折扣权重，越靠后的任务影响越小
T_m            ：任务 m 的时延
E_m            ：任务 m 的能耗
(T_m,E_m)^T    ：二目标代价向量
```

### 13.5 前向扩散

普通文本：

```text
x_k = sqrt(lambda_bar_k) * x_0
    + sqrt(1 - lambda_bar_k) * epsilon
```

含义：

```text
x_0             ：干净的动作概率 latent
x_k             ：第 k 步加噪后的动作 latent
lambda_bar_k    ：累计信号保留比例
epsilon         ：标准高斯噪声
```

### 13.6 VP 噪声调度

普通文本：

```text
beta_k = 1 - exp(
           - beta_min / K
           - ((2k - 1) / (2K^2)) * (beta_max - beta_min)
         )

lambda_k = 1 - beta_k

lambda_bar_k = lambda_1 * lambda_2 * ... * lambda_k
```

作用：

```text
公式 6 说明每个扩散步添加的噪声强度。

beta_k      ：第 k 步添加的噪声强度
lambda_k    ：第 k 步保留的原始信号比例
lambda_bar_k：从第 1 步到第 k 步累计保留的原始信号比例
```

推导思路：

```text
1. 设连续噪声强度线性变化：
   beta(t) = beta_min + t * (beta_max - beta_min)

2. 第 k 步对应时间区间：
   [(k - 1) / K, k / K]

3. 对 beta(t) 在这个区间积分：
   integral = beta_min / K
            + ((2k - 1) / (2K^2)) * (beta_max - beta_min)

4. VP diffusion 中信号保留率：
   lambda_k = exp(-integral)

5. 因为：
   beta_k = 1 - lambda_k

6. 得到：
   beta_k = 1 - exp(-integral)
```

### 13.7 反向去噪

普通文本：

```text
x_{k-1}
= 1 / sqrt(lambda_k)
  * [
      x_k
      - beta_k / sqrt(1 - lambda_bar_k)
        * epsilon_theta(x_k, k, s, omega)
    ]
  + sqrt(beta_tilde_k) * epsilon
```

推导来源：

```text
1. 前向扩散：
   x_k = sqrt(lambda_bar_k) * x_0
       + sqrt(1 - lambda_bar_k) * epsilon

2. 由 x_k 估计干净 x_0：
   x_0_hat =
   [x_k - sqrt(1 - lambda_bar_k) * epsilon_theta] / sqrt(lambda_bar_k)

3. DDPM 后验：
   q(x_{k-1} | x_k, x_0) 是高斯分布

4. 用 x_0_hat 替代真实 x_0，化简后得到反向均值：
   mu_theta =
   1 / sqrt(lambda_k)
   * [
       x_k
       - beta_k / sqrt(1 - lambda_bar_k)
         * epsilon_theta
     ]

5. 最后采样：
   x_{k-1} = mu_theta + sqrt(beta_tilde_k) * epsilon
```

直观含义：

```text
当前带噪动作 latent
→ Actor 预测噪声
→ 减掉预测噪声
→ 得到更干净的上一层 latent
→ 加一点随机扰动
```

### 13.8 Masked Softmax 策略

普通文本：

```text
pi_theta(a | s, x_K, K, omega)
= [softmax(x_0) * mask(s)]
  / L1_norm[softmax(x_0) * mask(s)]
```

含义：

```text
pi_theta       ：Actor 输出的动作概率分布
a              ：卸载动作，即选择哪个服务器
x_0            ：反向扩散最终生成的干净动作 latent
softmax(x_0)   ：把 latent 转成动作概率
mask(s)        ：有效动作 mask，非法服务器位置为 0
L1_norm        ：所有概率之和，用于重新归一化
```

动作选择：

```text
a_{t,n} = argmax_a pi_theta(a | ...)
```

训练阶段代码通常可以随机采样动作，评估阶段用 `argmax`。

### 13.9 SAC 偏好条件化目标

普通文本：

```text
J(pi_theta; omega)
= E [
    sum_t gamma^t *
    (
      (omega * alpha)^T * r_{t,n}
      + alpha_H * H(pi_theta)
    )
  ]
```

参数：

```text
J(pi_theta; omega)        ：在偏好 omega 下策略 pi_theta 的优化目标
pi_theta                  ：Actor 策略网络，也就是 PC-FDN
theta                     ：Actor 网络参数
omega                     ：用户偏好向量，例如 [omega_T, omega_E]
omega_T                   ：时延目标权重
omega_E                   ：能耗目标权重
alpha                     ：目标通道缩放系数，例如 [alpha_T, alpha_E]
r_{t,n}                   ：任务 n 在时间槽 t 的奖励向量，例如 [r_T, r_E]
r_T                       ：时延奖励，代码中为 -delay * delay_scale
r_E                       ：能耗奖励，代码中为 -energy * energy_scale
(omega * alpha)^T r       ：按照当前偏好加权后的单步标量奖励
alpha_H                   ：SAC 熵温度系数，控制探索强度
H(pi_theta)               ：策略熵，表示动作分布的随机性
gamma^t                   ：折扣因子，越远的未来奖励影响越小
```

直观含义：

```text
模型不仅要在当前偏好 omega 下获得更高的 delay-energy 综合奖励，
还要保持一定策略熵，避免过早固定到单一动作，从而保留探索能力。
```

## 14. 中期 PPT 信息

PPT 是根据用户提供的 DOCX 和大纲生成的，风格含华南师范大学元素。

PPT 文件路径：

```text
outputs/019e6924-c217-70d1-9072-cd17a53b7708/presentations/pc-fdn-midterm/output/pc-fdn-midterm-report.pptx
```

中期材料中提到的个人信息：

```text
学生：陈亮
导师：刘波
学院：华南师范大学计算机学院
日期：2026.5.28
```

PPT 标题：

```text
面向多目标边缘卸载的偏好条件化反馈扩散策略研究
```

## 15. 后续继续时的注意事项

1. 如果用户问“当前 v5 模型”，优先按 `mofd_v5.py` 源码解释，而不是只按 PPT 概念解释。
2. 如果涉及“三目标 accuracy”，要明确当前 V5 代码是 delay-energy 二目标，accuracy/fidelity 是研究扩展计划。
3. 如果解释公式，尽量按“这里各参数含义是：”逐项列出。
4. 如果解释公式推导，优先使用普通文本公式。
5. 如果涉及代码运行，注意 Windows 环境和项目路径：

```text
D:\python_project\实验版5\FDEdge-main
```

6. 如果要生成论文文字，保持术语：

```text
preference-conditioned feedback diffusion policy
PC-FDN
Vector-Q critic
Conflict Objective Regularization, COR
PopArt-lite channel normalization
multi-objective edge offloading
delay-energy trade-off
```
