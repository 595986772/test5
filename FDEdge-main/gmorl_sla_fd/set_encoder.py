"""Set-conv 编码器 (从 GMORL Model.py 抽出, 去掉 tianshou 依赖)。

GMORL 的 "set-conv" 本质是 Conv1d(in_ch, ch, kernel_size=1): 对每台服务器槽
共享同一套权重做特征变换 (PointNet 式 point encoder)。配合 mask + 补零, 支持
可变边缘数 (≤MAX_EDGE_NUM)。这里把它做成独立 nn.Module, 输出一个"条件向量"
cond, 供反馈扩散 actor 当 conditioning (编码一次, 去噪循环里复用)。

诚实边界: 后接 flatten+MLP, 故 **不是置换不变**, 且服务器数封顶 MAX_EDGE_NUM+1。
论文只能宣称"支持可变边缘数 ≤N", 不能吹"任意拓扑/置换不变"。
"""
import torch
import torch.nn as nn

MAX_EDGE_NUM = 10
N_SLOTS = MAX_EDGE_NUM + 1   # cloud + 最多 10 个 edge = 11 个服务器槽
SERVER_FEAT = 67             # 每槽特征: 7 标量 + 60 格任务直方图 (见 Env.get_obs)


class ConvResBlock(nn.Module):
    """1x1 Conv1d 残差块, 对服务器维 (length) 共享权重。源自 GMORL conv_resblock。"""

    def __init__(self, in_ch, ch, out_ch=None, block_num=2):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, ch, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=False),
        )
        self.blocks = nn.ModuleList()
        for _ in range(block_num):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, kernel_size=1),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv1d(ch, ch, kernel_size=1),
            ))
        self.acts = nn.ModuleList([nn.LeakyReLU(0.1, inplace=False) for _ in range(block_num)])
        self.out_conv = nn.Conv1d(ch, out_ch, kernel_size=1) if out_ch else None

    def forward(self, x):  # x: [B, in_ch, L]
        x = self.in_conv(x)
        for blk, act in zip(self.blocks, self.acts):
            x = act(blk(x) + x)
        if self.out_conv is not None:
            x = self.out_conv(x)
        return x


class SetConvEncoder(nn.Module):
    """dict obs -> 条件向量 cond [B, cond_dim]。

    forward 输入 (都已是 torch tensor, 在正确 device):
      servers:    [B, SERVER_FEAT, N_SLOTS]  每槽 67 维特征 (含补零槽)
      preference: [B, 2]                      偏好 [w, 1-w]
      mask2:      [B, N_SLOTS]                合法服务器=1, 补零槽=0
    输出:
      cond:       [B, cond_dim]               扩散 actor 的 conditioning
    """

    def __init__(self, in_ch=SERVER_FEAT, conv_ch=256, cond_dim=256,
                 n_slots=N_SLOTS, pref_dim=2, block_num=2):
        super().__init__()
        self.n_slots = n_slots
        self.conv = ConvResBlock(in_ch=in_ch, ch=conv_ch, out_ch=conv_ch, block_num=block_num)
        # flatten 所有槽 -> MLP. 先用 mask 把补零槽的特征清零, 避免 -1 垃圾污染。
        self.proj = nn.Sequential(
            nn.Linear(conv_ch * n_slots, cond_dim),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(cond_dim, cond_dim),
        )
        # GMORL 做法: 偏好 w 作为加性 bias 注入
        self.pref_bias = nn.Sequential(
            nn.Linear(pref_dim, cond_dim),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_dim = cond_dim

    def forward(self, servers, preference, mask2):
        B = servers.size(0)
        f = self.conv(servers)                      # [B, conv_ch, n_slots], 每槽共享编码
        f = f * mask2.unsqueeze(1)                  # 补零槽特征清零 ([B,1,n_slots] 广播)
        flat = f.reshape(B, -1)                     # [B, conv_ch*n_slots]
        cond = self.proj(flat)                      # [B, cond_dim]
        cond = cond + self.pref_bias(preference)    # 偏好加性 bias (GMORL)
        return cond
