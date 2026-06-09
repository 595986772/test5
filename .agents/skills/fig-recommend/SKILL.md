---
name: fig-recommend
description: 给定实验数据形状/对比维度,推荐合适的学术图表类型与设计要点。当用户说"这数据用什么图 / 帮我选个图表 / 怎么可视化"时调用。仅给推荐,不画图。
---

# Figure Type Recommendation

## 输入需要的信息 (不全则先问)

1. **数据本身**: 维度 (变量个数)、类别 vs 连续、有无时间轴
2. **比较对象**: 几个 method / 几个 setting / 几个 ω
3. **故事**: 你想让读者一眼看出什么? (e.g., "我方法在所有设置都最好" / "随时间收敛更快")
4. **位置**: 单栏 / 双栏 / 半页?

## 图表类型决策树

### 比较 N 个方法在 K 个指标上

| K \ N | N≤3 | N=4-6 | N>6 |
|---|---|---|---|
| K=1 | bar (vertical) | bar | horizontal bar |
| K=2 | 散点 + Pareto frontier | 散点 + 凸包 | 拒绝,改成表 |
| K=3 | 3D 散点 / 三轴雷达 | radar | table + 缩并维度 |
| K≥4 | radar / parallel coordinates | parallel coord | heatmap |

### 时间 / 训练步数

- 单方法多 seed → line + 阴影区间 (±1 std)
- 多方法对比 → 多条 line + 不同 marker / linestyle (避免靠纯颜色区分,色盲友好)
- 长尾收敛 → 双 y 轴或 log x 轴

### 分布

| 数据 | 推荐 |
|---|---|
| 单变量分布 | violin > box > histogram (顶会多用 violin) |
| 二维分布 | hexbin / KDE contour |
| 类别×连续 | grouped violin / split violin |

### 二维关系 / Pareto

- **Pareto frontier**: 散点 + 前沿线 (filled-step 或 spline); 各方法不同 marker; 注明 ↑/↓ 方向
- 三目标 Pareto: 投影成 3 个二维子图 + 一个 3D 透视图,组合成 (a)(b)(c)(d)

### 消融 / 矩阵

- ablation 用 grouped bar (横轴: setting; 颜色: ablated component)
- 二维超参敏感性 → heatmap (color = metric, 标 best 单元格)

## 设计硬性规范 (顶会风格)

1. **字号**: 轴标签 ≥ 8pt @ 单栏图宽,刻度 ≥ 7pt; 图内文字与正文最小字号匹配
2. **配色**:
   - 默认: `tab10` / `viridis` / `cividis` (色盲友好)
   - 避免: 纯红绿对比 (色盲); 过饱和原色
3. **marker**: 不同方法用不同形状 + 颜色 (双重编码),便于黑白打印
4. **网格**: 浅灰 (`#cccccc`) 仅在需要读数时加;Pareto 图不加纵网格
5. **坐标轴**:
   - 共享单位的子图必须统一刻度范围
   - 量级差异大 → log scale,**明确标 "log scale"** 在轴标签
   - 不允许默认的 `1e8` 科学计数法漂浮在轴外
6. **误差**: 多 seed 必须画误差带 / 误差棒 (±1 std 或 95% CI),caption 说明
7. **图例**: 位置不挡数据;长图例放图外右侧;同一篇论文内位置保持一致
8. **背景**: 白色,无边框 / 无阴影

## 反例 (禁止)

- 3D 柱状图 (顶会几乎不接受)
- 立体饼图
- 默认 matplotlib 紫色 + 半透明阴影 (像 PPT)
- 多于 7 种颜色 (人眼区分极限)
- 用 jpg 不用 pdf/svg (LaTeX 应嵌矢量图)

## 输出格式

```
[推荐图表]
<图表类型,一句话>

[核心理由]
<为什么是这个,而不是其他常见选项 (2-3 句话)>

[设计要点]
- 轴: x=..., y=..., (若 log 则注明)
- 颜色: ...
- marker: ...
- 误差: ...
- 图例位置: ...
- 子图布局 (若多面板): (a)(b)(c) 分别画...

[备选方案]
<若用户的故事改变, 可改用 ... 类型>
```

## FDEdge 项目常用图

| 实验 | 推荐 |
|---|---|
| 三方法 vs 三目标 (Pareto) | 3 二维投影子图 + 1 3D 子图; markers 方法; 颜色 ω |
| HV 随训练步 | line + ±1 std 阴影; 横轴 log episodes |
| ω-shift 漂移恢复 | line + 垂直虚线标 shift 时刻; 双 y 轴 (HV + drift score) |
| 通道分布漂移 | grouped violin (横轴: setting; 颜色: method) |
| ω 敏感性 | radar (顶点: 3 目标; 多边形: 多 ω 设置) |
