---
name: caption-gen
description: 给已经画好的图或表生成符合顶会风格的英文 caption。当用户说"给这图配个 caption / 写 Figure 1 标题 / Table caption / 图注 / 表注"时调用。
---

# Figure / Table Caption Generation

## 共性规则

1. **不写 "Figure 1:" / "Table 2:" 前缀** —— LaTeX `\caption{}` 会自动加
2. **不写 "The figure shows..." / "This table presents..."** —— 直接以名词或动词开头
3. **首字母大写**:
   - **名词短语 caption**: Title Case (主要词大写) + **不加句号**
     - 例: "Architecture of the ω-adaptive Allocation Network"
   - **完整句 caption**: Sentence case (仅首字母 + 专名大写) + **加句号**
     - 例: "Our method achieves higher hypervolume than GMORL across all preference vectors."
4. **特殊字符 LaTeX 转义** (`%`, `_`, `&`, `#`)
5. **数学符号包公式**: `$\omega$`, `$\Delta\text{HV}$`
6. **避免大模型词**: depict / showcase / illustrate (过度装饰) → 用 show / compare / report

## 图 caption 推荐句式

| 图类型 | 推荐开头 |
|---|---|
| Architecture / framework | "Architecture of ..." / "Overview of the proposed ..." |
| Comparison curve | "Comparison of ... across ..." / "Performance under ..." |
| Ablation | "Effect of ... on ..." / "Ablation on ..." |
| Pareto frontier | "Pareto frontier achieved by ... on the {delay, energy, accuracy} space." |
| Convergence | "Training curve of ... over ... episodes." |
| Heatmap | "Heatmap of ... as a function of ... and ...." |

### 多面板图

```
Pareto frontiers of the proposed FDEdge and three baselines. (a) Stationary
setting; (b) ω-shift at step 200; (c) Channel drift. Markers indicate
hypervolume reference points.
```
(用 (a)(b)(c) 标识子图,每个子图一短句)

## 表 caption 推荐句式

| 表类型 | 推荐开头 |
|---|---|
| 主结果 | "Results on ..." / "Comparison with state-of-the-art on ..." |
| 消融 | "Ablation study on ..." / "Effect of removing ... ." |
| 超参 | "Hyperparameter settings used in ..." |
| 复杂度 | "Computational cost of ... ." |

### 加注释
对表中加粗、下划线、↑↓ 等的解释,放 caption 末尾或表脚注:
```
Results on the FDEdge V2 benchmark. Best is bolded; second-best underlined.
$\uparrow$ / $\downarrow$ denotes that higher / lower is better.
```

## 输出格式

```
[Caption]
<纯 caption 文本,可直接粘进 \caption{}>

[备选版本] (可选,若用户没指定语气)
- 紧凑版: <更短>
- 详细版: <更长,带 (a)(b) 子图说明>

[中文意译]
<...>
```

## FDEdge 项目特化术语

- 三个目标轴写法: "{delay, energy, accuracy}" 而非 "delay/energy/accuracy" (后者只在正文密集表达时用)
- ω 永远是 LaTeX `$\omega$`,不是 "omega"
- HV 首次出现写全: "hypervolume (HV)"
