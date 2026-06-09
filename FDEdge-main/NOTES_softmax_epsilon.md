# NOTES: ε-softmax 语义一致性问题

## 背景
SCI 投稿前自审时怀疑 PCPolicyNet 每步 softmax 与 DDPM 高斯假设冲突。
读完 FDEdge 原论文后，**严重程度从 P0 降为 P2**，但问题没被"论证"，只是被论文回避。

## 论文怎么说（FDEdge, IEEE TWC 2024）

- **Sec IV-A.4 第 4 条**：softmax 只在去噪结束、得到 `x_{t,n,0}` 之后用一次。
- **Theorem 2**：`ε_θ(x, i, s)` 是 MLP 输出，`ε ∼ N(0, I)`，无界、不在单纯形上。
- 论文叙事是干净的：MLP 出 ℝ⁶ → 标准 DDPM 反向更新 → 末端单次 softmax。

## 代码实际怎么做

`mofd_model.py:87` 与 `fdsac_model.py:74` 都是：
```python
return F.softmax(out, dim=1)   # 每一步都 softmax
```
这跟 FDEdge GitHub 参考实现一致，**但和论文 Theorem 2 假设不一致**。

## 所以是漏洞吗？

**严格意义上是**：论文论证的是"末端 softmax + 高斯 ε"，代码做的是"每步 softmax + 单纯形 ε"，
Theorem 2 在代码的设定下不严格成立。

**实操意义上不是**：
1. FDEdge 已通过 IEEE TWC 同行评审，代码层面这么写没被揪。
2. categorical / multinomial diffusion 文献（Hoogeboom 2021 等）也有类似的 simplex 投影做法。
3. 引用 FDEdge 即获得先例保护。

## 投稿应对（最小成本方案）

**不改代码**。但 Method 章节加 1-2 句：
- 末端 softmax 是把输出投到动作单纯形 Δ。
- 网络结构沿用 FDEdge [Xu et al., 2024]。
- Discussion / Limitations 一句话："a strictly Gaussian ε would require dropping
  the per-step softmax, which we leave for future work"。

## 真要修（不建议在投稿前动）

```python
# mofd_model.py PCPolicyNet.forward
return out                        # ε ∈ ℝ⁶, 不再 softmax

# feedback_diffusion.py p_sample_loop 末尾
return F.softmax(x, dim=-1)       # 只在最后一次投影
```
代价：训练动态彻底变，需重训全套实验 + 重新调超参。

## 当前定级
| 项 | 之前 | 现在 |
|---|---|---|
| 优先级 | P0 必修 | P2 写作处理 |
| 投稿风险 | 高（理论错误） | 低（设计选择，有先例） |
| 行动 | 改代码重训 | 加注释 + 论文标注 |
