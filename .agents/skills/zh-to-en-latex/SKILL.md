---
name: zh-to-en-latex
description: 把中文论文草稿翻译并润色为可直接粘进 LaTeX 源码的英文,目标顶会水准 (NeurIPS / ICML / ICLR / INFOCOM / TMC)。当用户给出中文段落要求"翻译成英文" / "写成 LaTeX" / "把这段写成论文英文" 时调用。
---

# Chinese → English LaTeX

## 工作原则

1. **logic 优先于 literal**: 优先重组使逻辑顺畅,而非逐字翻译。
2. **保持 LaTeX 纯净**: 不主动添加 `\textbf`、`\emph`、列表、emoji 等装饰。原文有的命令保留;原文没有的不加。
3. **学术用词**: 选择 discipline-standard 词,拒绝华丽词 (leverage / delve / pivotal / paradigm)。常见替换:
   - leverage → use / employ
   - delve into → investigate / examine
   - showcase → show / present
   - pivotal → key / central
   - paradigm shift → 删掉或改 "new approach"
4. **时态**: 方法/贡献描述用 present tense;只有"prior work proposed..."这类历史叙述才 past tense。
5. **数学/特殊字符 LaTeX 转义**:
   - `95%` → `95\%`
   - `model_v1` → `model\_v1`
   - `&` → `\&`,`#` → `\#`
   - 公式包在 `$...$` 或 `\(...\)` 中
6. **属格回避**: 写 "the performance of GMORL" 而非 "GMORL's performance"。
7. **避免缩约**: it's → it is, don't → do not。

## 项目术语对照 (必须严格使用,见 paper-writing skill)

`ω-adaptive allocation` / `preference vector ω` / `Pareto frontier` / `hypervolume (HV)` / `three-objective {delay, energy, accuracy}` / `replay buffer` / `test-time adaptation (TTA)` / `non-stationary environment`

## 输出格式 (固定三段)

```
[LaTeX]
<可直接复制的英文段落,LaTeX 转义就位>

[中文回译]
<把上面英文逐句意译回中文,方便用户审校>

[修改日志]
<如果对原文做了非翻译性修改 (合并句、补充逻辑词、调整顺序), 列出原因; 否则写"仅翻译,无结构调整">
```

## 反例 (一律不允许)

- 输出包含 markdown 列表
- 输出含 emoji
- 翻译为 "Our method's accuracy" (属格)
- 用 "It's worth noting that..." (废话开场 + 缩约)
- 凭印象生造实验数字
- 句首突然 "Furthermore," / "Moreover," 堆砌 (除非原文真的有递进)
