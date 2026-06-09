---
name: compress-expand
description: 段落字数控制 —— 缩写 (减 5-15 词) 或扩写 (增 5-15 词,通过逻辑挖掘而非灌水)。当用户说"缩短 / 压缩 / 压一下 / over page limit / 删点字" 或 "扩写 / 展开一下 / 字数不够" 时调用。
---

# Compress / Expand

## 模式 1: 缩写 (Compress)

### 缩写技法 (按优先级)

1. **从句 → 短语**: "which is responsible for X" → "responsible for X"
2. **被动 → 主动**: "is conducted by" → 删动词主语化
3. **冗余词删除**:
   - "in order to" → "to"
   - "due to the fact that" → "because"
   - "a number of" → "several"
   - "the majority of" → "most"
   - "is able to" → "can"
   - "it is important to note that X" → 直接说 X
4. **形容词副词砍**: 修饰性的 "carefully / thoroughly / significantly" 可删
5. **同义重复合并**: "novel and new" → "new"

### 缩写硬性约束

- **不删数据、参数、术语、citation**
- **不改结论**
- **不破坏 LaTeX 结构** (公式 / `\ref` / `\cite` 原样保留)
- 目标范围: 减 5-15 词 (除非用户明确要求更多)
- 若已无可删处,**输出原文** + "[已达最简,无可压缩]"

## 模式 2: 扩写 (Expand)

### 扩写技法 (按优先级)

1. **暴露隐含结论**: "X improves Y" → 加 "; this improvement is driven by Z" (Z 是逻辑上必然但未明说的)
2. **补因果链**: 把两个独立陈述用 "because / thus / consequently" 串起来
3. **补对比项**: "method A achieves X" → "method A achieves X, while baseline B reaches only Y" (前提是 B 的数字真实存在)
4. **补限定条件**: "X works" → "X works under the assumption that ω is observable"
5. **补强连接词**: Furthermore, Notably, In particular (谨慎用,见 dehumanize skill 反对滥用)

### 扩写硬性约束

- **不灌水**: 加的每个词必须有逻辑功能
- **不编造数字**: 扩写时引用数字必须用户提供过 (参见 `[[feedback_data_rigor]]`)
- 目标范围: 加 5-15 词
- 不允许"水化"成 AI 味段落 (与 `dehumanize` skill 冲突时优先去 AI 味)

## 输出格式

```
[LaTeX]
<新版段落>

[中文回译]
<...>

[修改日志 — 缩写]
- 删: "in order to (3 词 → 1 词)"
- 改从句为短语: "...which is responsible for..." → "...responsible for..."
- 净减少: -N 词

[修改日志 — 扩写]
- 补: 在 "X improves Y" 后加 "; primarily due to the ω-conditioned policy capturing trade-offs explicitly" (+12 词)
- 净增加: +N 词
```
