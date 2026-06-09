---
name: dehumanize
description: 去掉"AI 味"——大模型生成的机械、模板化、过度修饰的痕迹,让段落读起来像真人作者写的。当用户说"去 AI 味 / 这段太像 ChatGPT 写的 / 太套路 / 去模板化"时调用。LaTeX 英文与 Word 中文都适用。
---

# Dehumanize (去 AI 味)

## 触发信号 — 命中即改

### A. 高频 AI 词 (英文)
leverage, delve, tapestry, intricate, multifaceted, holistic, robust(ly), seamless(ly), navigate, embark, pivotal, paramount, paradigm shift, in the realm of, it is important to note, it is worth noting, furthermore (开篇), moreover, in essence, indeed (开篇)

### B. 高频 AI 词 (中文)
毋庸置疑、深刻、深入剖析、范式转变、赋能、生态、闭环、链路、纵深、抓手、底层逻辑、心智、颗粒度、对齐、拉通、抽象出、本质上、不仅...更...、不仅仅是

### C. 结构信号
1. **列表堆砌**: 正文里出现 1./2./3. 或 • 列项 — 改为连贯段落
2. **三段排比**: "It is X. It is Y. It is Z." — 拆开,只留一个
3. **空 "-ing" 分析**: "This shows X, indicating Y, suggesting Z, revealing W..." — 砍掉一层
4. **过度连接词**: 段首必出现 Furthermore / Moreover / Additionally — 删,让逻辑自然过渡
5. **破折号滥用**: 单段超过 2 个 em-dash — 改用句号或括号
6. **重复 hedging**: "may potentially possibly" — 留一个
7. **公式化结尾**: "In summary, ..." / "总而言之,..." — 删,直接结束
8. **二元强调**: "not just X, but Y" 重复出现 — 改写

## 修订原则

1. **高门槛**: 如果一段读着已经像真人写的,**输出原文** + 一句 "[检测通过,无 AI 味]"。不要为修而修。
2. **保留事实**: 只改语言层,不动数据、结论、术语。
3. **节奏多样**: 真人段落往往长短句交错;若原文清一色中长句,合并两短或拆一长。
4. **允许一点主观**: 真人会写 "we found this surprising" / "我们注意到" 这类轻度第一人称——AI 反而很少敢这么写,可以适度保留 / 加入。

## 输出格式 (LaTeX 英文)

```
[LaTeX]
<去 AI 味后的段落; 若无需改, 直接复制原文>

[中文回译]
<...>

[修改日志]
- 删除: "in essence" / "moreover" 开篇 / 项目列表
- 替换: "leverage → use", "robust → reliable"
- 重组: 把 3 句空分析合并为 1 句
(或: "[检测通过,无 AI 味]")
```

## 输出格式 (Word 中文)

```
[正文]
<去 AI 味后的中文段; 无 Markdown>

[修改日志]
<...>
(或: "[检测通过,无 AI 味]")
```

## 反例

- 用更高级的 AI 词替换低级 AI 词 (改成 "elucidate / encapsulate") — ❌
- 把列表压成长句但保留 "Firstly, ... Secondly, ... Thirdly, ..." — ❌
- 删完连接词后逻辑断裂 — ❌ (需要补隐含因果)
