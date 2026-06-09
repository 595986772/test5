"""
生成 "研究问题 → 研究内容" 对照图 (中期 img.png 同款排版).

输出: draw/research_problem_content.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# Windows 下中文字体回退链
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


TITLE = '边缘环境\n多目标\n卸载策略'

ROWS = [
    {
        'problem': (
            '现有方法大多采用固定偏好权重进行训练,当用户偏好发生变化时\n'
            '需要重新训练模型,难以适应多样化场景需求,尚未实现\n'
            '“一次训练、多偏好适配”。'
        ),
        'content': (
            '提出偏好条件化的反馈扩散策略网络,将用户偏好向量编码为策略\n'
            '输入,结合 ω 循环训练机制,使单一策略适应不同偏好需求,\n'
            '推理阶段仅需调整输入偏好即可覆盖完整 Pareto 前沿。'
        ),
    },
    {
        'problem': (
            '时延奖励与能耗奖励通常存在较大的量级差异,部分情况下可\n'
            '相差二十倍以上。在传统多目标强化学习框架下,时延目标容易\n'
            '占据主导,导致 Pareto 前沿在能耗维度上分布不均衡。'
        ),
        'content': (
            '设计向量化 Critic 网络结构,结合冲突目标正则化与自适应\n'
            '量级归一化方法,对不同目标分别评估,缓解多目标优化中因\n'
            '奖励尺度差异导致的梯度失衡,提高优化稳定性与效果。'
        ),
    },
    {
        'problem': (
            '现有边缘卸载研究多聚焦时延—能耗双目标,但边缘 AI 推理中\n'
            '推理精度同样关键(低分辨率模型省时但损精度,云端高精度模型\n'
            '通信开销高),三目标联合且单策略覆盖 Pareto 的工作仍不足。'
        ),
        'content': (
            '提出时延—能耗—精度三目标联合优化框架,在动作空间中引入\n'
            '推理保真度等级,将卸载决策由“选服务器”扩展为\n'
            '“服务器 + 推理精度等级”联合决策,统一建模 Pareto 曲面。'
        ),
    },
]


def _to_full_width(text):
    """把字符串里的西文标点(逗号/分号/冒号/括号/感叹号/问号)替换成中文形式."""
    table = str.maketrans({
        ',': ',',
        ';': ';',
        ':': ':',
        '(': '(',
        ')': ')',
        '!': '!',
        '?': '?',
    })
    return text.translate(table)


ROWS = [
    {k: _to_full_width(v) for k, v in row.items()}
    for row in ROWS
]


def add_box(ax, x, y, w, h, text, facecolor='#F4F7FB', edgecolor='#2B6CB0',
            fontsize=10, text_color='#1A202C', text_kwargs=None,
            align='left', pad_left=0.18):
    """画一个圆角矩形 + 文本. align='left' 左对齐, 'center' 居中
    (两种模式 va 都是 center)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.04',
        linewidth=1.2, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(box)
    if align == 'center':
        tx, ha = x + w / 2, 'center'
    else:
        tx, ha = x + pad_left, 'left'
    kw = dict(ha=ha, va='center', fontsize=fontsize,
              color=text_color, linespacing=1.5)
    if text_kwargs:
        kw.update(text_kwargs)
    ax.text(tx, y + h / 2, text, **kw)


def add_dashed(ax, x1, y1, x2, y2, color='#4A5568'):
    ax.add_line(Line2D([x1, x2], [y1, y2], linestyle=(0, (3, 3)),
                       color=color, linewidth=1.2))


def main():
    fig_w, fig_h = 18, 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 18); ax.set_ylim(0, 9)
    ax.set_aspect('auto'); ax.axis('off')

    # ---- 左侧标题竖条 (居中) ----
    title_x, title_y, title_w, title_h = 0.4, 0.6, 1.7, 7.8
    add_box(ax, title_x, title_y, title_w, title_h, TITLE,
            facecolor='#2B6CB0', edgecolor='#1A365D', fontsize=18,
            text_color='white', align='center',
            text_kwargs=dict(weight='bold'))

    # ---- 列标题 (居中) ----
    header_y = 7.4
    header_h = 0.75
    prob_x, prob_w = 3.5, 5.5
    cont_x, cont_w = 9.6, 5.5
    add_box(ax, prob_x, header_y, prob_w, header_h, '研究问题',
            facecolor='#2B6CB0', edgecolor='#1A365D',
            fontsize=20, text_color='white', align='center',
            text_kwargs=dict(weight='bold'))
    add_box(ax, cont_x, header_y, cont_w, header_h, '研究内容',
            facecolor='#22543D', edgecolor='#1C3D2A',
            fontsize=20, text_color='white', align='center',
            text_kwargs=dict(weight='bold'))

    # 顶部连线: 标题 → 研究问题 → 研究内容
    header_cy = header_y + header_h / 2
    ax.add_line(Line2D([title_x + title_w, prob_x], [header_cy, header_cy],
                       color='#1A365D', linewidth=1.6))
    ax.add_line(Line2D([prob_x + prob_w, cont_x], [header_cy, header_cy],
                       color='#1A365D', linewidth=1.6))

    # ---- 三行问题/内容 (固定宽度, 左对齐, 右侧留白) ----
    row_h = 1.9
    gap = 0.35
    top = header_y - 0.45

    for i, row in enumerate(ROWS):
        y = top - (i + 1) * row_h - i * gap
        add_box(ax, prob_x, y, prob_w, row_h, row['problem'],
                facecolor='#EBF4FF', edgecolor='#2B6CB0', fontsize=13)
        add_box(ax, cont_x, y, cont_w, row_h, row['content'],
                facecolor='#F0FFF4', edgecolor='#22543D', fontsize=13)

        cy = y + row_h / 2
        # 标题 → 问题框 (虚线)
        add_dashed(ax, title_x + title_w, cy, prob_x, cy, color='#2B6CB0')
        # 问题框 → 内容框 (虚线)
        add_dashed(ax, prob_x + prob_w, cy, cont_x, cy, color='#4A5568')

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(out_dir, 'research_problem_content')
    # 同时输出 PNG (位图预览) / SVG (Inkscape/Illustrator 编辑) / PDF (论文嵌入)
    plt.savefig(base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(base + '.svg',          bbox_inches='tight', facecolor='white')
    plt.savefig(base + '.pdf',          bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {base}.png / .svg / .pdf')


if __name__ == '__main__':
    main()
