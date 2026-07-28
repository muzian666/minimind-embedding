"""
scripts/plot_concepts.py — 核心概念原理图(让小学生也能看懂)

生成:
  images/concept_last_token_pool.png  — last-token pooling 原理
  images/concept_infonce.png          — InfoNCE 对比学习原理
  images/concept_false_neg_mask.png   — 假负样本 mask 原理
  images/concept_mrl.png              — Matryoshka 多维度原理
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def setup_zh_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC',
                  'WenQuanYi Micro Hei', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return 'DejaVu Sans'

setup_zh_font()


def draw_box(ax, x, y, w, h, text, color='#dbeafe', edge='#2563eb', fontsize=10, bold=False):
    """画一个带文字的圆角矩形。"""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fw, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#475569'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))


# ---------------------------------------------------------------------------
# 1. Last-token Pooling 原理
# ---------------------------------------------------------------------------
def plot_last_token_pool():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis('off')
    ax.set_title('Last-Token Pooling（最后 token 池化）', fontsize=15, fontweight='bold', pad=15)

    # 输入句子(token 化)
    tokens = ['天空', '为什么', '是', '蓝色', '的', '<EOS>']
    colors_tok = ['#e0e7ff']*5 + ['#fbbf24']  # EOS 高亮
    for i, (tok, c) in enumerate(zip(tokens, colors_tok)):
        draw_box(ax, 0.3 + i*1.3, 4, 1.1, 0.7, tok, color=c,
                 edge='#6366f1' if i < 5 else '#d97706', fontsize=11, bold=(i==5))

    # causal attention 箭头(每个 token 看到前面所有)
    for i in range(1, 6):
        for j in range(i):
            ax.annotate('', xy=(0.85+i*1.3, 4.55), xytext=(0.85+j*1.3, 4.55),
                        arrowprops=dict(arrowstyle='-', color='#cbd5e1', lw=0.8, connectionstyle='arc3,rad=-0.3'))

    # hidden states
    for i, tok in enumerate(tokens):
        c = '#dcfce7' if i < 5 else '#86efac'
        e = '#16a34a'
        draw_box(ax, 0.3 + i*1.3, 2.6, 1.1, 0.7,
                 f'h{i}', color=c, edge=e, fontsize=10)
        draw_arrow(ax, 0.85+i*1.3, 3.95, 0.85+i*1.3, 3.35, color='#16a34a')

    # 取最后一个(EOS 的 hidden)
    draw_box(ax, 6.8, 2.6, 1.1, 0.7, 'h5', color='#86efac', edge='#16a34a', fontsize=11, bold=True)
    draw_arrow(ax, 6.85+5*1.3-6.5, 2.95, 6.8, 2.95, color='#dc2626')
    ax.annotate('', xy=(6.8, 2.95), xytext=(0.85+5*1.3, 2.95),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.5,
                                connectionstyle='arc3,rad=-0.3'))

    # 输出向量
    draw_box(ax, 6.8, 0.8, 2.8, 0.9, '句向量 [768]\n(L2归一化)', color='#fef3c7', edge='#d97706', fontsize=11, bold=True)
    draw_arrow(ax, 7.35, 2.5, 7.35, 1.75, color='#d97706')

    # 说明文字
    ax.text(0.3, 1.7, '💡 关键洞察:EOS 之前的每个 token 都被 EOS 通过\n'
            '   causal attention "看到"了,所以 h5 浓缩了整句的语义',
            fontsize=10, color='#475569', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdf4', edgecolor='#86efac'))

    plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, 'images', 'concept_last_token_pool.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ concept_last_token_pool.png')


# ---------------------------------------------------------------------------
# 2. InfoNCE 对比学习
# ---------------------------------------------------------------------------
def plot_infonce():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('InfoNCE 对比学习:让"正样本"靠近,"负样本"远离', fontsize=14, fontweight='bold', pad=15)

    # query 在中间
    draw_box(ax, 4.2, 4.8, 1.6, 0.8, 'Query\n"天空为什么蓝"', color='#dbeafe', edge='#2563eb', fontsize=10, bold=True)

    # 正样本(近)
    draw_box(ax, 1.5, 2.5, 2.2, 0.8, '✅ 正样本(相关)\n"阳光散射使天空呈蓝色"', color='#dcfce7', edge='#16a34a', fontsize=9)
    draw_arrow(ax, 5.0, 4.7, 2.6, 3.35, color='#16a34a')
    ax.text(3.0, 4.2, '拉近\n(高相似度)', fontsize=9, color='#16a34a', ha='center',
            bbox=dict(facecolor='#dcfce7', edgecolor='none', alpha=0.8))

    # 负样本1(远)
    draw_box(ax, 6.8, 2.5, 2.5, 0.8, '❌ 负样本(无关)\n"红烧肉的做法..."', color='#fee2e2', edge='#dc2626', fontsize=9)
    draw_arrow(ax, 5.5, 4.7, 7.5, 3.35, color='#dc2626')
    ax.text(7.0, 4.2, '推远\n(低相似度)', fontsize=9, color='#dc2626', ha='center',
            bbox=dict(facecolor='#fee2e2', edgecolor='none', alpha=0.8))

    # 负样本2(批内)
    draw_box(ax, 6.8, 0.8, 2.5, 0.8, '❌ 批内负样本\n(其他query的正样本)', color='#fef3c7', edge='#d97706', fontsize=9)
    ax.annotate('', xy=(7.5, 1.65), xytext=(5.5, 4.8),
                arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.8, linestyle='--'))

    # 公式简化版
    ax.text(0.3, 0.5,
            '简化理解:loss = -log( 正样本得分 / (正样本得分 + 所有负样本得分之和) )\n'
            '目标:让正样本的得分占比尽可能接近 1(即正样本得分远大于负样本)',
            fontsize=9.5, color='#475569', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#cbd5e1'))
    ax.text(0.3, 1.85, '数学公式: $\\mathcal{L} = -\\log \\frac{e^{s(q,d^+)/\\tau}}{e^{s(q,d^+)/\\tau} + \\sum_j e^{s(q,d_j^-)/\\tau}}$',
            fontsize=10, color='#1e293b')

    plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, 'images', 'concept_infonce.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ concept_infonce.png')


# ---------------------------------------------------------------------------
# 3. 假负样本 Mask
# ---------------------------------------------------------------------------
def plot_false_neg_mask():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左:问题——假负样本
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('问题:假负样本', fontsize=13, fontweight='bold', color='#dc2626')

    draw_box(ax, 4.0, 4.8, 2.0, 0.8, 'Query\n"感冒怎么治"', color='#dbeafe', edge='#2563eb', fontsize=10, bold=True)
    draw_box(ax, 1.0, 2.8, 2.5, 0.8, '✅ 正样本\n"多休息多喝水\n7天自愈"', color='#dcfce7', edge='#16a34a', fontsize=9)
    draw_box(ax, 6.5, 2.8, 2.8, 0.8, '⚠️ 假负样本!\n"感冒需对症治疗\n注意休息"', color='#fef3c7', edge='#d97706', fontsize=9)
    draw_arrow(ax, 4.8, 4.7, 2.2, 3.65, color='#16a34a')
    draw_arrow(ax, 5.2, 4.7, 7.9, 3.65, color='#d97706')
    ax.text(5.0, 3.9, '其实也相关!', fontsize=10, color='#d97706', ha='center', fontweight='bold',
            bbox=dict(facecolor='#fef3c7', edgecolor='#d97706'))
    ax.text(0.5, 1.0, '💡 如果把这个"假负样本"当真负样本去推远,\n   模型会被误导——它在惩罚一个本该靠近的样本!',
            fontsize=9.5, color='#475569', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2', edgecolor='#fca5a5'))

    # 右:解决——mask 掉
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('解决:假负样本 Mask', fontsize=13, fontweight='bold', color='#16a34a')

    draw_box(ax, 4.0, 4.8, 2.0, 0.8, 'Query\n"感冒怎么治"', color='#dbeafe', edge='#2563eb', fontsize=10, bold=True)
    draw_box(ax, 1.0, 2.8, 2.5, 0.8, '✅ 正样本\n"多休息多喝水\n7天自愈"', color='#dcfce7', edge='#16a34a', fontsize=9)
    draw_box(ax, 6.5, 2.8, 2.8, 0.8, '🚫 屏蔽(mask=0)\n"感冒需对症治疗\n注意休息"', color='#e5e7eb', edge='#9ca3af', fontsize=9)
    draw_arrow(ax, 4.8, 4.7, 2.2, 3.65, color='#16a34a')
    # 屏蔽符号
    ax.plot([6.5, 9.3], [2.8, 3.6], color='#9ca3af', lw=2, linestyle=':')
    ax.text(7.9, 2.4, '✗ 不参与训练', fontsize=9, color='#9ca3af', ha='center')

    ax.text(0.5, 1.0,
            '💡 判定规则:如果一个"负样本"和 query 的相似度\n   甚至比正样本还高(超过 margin=0.1),\n   就判定它是假负样本,从训练中屏蔽掉',
            fontsize=9.5, color='#475569', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fdf4', edgecolor='#86efac'))

    ax.text(0.5, 0.2, '$m_{ij} = 0$ if $s_{ij} > s(q,d^+) + 0.1$, else $m_{ij} = 1$',
            fontsize=10, color='#1e293b')

    plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, 'images', 'concept_false_neg_mask.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ concept_false_neg_mask.png')


# ---------------------------------------------------------------------------
# 4. MRL Matryoshka
# ---------------------------------------------------------------------------
def plot_mrl():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5); ax.axis('off')
    ax.set_title('Matryoshka Representation Learning（俄罗斯套娃表示）', fontsize=14, fontweight='bold', pad=15)

    # 套娃图(同心圆)
    center_x, center_y = 2.5, 2.8
    dims_labels = [(2.2, '64维\n(粗筛)', '#fbbf24'), (1.7, '128维', '#f59e0b'),
                   (1.2, '256维', '#10b981'), (0.7, '512维', '#3b82f6')]
    for r, label, color in reversed(dims_labels):
        circle = plt.Circle((center_x, center_y), r, fill=False, edgecolor=color, lw=2.5)
        ax.add_patch(circle)
    # 最内层 768
    ax.plot(center_x, center_y, 'o', color='#1e293b', markersize=10)
    ax.text(center_x, center_y-0.35, '768\n(完整)', fontsize=8, ha='center', fontweight='bold')
    ax.text(center_x, center_y+2.4, '俄罗斯套娃', fontsize=9, ha='center', color='#64748b')

    # 右侧:不同维度的用途
    draw_box(ax, 5.5, 4.3, 4.0, 0.8, '768维: 精排(最准,存储最大)', color='#dbeafe', edge='#2563eb', fontsize=10)
    draw_box(ax, 5.5, 3.3, 4.0, 0.8, '512维: 平衡', color='#dbeafe', edge='#3b82f6', fontsize=10)
    draw_box(ax, 5.5, 2.3, 4.0, 0.8, '256维: 中等', color='#d1fae5', edge='#10b981', fontsize=10)
    draw_box(ax, 5.5, 1.3, 4.0, 0.8, '128维: 省存储', color='#fef3c7', edge='#f59e0b', fontsize=10)
    draw_box(ax, 5.5, 0.3, 4.0, 0.8, '64维: 粗筛(最快,最省)', color='#fef3c7', edge='#fbbf24', fontsize=10)

    for i, (yb, c) in enumerate([(4.7,'#2563eb'),(3.7,'#3b82f6'),(2.7,'#10b981'),(1.7,'#f59e0b'),(0.7,'#fbbf24')]):
        ax.annotate('', xy=(5.4, yb), xytext=(center_x+0.3+i*0.4, center_y),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.6))

    ax.text(0.3, 0.1,
            '💡 训练时对每个维度分别算 loss,所以截断到任意维度都有意义。\n'
            '   实战:先用 64 维粗筛召回,再用 768 维精排 → 又快又准',
            fontsize=9.5, color='#475569',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#cbd5e1'))

    plt.tight_layout()
    plt.savefig(os.path.join(_ROOT, 'images', 'concept_mrl.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ concept_mrl.png')


def main():
    os.makedirs(os.path.join(_ROOT, 'images'), exist_ok=True)
    print('生成核心概念原理图:')
    plot_last_token_pool()
    plot_infonce()
    plot_false_neg_mask()
    plot_mrl()
    print('\n✓ 4 张概念图已生成')


if __name__ == '__main__':
    main()
