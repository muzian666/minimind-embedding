"""
scripts/plot_rerank.py — Rerank 训练曲线 + MAP@10 演进图

从 rerank_noleak.log 解析 step/loss/acc,生成:
  images/rerank_training.png   — Rerank 训练 loss + accuracy 双轴曲线
  images/rerank_map_evolution.png — MAP@10 演进历程(三个 bug 修复)
"""
import os
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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


def parse_rerank_log(logpath):
    """解析 rerank 训练日志:step/loss/acc/lr。"""
    if not os.path.exists(logpath):
        return [], [], [], []
    steps, losses, accs, lrs = [], [], [], []
    with open(logpath, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'step (\d+)(?:/\d+)?\s+loss\s+([\d.]+)\s+acc\s+([\d.]+)\s+lr\s+([\d.e+-]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                accs.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    return steps, losses, accs, lrs


def plot_rerank_training():
    """Rerank loss + accuracy 双轴训练曲线。"""
    s, l, a, lr = parse_rerank_log(os.path.join(_ROOT, 'results', 'rerank_noleak.log'))
    if not s:
        print('  ⚠ 无 rerank 日志数据')
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    # loss
    ax1.plot(s, l, color='#dc2626', linewidth=1.5, alpha=0.85, label='Loss')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='#dc2626')
    ax1.tick_params(axis='y', labelcolor='#dc2626')
    ax1.set_title('Rerank Training: Loss & Accuracy (3 epochs, no data leakage)',
                   fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # accuracy 双轴
    ax2 = ax1.twinx()
    ax2.plot(s, a, color='#2563eb', linewidth=1.5, alpha=0.85, label='Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=12, color='#2563eb')
    ax2.tick_params(axis='y', labelcolor='#2563eb')
    ax2.set_ylim(0.4, 1.0)

    # epoch 分界
    total = s[-1]
    for i in range(1, 3):
        boundary = total * i // 3
        if boundary < total:
            ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.4)
            ax1.text(boundary, max(l)*0.95, f'epoch {i}', fontsize=8, color='gray', ha='center')

    # 标注起止
    ax1.annotate(f'{l[0]:.2f}', (s[0], l[0]), fontsize=9, color='#dc2626')
    ax1.annotate(f'{l[-1]:.2f}', (s[-1], l[-1]), fontsize=9, color='#dc2626', fontweight='bold')
    ax2.annotate(f'{a[0]:.1%}', (s[0], a[0]), fontsize=9, color='#2563eb')
    ax2.annotate(f'{a[-1]:.1%}', (s[-1], a[-1]), fontsize=10, color='#2563eb', fontweight='bold')

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'rerank_training.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ rerank_training.png')


def plot_map_evolution():
    """MAP@10 演进历程:从 bug 到修复。"""
    # 演进数据(从 results JSON 和 README 的三个 bug 故事)
    stages = [
        ('Bug 1\n(label映射\n反转)', 0.24, '#ef4444', 'rerank_scores_v1_trained'),
        ('Bug 2\n(新增token\n随机初始化)', 0.24, '#f97316', 'rerank_scores_v1_trained'),
        ('Bug 3\n(数据泄露\n→修复切分)', 0.94, '#eab308', 'rerank_full_fixed'),
        ('最终\n(无泄露\n3epoch)', 0.915, '#22c55e', 'rerank_noleak_stage3'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    bars = ax.bar(range(len(stages)), values, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, width=0.6)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('T2Reranking MAP@10', fontsize=12)
    ax.set_title('Rerank MAP@10 Evolution: Three Bugs → 0.915 (+94%)',
                  fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

    # 连线箭头表示演进
    for i in range(len(stages)-1):
        ax.annotate('', xy=(i+1, values[i+1]+0.05), xytext=(i, values[i]+0.05),
                     arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.5))

    # 零样本 baseline 线
    ax.axhline(y=0.47, color='#94a3b8', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(3.4, 0.47, 'zero-shot\nbaseline\n0.47', fontsize=8, color='#64748b', va='center')

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'rerank_map_evolution.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ rerank_map_evolution.png')


def main():
    os.makedirs(os.path.join(_ROOT, 'images'), exist_ok=True)
    print('生成 Rerank 图表:')
    plot_rerank_training()
    plot_map_evolution()
    print('\n✓ 完成')


if __name__ == '__main__':
    main()
