"""
scripts/plot_moe.py — MoE 训练曲线 + Dense vs MoE 对比图

从训练日志解析 step/loss,生成:
  images/moe_training_loss.png    — MoE Stage1 + Stage2 训练曲线
  images/dense_vs_moe.png         — Dense vs MoE STS 分数对比
  images/dense_vs_moe_loss.png    — Dense vs MoE 训练 loss 对比
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


def parse_train_log(logpath):
    if not os.path.exists(logpath):
        return [], []
    steps, losses = [], []
    with open(logpath, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'step (\d+)(?:/\d+)?\s+loss\s+([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def plot_moe_training():
    """MoE Stage1 + Stage2 合并曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    s1, l1 = parse_train_log(os.path.join(_ROOT, 'results', 'moe_stage1.log'))
    if s1:
        ax1.plot(s1, l1, color='#7c3aed', linewidth=1.5, alpha=0.85)
        ax1.set_title('MoE Stage 1: Weakly-Supervised', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.annotate(f'{l1[0]:.2f}', (s1[0], l1[0]), fontsize=9, color='#7c3aed')
        ax1.annotate(f'{l1[-1]:.2f}', (s1[-1], l1[-1]), fontsize=9, color='#7c3aed', fontweight='bold')

    s2, l2 = parse_train_log(os.path.join(_ROOT, 'results', 'moe_stage2_v2.log'))
    if s2:
        ax2.plot(s2, l2, color='#9333ea', linewidth=1.2, alpha=0.85)
        ax2.set_title('MoE Stage 2: Supervised (3 epochs, +MRL)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        for e in [21256, 42512]:
            if e < s2[-1]:
                ax2.axvline(x=e, color='gray', linestyle=':', alpha=0.4)
        ax2.annotate(f'{l2[0]:.2f}', (s2[0], l2[0]), fontsize=9, color='#9333ea')
        ax2.annotate(f'{l2[-1]:.2f}', (s2[-1], l2[-1]), fontsize=9, color='#9333ea', fontweight='bold')

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'moe_training_loss.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath}')


def plot_dense_vs_moe_sts():
    """Dense vs MoE STS 分数对比。"""
    tasks = ['ATEC', 'BQ', 'LCQMC', 'STSB']
    # Dense
    d = json.load(open(os.path.join(_ROOT, 'results', 'sts_stage3.json')))['scores']
    dense = [d[t] for t in tasks]
    # MoE
    m = json.load(open(os.path.join(_ROOT, 'results', 'sts_moe_stage3.json')))['scores']
    moe = [m[t] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, dense, width, label='Dense (64M)', color='#2563eb', alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, moe, width, label='MoE (198M-A64M)', color='#9333ea', alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Dense vs MoE: C-MTEB STS Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, v in zip(bars1, dense):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    for bar, v in zip(bars2, moe):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

    # 平均线
    ax.axhline(y=sum(dense)/4, color='#2563eb', linestyle='--', alpha=0.4)
    ax.axhline(y=sum(moe)/4, color='#9333ea', linestyle='--', alpha=0.4)
    ax.text(3.4, sum(dense)/4, f'avg {sum(dense)/4:.3f}', fontsize=8, color='#2563eb', va='bottom')
    ax.text(3.4, sum(moe)/4, f'avg {sum(moe)/4:.3f}', fontsize=8, color='#9333ea', va='bottom')

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'dense_vs_moe.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath}')


def plot_dense_vs_moe_loss():
    """Dense vs MoE Stage2 loss 对比(同一图)。"""
    sd, ld = parse_train_log(os.path.join(_ROOT, 'results', 'stage2_FINAL.log'))
    sm, lm = parse_train_log(os.path.join(_ROOT, 'results', 'moe_stage2_v2.log'))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if sd:
        ax.plot(sd, ld, color='#2563eb', linewidth=1.3, alpha=0.8, label='Dense (64M)')
    if sm:
        ax.plot(sm, lm, color='#9333ea', linewidth=1.3, alpha=0.8, label='MoE (198M-A64M)')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Dense vs MoE: Stage 2 Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # epoch 边界
    for e in [21256, 42512]:
        ax.axvline(x=e, color='gray', linestyle=':', alpha=0.3)
        ax.text(e, ax.get_ylim()[1]*0.95, f'epoch {e//21256}', fontsize=8, color='gray', ha='center')

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'dense_vs_moe_loss.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath}')


def main():
    os.makedirs(os.path.join(_ROOT, 'images'), exist_ok=True)
    print('生成 MoE 图表:')
    plot_moe_training()
    print('生成对比图:')
    plot_dense_vs_moe_sts()
    plot_dense_vs_moe_loss()
    print('\n✓ 完成')


if __name__ == '__main__':
    main()
