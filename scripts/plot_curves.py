"""
scripts/plot_curves.py — 从训练日志生成图表(替代 wandb private 链接)

生成:
  images/stage1_loss.png       — Stage1 弱监督 loss 曲线
  images/stage2_loss.png       — Stage2 监督 loss 曲线(3 epoch)
  images/full_training_loss.png — Stage1+Stage2 合并总览
  images/sts_scores.png        — C-MTEB STS 各任务分数
  images/model_comparison.png  — 与 BGE/Qwen3 对比
"""
import os
import re
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 中文字体配置(尝试常见中文字体,失败则用英文)
def setup_zh_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC',
                  'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'

USED_FONT = setup_zh_font()
print(f"使用字体: {USED_FONT}")


def parse_train_log(logpath):
    """从训练日志解析 (step, loss, lr) 列表。"""
    if not os.path.exists(logpath):
        return [], [], []
    steps, losses, lrs = [], [], []
    with open(logpath, encoding='utf-8', errors='replace') as f:
        for line in f:
            # 格式: epoch 0 step 200/63768 loss 1.6262 lr 1.00e-05
            m = re.search(r'step (\d+)(?:/\d+)?\s+loss\s+([\d.]+)\s+lr\s+([\d.e+-]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
    return steps, losses, lrs


def plot_single_loss(steps, losses, title, outpath, color='#2563eb', lr=None):
    """画单阶段 loss 曲线(可选 lr 双轴)。"""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(steps, losses, color=color, linewidth=1.5, alpha=0.85)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if lr:
        ax2 = ax1.twinx()
        ax2.plot(steps, lr, color='#dc2626', linewidth=1, alpha=0.5, linestyle='--', label='lr')
        ax2.set_ylabel('Learning Rate', fontsize=11, color='#dc2626')
        ax2.tick_params(axis='y', labelcolor='#dc2626')
    # 标注起止 loss
    if len(losses) > 1:
        ax1.annotate(f'{losses[0]:.2f}', (steps[0], losses[0]), textcoords="offset points",
                     xytext=(5, 8), fontsize=9, color=color)
        ax1.annotate(f'{losses[-1]:.2f}', (steps[-1], losses[-1]), textcoords="offset points",
                     xytext=(5, -12), fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_combined_loss():
    """Stage1 + Stage2 合并总览(分两个子图)。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Stage 1
    s1, l1, _ = parse_train_log(os.path.join(_ROOT, 'results', 'stage1.log'))
    if s1:
        ax1.plot(s1, l1, color='#2563eb', linewidth=1.5, alpha=0.85)
        ax1.set_title('Stage 1: Weakly-Supervised Pre-training', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.annotate(f'start {l1[0]:.2f}', (s1[0], l1[0]), fontsize=9, color='#2563eb')
        ax1.annotate(f'end {l1[-1]:.2f}', (s1[-1], l1[-1]), fontsize=9, color='#2563eb', fontweight='bold')

    # Stage 2
    s2, l2, _ = parse_train_log(os.path.join(_ROOT, 'results', 'stage2_FINAL.log'))
    if s2:
        ax2.plot(s2, l2, color='#16a34a', linewidth=1.2, alpha=0.85)
        ax2.set_title('Stage 2: Supervised Fine-tuning (3 epochs, +MRL)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        # 标注 epoch 边界(每 21256 步一个 epoch)
        for e in [21256, 42512]:
            if e < s2[-1]:
                ax2.axvline(x=e, color='gray', linestyle=':', alpha=0.4)
                ax2.text(e, max(l2)*0.95, f'epoch {e//21256}', fontsize=8, color='gray', ha='center')
        ax2.annotate(f'start {l2[0]:.2f}', (s2[0], l2[0]), fontsize=9, color='#16a34a')
        ax2.annotate(f'end {l2[-1]:.2f}', (s2[-1], l2[-1]), fontsize=9, color='#16a34a', fontweight='bold')

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'full_training_loss.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_sts_scores():
    """C-MTEB STS 各任务分数柱状图。"""
    path = os.path.join(_ROOT, 'results', 'sts_stage3.json')
    if not os.path.exists(path):
        return
    data = json.load(open(path))['scores']
    tasks = ['ATEC', 'BQ', 'LCQMC', 'STSB']
    scores = [data[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#dc2626' if s < 0.4 else '#f59e0b' if s < 0.6 else '#16a34a' for s in scores]
    bars = ax.bar(tasks, scores, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('C-MTEB STS Evaluation (Stage 3 merged)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{s:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=sum(scores)/len(scores), color='blue', linestyle='--', alpha=0.5, label=f'Average: {sum(scores)/len(scores):.3f}')
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'sts_scores.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_model_comparison():
    """与其他模型对比的柱状图。"""
    models = ['MiniMind-Embed\n(this, 64M)', 'BGE-small-zh\n(24M)', 'BGE-large-zh\n(326M)', 'Qwen3-Embed-0.6B\n(600M)']
    scores = [0.478, 0.60, 0.66, 0.66]  # approximate reference values
    colors = ['#2563eb', '#94a3b8', '#94a3b8', '#94a3b8']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(models, scores, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('C-MTEB STS Average (Spearman)', fontsize=12)
    ax.set_title('Model Comparison on C-MTEB STS', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{s:.2f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'model_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def main():
    os.makedirs(os.path.join(_ROOT, 'images'), exist_ok=True)
    print("生成训练曲线图:")

    # Stage1
    s1, l1, lr1 = parse_train_log(os.path.join(_ROOT, 'results', 'stage1.log'))
    if s1:
        plot_single_loss(s1, l1, 'Stage 1: Weakly-Supervised Pre-training (InfoNCE)',
                         os.path.join(_ROOT, 'images', 'stage1_loss.png'), color='#2563eb', lr=lr1)

    # Stage2
    s2, l2, lr2 = parse_train_log(os.path.join(_ROOT, 'results', 'stage2_FINAL.log'))
    if s2:
        plot_single_loss(s2, l2, 'Stage 2: Supervised Fine-tuning (3 epochs, InfoNCE+mask+MRL)',
                         os.path.join(_ROOT, 'images', 'stage2_loss.png'), color='#16a34a', lr=lr2)

    # 合并
    print("生成合并图:")
    plot_combined_loss()

    # STS 分数
    print("生成评测图:")
    plot_sts_scores()
    plot_model_comparison()

    print("\n✓ 所有图表已生成到 images/")


if __name__ == '__main__':
    main()
