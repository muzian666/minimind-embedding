"""
scripts/plot_rerank_moe.py — Rerank Dense vs MoE 对比 + MoE 训练曲线
"""
import os, re, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def setup_zh_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_zh_font()

def parse_rerank_log(logpath):
    if not os.path.exists(logpath): return [], [], []
    steps, losses, accs = [], [], []
    with open(logpath, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'step (\d+)(?:/\d+)?\s+loss\s+([\d.]+)\s+acc\s+([\d.]+)', line)
            if m:
                steps.append(int(m.group(1))); losses.append(float(m.group(2))); accs.append(float(m.group(3)))
    return steps, losses, accs


def plot_rerank_dense_vs_moe():
    """Rerank Dense vs MoE MAP@10 + accuracy 对比。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # MAP@10
    labels = ['Dense\n(64M)', 'MoE\n(198M-A64M)']
    d_json = json.load(open(os.path.join(_ROOT, 'results', 'rerank_noleak_stage3.json')))
    m_json = json.load(open(os.path.join(_ROOT, 'results', 'rerank_moe_noleak.json')))
    map_scores = [d_json['T2Reranking_MAP@10'], m_json['T2Reranking_MAP@10']]
    colors = ['#2563eb', '#9333ea']
    bars = ax1.bar(labels, map_scores, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.5)
    ax1.set_ylabel('MAP@10', fontsize=12)
    ax1.set_title('Rerank: Dense vs MoE (MAP@10)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.2, axis='y')
    for bar, v in zip(bars, map_scores):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')

    # Training accuracy 对比
    sd, ld, ad = parse_rerank_log(os.path.join(_ROOT, 'results', 'rerank_noleak.log'))
    sm, lm, am = parse_rerank_log(os.path.join(_ROOT, 'results', 'moe_rerank.log'))
    if sd: ax2.plot(sd, ad, color='#2563eb', linewidth=1.3, alpha=0.8, label='Dense')
    if sm: ax2.plot(sm, am, color='#9333ea', linewidth=1.3, alpha=0.8, label='MoE')
    ax2.set_xlabel('Step', fontsize=12); ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Rerank Training Accuracy: Dense vs MoE', fontsize=13, fontweight='bold')
    ax2.set_ylim(0.4, 1.0); ax2.legend(fontsize=11); ax2.grid(True, alpha=0.2)
    for e in [5000, 10000]:
        ax2.axvline(x=e, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(_ROOT, 'images', 'rerank_dense_vs_moe.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  ✓ rerank_dense_vs_moe.png')


def main():
    os.makedirs(os.path.join(_ROOT, 'images'), exist_ok=True)
    print('生成 Rerank Dense vs MoE 对比图:')
    plot_rerank_dense_vs_moe()
    print('✓ 完成')

if __name__ == '__main__':
    main()
