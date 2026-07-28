"""
scripts/merge_models.py — Stage 3 模型融合(SLERP)

对齐 Qwen3-Embedding 报告的 Stage 3:对训练过程中保存的多个 checkpoint 做
球面线性插值(SLERP,Spherical Linear Interpolation)融合,提升鲁棒性。

SLERP 原理:
  对两个向量 v0, v1,按参数 t(默认0.5)在球面上插值:
    - 若 v0,v1 方向接近(夹角小): 退化为线性插值(近似平均)
    - 若 v0,v1 方向差异大: 沿大圆弧旋转,保留两者的方向特征
  对所有参数逐元素做 SLERP。多模型用两两递归 SLERP。

用法:
  # 融合 stage2 训练保存的最后 N 个 checkpoint
  python scripts/merge_models.py \\
      --checkpoints checkpoints/embedding/embedding_stage2_768_step40000.pth \\
                    checkpoints/embedding/embedding_stage2_768_step45000.pth \\
                    checkpoints/embedding/embedding_stage2_768_step50000.pth \\
      --output checkpoints/embedding/embedding_stage3_768.pth \\
      --config embed_dense_64m

  # 或用通配符自动选最后 N 个
  python scripts/merge_models.py --glob "checkpoints/embedding/embedding_stage2_768_step*.pth" \\
      --output checkpoints/embedding/embedding_stage3_768.pth --config embed_dense_64m
"""
import os
import sys
import glob
import argparse
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995):
    """球面线性插值。

    对任意形状的张量,沿 flatten 后的最后一维做 SLERP。
    当 v0/v1 余弦相似度 > dot_threshold(几乎同向)时退化为线性插值(数值稳定)。

    参数:
        t: 插值参数,0→v0, 1→v1, 0.5→中点(默认)
        v0, v1: 同形状张量
    返回:
        同形状插值结果
    """
    v0_orig_shape = v0.shape
    v0 = v0.flatten().float()
    v1 = v1.flatten().float()
    dot = torch.dot(v0, v1) / (v0.norm() * v1.norm() + 1e-12)
    if torch.abs(dot) > dot_threshold:
        # 方向几乎一致,用线性插值更稳
        out = v0 + t * (v1 - v0)
    else:
        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin((1 - t) * theta_0) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        out = s0 * v0 + s1 * v1
    return out.to(v0.dtype).view(v0_orig_shape)


def merge_two_state_dicts(sd0, sd1, t=0.5):
    """对两个 state_dict 逐 key SLERP 融合。"""
    merged = {}
    keys0 = set(sd0.keys())
    keys1 = set(sd1.keys())
    common = keys0 & keys1
    only0, only1 = keys0 - keys1, keys1 - keys0

    for k in common:
        if sd0[k].shape == sd1[k].shape and sd0[k].dtype.is_floating_point:
            merged[k] = slerp(t, sd0[k], sd1[k])
        else:
            # 形状不同或非浮点(如 buffer):取第一个
            merged[k] = sd0[k].clone()
    for k in only0:
        merged[k] = sd0[k].clone()
    for k in only1:
        merged[k] = sd1[k].clone()
    return merged


def merge_state_dicts(state_dicts, t=0.5):
    """递归两两 SLERP 融合多个 state_dict。"""
    if len(state_dicts) == 1:
        return state_dicts[0]
    print(f"  融合 {len(state_dicts)} 个 checkpoint(递归两两 SLERP, t={t})...")
    current = state_dicts[0]
    for i, nxt in enumerate(state_dicts[1:], 1):
        current = merge_two_state_dicts(current, nxt, t=t)
        print(f"    已融合 {i+1}/{len(state_dicts)}")
    return current


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 SLERP 模型融合")
    p.add_argument("--checkpoints", nargs="+", required=False,
                   help="待融合的 checkpoint 路径列表(至少2个)")
    p.add_argument("--glob", type=str, default=None,
                   help="通配符匹配 checkpoint(自动取最后 N 个)")
    p.add_argument("--last_n", type=int, default=5,
                   help="--glob 模式下取最后 N 个(按文件名 step 排序)")
    p.add_argument("--output", required=True, help="融合后的输出路径")
    p.add_argument("--t", type=float, default=0.5, help="SLERP 插值参数(0.5=等权)")
    return p.parse_args()


def main():
    args = parse_args()

    # 收集 checkpoint 路径
    if args.glob:
        paths = sorted(glob.glob(args.glob), key=lambda p: int(p.split("step")[-1].split(".")[0]) if "step" in p else 0)
        paths = paths[-args.last_n:]
    else:
        paths = args.checkpoints

    if len(paths) < 2:
        print(f"❌ 至少需要 2 个 checkpoint 来融合,当前 {len(paths)} 个")
        return

    print(f"== Stage 3 SLERP 融合 ==")
    print(f"待融合 checkpoint:")
    for p in paths:
        size_mb = os.path.getsize(p) / 1e6
        print(f"  {p}  ({size_mb:.0f} MB)")

    # 加载所有 state_dict
    state_dicts = []
    for p in paths:
        print(f"  加载 {os.path.basename(p)} ...")
        sd = torch.load(p, map_location="cpu")
        state_dicts.append(sd)

    # 递归 SLERP
    merged = merge_state_dicts(state_dicts, t=args.t)

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # 转 half 节省空间(与训练保存一致)
    merged = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in merged.items()}
    torch.save(merged, args.output)
    print(f"\n✓ 融合完成 -> {args.output} ({os.path.getsize(args.output)/1e6:.0f} MB)")
    print(f"  共 {len(merged)} 个参数")


if __name__ == "__main__":
    main()
