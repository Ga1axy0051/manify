import torch
from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature import sectional_curvature
from manify.curvature_estimation.sectional_curvature_strict_fast import sectional_curvature_strict_fast

# 加载数据集
feats, dists, adj, _ = load_hf("pubmed")
adj, dists = adj.cuda(), dists.cuda()

# 运行原版
print("▶ Running original version...")
res1 = sectional_curvature(adj, dists, device="cuda", show_progress=False)

# 运行fast版本
print("▶ Running fast version...")
res2 = sectional_curvature_strict_fast(adj, dists, device="cuda", show_progress=False)

# 对比
diff = (res1 - res2).abs()
print("\n==== 数值对比结果 ====")
print(f"平均差异: {diff.mean().item():.6e}")
print(f"最大差异: {diff.max().item():.6e}")
print(f"结果一致比例: {torch.sum(diff < 1e-5).item() / diff.numel() * 100:.2f}%")
