import torch
import time
from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature_0 import sectional_curvature
import matplotlib.pyplot as plt

# 固定使用 CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# 加载 Cora 数据集
features, dists, adj, labels = load_hf("cora")

start_time = time.time()

# 只取前100个节点测试
n_test = 100
adj_small = adj[:n_test, :n_test]
dists_small = dists[:n_test, :n_test]

print(f"✅ 使用前 {n_test} 个节点进行快速测试")
print(f"邻接矩阵大小: {adj_small.shape}")
print(f"距离矩阵大小: {dists_small.shape}")

# 计算截面曲率
print("\n⏳ 正在CPU上计算曲率，请稍候...")
curvatures = sectional_curvature(adj_small, dists_small)

#  结束计时
end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ 曲率计算耗时: {elapsed:.2f} 秒\n")

# 输出结果
print("\n✅ 曲率计算完成：")
print(f"平均曲率: {curvatures.mean().item():.6f}")
print(f"最小曲率: {curvatures.min().item():.6f}")
print(f"最大曲率: {curvatures.max().item():.6f}")

# 可视化分布
plt.hist(curvatures.numpy(), bins=20, color="lightcoral", edgecolor="black")
plt.title("Cora (100 nodes) 曲率分布 (CPU计算)")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.show()