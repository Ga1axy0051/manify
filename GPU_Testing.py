import torch
import time
import matplotlib.pyplot as plt
from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature import sectional_curvature

# =========================================================
# 自动选择设备
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")

# =========================================================
# 加载数据集
# =========================================================
features, dists, adj, labels = load_hf("pubmed")

if features is None:
    print("Warning: features 为 None，使用单位矩阵代替。")
    features = torch.eye(adj.shape[0])

print(f"原始图节点数: {adj.shape[0]}")

# =========================================================
# 截取前10000个节点子图
# =========================================================
n_test = 10000
adj_sub = adj[:n_test, :n_test]
dists_sub = dists[:n_test, :n_test]
features_sub = features[:n_test]

print(f"测试子图节点数: {n_test}")

# =========================================================
# 移动到设备
# =========================================================
adj_sub = adj_sub.to(device)
dists_sub = dists_sub.to(device)

# =========================================================
# 开始计算
# =========================================================
print("\n开始计算截面曲率（仅前10000个节点）...")
start_time = time.time()

curvatures = sectional_curvature(adj_sub, dists_sub, device=device, show_progress=True)

end_time = time.time()
elapsed = end_time - start_time
print(f"✅ 10000节点测试完成，用时 {elapsed:.2f} 秒\n")

# =========================================================
# 输出结果
# =========================================================
curvatures_cpu = curvatures.detach().cpu()
print(" 距离矩阵统计：")
print("最小值:", torch.min(dists))
print("最大值:", torch.max(dists))
print("是否包含 inf:", torch.isinf(dists).any().item())
print("是否包含 nan:", torch.isnan(dists).any().item())
print("平均值:", torch.mean(dists))


# =========================================================
# 绘制曲率分布直方图
# =========================================================
plt.figure(figsize=(6, 4))
plt.hist(curvatures_cpu.numpy(), bins=20, color="salmon", edgecolor="black")
plt.title("Sectional Curvature Distribution (PubMed 前100节点)")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()
