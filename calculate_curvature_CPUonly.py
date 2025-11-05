import torch
import time
import matplotlib.pyplot as plt
from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature_0 import sectional_curvature

# 仅使用 CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# =========================================================
# 加载数据集：切换到 PubMed
# =========================================================
# features, dists, adj, labels = load_hf("cora")  #  Cora
features, dists, adj, labels = load_hf("pubmed")  # 改为 PubMed

# 检查特征矩阵
if features is None:
    print(" Warning: features 为 None，使用单位矩阵代替。")
    features = torch.eye(adj.shape[0])

print(" 数据加载成功：")
print(f"节点数: {adj.shape[0]}")
print(f"特征维度: {features.shape[1]}")
print(f"类别数: {len(torch.unique(labels))}\n")

# 开始计时
start_time = time.time()

# 计算截面曲率（CPU）
curvatures = sectional_curvature(adj, dists)

# 结束计时
end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ 曲率计算耗时: {elapsed:.2f} 秒\n")

# 输出结果
print("平均曲率:", curvatures.mean().item())
print("曲率最小值:", curvatures.min().item())
print("曲率最大值:", curvatures.max().item())

# 查看分布
plt.hist(curvatures.numpy(), bins=30, color="skyblue", edgecolor="black")
plt.title("PubMed 图节点曲率分布")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.show()