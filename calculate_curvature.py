import torch
from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature import sectional_curvature

# 1️⃣ 加载 Cora 数据集
features, dists, adj, labels = load_hf("cora")

# 检查特征矩阵
if features is None:
    print("⚠️ Warning: features 为 None，使用单位矩阵代替。")
    features = torch.eye(adj.shape[0])

print("✅ 数据加载成功：")
print(f"节点数: {adj.shape[0]}")
print(f"特征维度: {features.shape[1]}")
print(f"类别数: {len(torch.unique(labels))}\n")

# 2️⃣ 计算截面曲率
curvatures = sectional_curvature(adj, dists)

# 3️⃣ 输出结果
print("平均曲率:", curvatures.mean().item())
print("曲率最小值:", curvatures.min().item())
print("曲率最大值:", curvatures.max().item())

# 4️⃣ （可选）查看分布
import matplotlib.pyplot as plt

plt.hist(curvatures.numpy(), bins=30, color="skyblue", edgecolor="black")
plt.title("Cora 图节点曲率分布")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.show()
