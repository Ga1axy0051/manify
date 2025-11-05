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
features, dists, adj, labels = load_hf("cora")  # 小图测试
#features, dists, adj, labels = load_hf("pubmed")  # 改为 PubMed

if features is None:
    print("Warning: features 为 None，使用单位矩阵代替。")
    features = torch.eye(adj.shape[0])

print("数据加载成功：")
print(f"节点数: {adj.shape[0]}")
print(f"特征维度: {features.shape[1]}")
print(f"类别数: {len(torch.unique(labels))}\n")

# =========================================================
# 移动数据到设备
# =========================================================
adj = adj.to(device)
dists = dists.to(device)

# =========================================================
# 开始计时 + 曲率计算
# =========================================================
print("开始计算截面曲率...")
start_time = time.time()

try:
    # 调用内部带进度条的 GPU 版本函数
    curvatures = sectional_curvature(adj, dists, device=device, show_progress=True)

except RuntimeError as e:
    if "CUDA" in str(e):
        print("\nGPU 显存不足，自动切换到 CPU 重新计算。")
        torch.cuda.empty_cache()
        adj = adj.cpu()
        dists = dists.cpu()
        curvatures = sectional_curvature(adj, dists, device="cpu", show_progress=True)
    else:
        raise e

end_time = time.time()
elapsed = end_time - start_time
print(f"\n曲率计算总耗时: {elapsed / 60:.2f} 分钟\n")

# =========================================================
# 输出结果
# =========================================================
curvatures_cpu = curvatures.detach().cpu()
print("平均曲率:", curvatures_cpu.mean().item())
print("最小值:", curvatures_cpu.min().item())
print("最大值:", curvatures_cpu.max().item())

# =========================================================
# 绘制曲率分布直方图
# =========================================================
plt.figure(figsize=(8, 5))
plt.hist(curvatures_cpu.numpy(), bins=30, color="skyblue", edgecolor="black")
plt.title("PubMed 图节点曲率分布 (GPU 自动进度监控)")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()
