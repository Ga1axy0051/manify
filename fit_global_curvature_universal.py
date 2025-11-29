# manify/fit_global_curvature_universal.py
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import deque
from manify.utils.dataloaders import load_hf


# --------------------------------------------------------
# 正确 Poincaré Ball 距离（Nickel & Kiela 2017）
# --------------------------------------------------------
def poincare_dist(x, y, c):
    # c > 0, curvature = -c
    c = torch.clamp(c, min=1e-9)
    sqrt_c = torch.sqrt(c)

    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    norm_xy = torch.norm(x - y, dim=-1)

    denom = (1 - c * x2) * (1 - c * y2)
    argument = 1 + 2 * c * norm_xy**2 / denom
    argument = torch.clamp(argument, min=1 + 1e-7)

    return torch.acosh(argument) / sqrt_c  # ★ 正确公式


# --------------------------------------------------------
# 拟合全局曲率
# --------------------------------------------------------
def fit_global_curvature(dataset_name="cora", sub_n=2000, max_dist=10, epochs=200, lr=0.01):

    print(f"\n 加载数据集：{dataset_name}")
    features, dists, adj, labels = load_hf(dataset_name)
    N = adj.shape[0]

    print(f"✔ 节点数: {N}, 特征维度: {None if features is None else features.shape[1]}")


    # --------------------------------------------------------
    # Step 1 — 子图采样
    # --------------------------------------------------------
    sub_nodes = torch.randperm(N)[:sub_n].tolist()
    print(f" 子图大小 sub_n = {sub_n}")


    # --------------------------------------------------------
    # Step 2 — BFS shortest path
    # --------------------------------------------------------
    print(f"\n BFS 最短路 (max_dist={max_dist}) ...")

    D = torch.full((sub_n, sub_n), max_dist, dtype=torch.float32)

    for i, src in enumerate(tqdm(sub_nodes)):
        visited = {src: 0}
        q = deque([src])

        while q:
            u = q.popleft()
            if visited[u] >= max_dist:
                continue

            neighbors = torch.nonzero(adj[u] > 0, as_tuple=False).flatten().tolist()
            for v in neighbors:
                if v not in visited:
                    visited[v] = visited[u] + 1
                    q.append(int(v))

        # 写入 dist matrix
        for j, node in enumerate(sub_nodes):
            if node in visited:
                D[i, j] = visited[node]

    print("✔ BFS 结束")

    # pair selection
    mask = (D > 0) & (D < max_dist)
    pairs = mask.nonzero(as_tuple=False).long()
    print(f"✔ 有效 (i,j) pair 数: {pairs.shape[0]:,}")


    # --------------------------------------------------------
    # Step 3 — 迁移到 GPU（必须在这里！！）
    # --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n 使用设备: {device}")

    D = D.to(device)
    pairs = pairs.to(device)


    # --------------------------------------------------------
    # Step 4 — 初始化 Poincaré embedding + curvature
    # --------------------------------------------------------
    emb_dim = 10
    X = torch.nn.Parameter(torch.randn(sub_n, emb_dim, device=device) * 0.01)
    curv_param = torch.nn.Parameter(torch.tensor(1.0, device=device))

    optim = torch.optim.Adam([X, curv_param], lr=lr)

    print("\n 开始训练 ...")

    I = pairs[:, 0]
    J = pairs[:, 1]


    # --------------------------------------------------------
    # Step 5 — 训练迭代
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):
        optim.zero_grad()

        c = torch.relu(curv_param) + 1e-6
        dij_pred = poincare_dist(X[I], X[J], c)
        dij_true = D[I, J]

        loss = (dij_pred - dij_true).pow(2).mean()
        loss.backward()
        optim.step()

        K = -float(c.item())

        print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | 估计曲率 K ≈ {K:.4f}")

    print(f"\n 最终估计曲率 K ≈ {K:.6f}")
    return K



if __name__ == "__main__":
    fit_global_curvature(dataset_name="cora")
