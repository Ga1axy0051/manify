#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在 WordNet 图上拟合 Poincaré embedding，并学习一个全局曲率 K（负数）。
思路：
  1. 从 wordnet_graph.pt 读取稀疏邻接矩阵（closure 图）
  2. 抽取若干个节点组成子图（例如 2000 个节点）
  3. 在子图上计算 BFS 最短路距离 D_sub
  4. 在 Poincaré 球模型中学习：
        - 节点嵌入 emb[i] ∈ H^d
        - 曲率参数 c > 0 （K = -c）
     使得双曲距离 d_c(emb_i, emb_j) 拟合图距离 D_sub(i,j)
"""

import math
import random
from collections import deque

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


# ===========================
# 1. 读取 WordNet 图
# ===========================

# 按照你 manify 的路径来
PATH = r"./data/wordnet_graph.pt"

print(f"📘 加载 WordNet 图: {PATH}")
data = torch.load(PATH)
adj = data["adj_sparse"].coalesce()   # 稀疏邻接矩阵
N = adj.size(0)
print(f"✔ Loaded WordNet graph: N={N}, nnz={adj._nnz()}")


# ===========================
# 2. 抽取子图，计算最短路距离
# ===========================

SUB_N = 1000      # 子图节点数，可以改大/小，比如 1000, 3000
MAX_DIST = 10     # BFS 最大深度，只关心这个范围内的距离

random.seed(0)
sub_nodes = sorted(random.sample(range(N), SUB_N))
sub_id_map = {old: i for i, old in enumerate(sub_nodes)}

# 构造邻接表（无向）
row, col = adj.indices()
neighbors = [[] for _ in range(N)]
for u, v in zip(row.tolist(), col.tolist()):
    neighbors[u].append(v)
    neighbors[v].append(u)


def bfs_dists(start, max_dist=MAX_DIST):
    """从 start 做 BFS，返回 {node:dist}，只搜到 max_dist。"""
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if dist[u] >= max_dist:
            continue
        for v in neighbors[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


print(f" 在子图上计算最短路距离矩阵 (SUB_N={SUB_N}, MAX_DIST={MAX_DIST}) ...")
D_sub = torch.full((SUB_N, SUB_N), float("inf"), dtype=torch.float32)
for i, u in enumerate(tqdm(sub_nodes, desc="BFS from sub-nodes", ncols=90)):
    dist_map = bfs_dists(u, max_dist=MAX_DIST)
    for v, d in dist_map.items():
        j = sub_id_map.get(v, None)
        if j is not None:
            D_sub[i, j] = float(d)
D_sub.fill_diagonal_(0.0)

# 只保留有限且 > 0 的距离对
pairs = torch.nonzero((D_sub > 0) & torch.isfinite(D_sub), as_tuple=False)
true_dists = D_sub[pairs[:, 0], pairs[:, 1]]
print(f"✔ 可用的 (i,j) pair 数量: {pairs.shape[0]}")


# ===========================
# 3. 定义带可学习曲率的 Poincaré 嵌入
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" 使用设备:", DEVICE)

dim = 5  # 嵌入维度，可以改：2 / 5 / 10 / 20 等
emb = nn.Embedding(SUB_N, dim)
nn.init.normal_(emb.weight, mean=0.0, std=1e-3)

# 用参数 phi，经 softplus 后得到 c>0，曲率 K = -c
phi = nn.Parameter(torch.tensor(0.0))


def curvature_c():
    # softplus 确保 c>0
    return torch.nn.functional.softplus(phi) + 1e-5


def poincare_distance(x, y, c):
    """
    Poincaré 球模型下的双曲距离，曲率为 -c。
    x, y: (..., dim)
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    diff2 = ((x - y) * (x - y)).sum(dim=-1, keepdim=True)

    cx2 = torch.clamp(c * x2, max=1 - 1e-5)
    cy2 = torch.clamp(c * y2, max=1 - 1e-5)

    num = 2 * c * diff2
    denom = (1 - cx2) * (1 - cy2)
    z = 1 + num / denom
    z = torch.clamp(z, min=1 + 1e-7)  # acosh 定义域

    return torch.acosh(z) / torch.sqrt(c)


# ===========================
# 4. 构造训练数据 & 优化器
# ===========================

pairs_idx = pairs
true_dists = true_dists

# 可以乘一个 scale 调一下量纲，此处先设为 1.0
scale = 1.0

params = list(emb.parameters()) + [phi]
opt = Adam(params, lr=1e-2)

num_epochs = 200
batch_size = 4096

emb.to(DEVICE)
phi.to(DEVICE)

print("\n 开始拟合 Poincaré 嵌入 + 曲率参数 ...\n")

for epoch in range(1, num_epochs + 1):
    perm = torch.randperm(pairs_idx.shape[0])
    pairs_shuffled = pairs_idx[perm]
    dists_shuffled = true_dists[perm]

    total_loss = 0.0
    total_batches = 0

    for start in range(0, pairs_shuffled.shape[0], batch_size):
        end = min(start + batch_size, pairs_shuffled.shape[0])
        batch_pairs = pairs_shuffled[start:end]
        batch_true = dists_shuffled[start:end].to(DEVICE)

        i = batch_pairs[:, 0].to(DEVICE)
        j = batch_pairs[:, 1].to(DEVICE)

        x = emb(i)
        y = emb(j)

        c = curvature_c()
        d_hat = poincare_distance(x, y, c).squeeze(-1)

        # 图距离 * scale 与 双曲距离拟合
        loss = ((d_hat - batch_true * scale) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_batches += 1

    with torch.no_grad():
        K = -curvature_c().item()

    print(f"Epoch {epoch:03d} | loss = {total_loss/total_batches:.4f} | 估计曲率 K ≈ {K:.4f}")

print("\n 训练结束。最终估计曲率 K ≈", K)
