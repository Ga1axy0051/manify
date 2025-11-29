# manify/curvature_estimation/gromov_curvature_wordnet.py

import torch
import random
from collections import deque
from tqdm import tqdm


# =========================================================
# 邻接表构建
# =========================================================
def build_neighbors_from_sparse(adj_sparse: torch.Tensor):
    A = adj_sparse.coalesce()
    row, col = A.indices()
    n = A.size(0)
    neigh = [[] for _ in range(n)]
    for u, v in zip(row.tolist(), col.tolist()):
        neigh[u].append(v)
    return neigh


# =========================================================
# BFS until targets
# =========================================================
def bfs_until(neigh, start, targets, max_dist=10):
    dist = {start: 0}
    q = deque([start])
    remaining = set(targets)

    while q:
        u = q.popleft()
        if dist[u] >= max_dist:
            continue
        for v in neigh[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                if v in remaining:
                    remaining.remove(v)
                    if not remaining:
                        return dist
                q.append(v)
    return dist


# =========================================================
# 四点 Gromov δ
# =========================================================
def gromov_delta(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd):
    s1 = d_ab + d_cd
    s2 = d_ac + d_bd
    s3 = d_ad + d_bc
    arr = sorted([s1, s2, s3])
    return 0.5 * (arr[2] - arr[1])


# =========================================================
#  主函数（带进度条）
# =========================================================
def compute_gromov_curvature_wordnet(
    adj_sparse: torch.Tensor,
    num_samples: int = 20000,
    max_dist: int = 10,
    seed: int = 0
):
    """
    采样 WordNet 图的 Gromov 四点曲率（δ-hyperbolicity）
    返回:
        mean_delta (float)
        deltas (Tensor[num_samples])
    """

    print(f" 开始 WordNet Gromov 曲率采样：num_samples={num_samples}, max_dist={max_dist}")

    random.seed(seed)
    torch.manual_seed(seed)

    n = adj_sparse.size(0)
    neigh = build_neighbors_from_sparse(adj_sparse)

    deltas = []
    attempts = 0

    # 用 tqdm 包住采样进度
    pbar = tqdm(total=num_samples, desc="Sampling 4-point tuples", ncols=80)

    while len(deltas) < num_samples and attempts < num_samples * 15:
        attempts += 1

        # 随机采样四个节点
        a, b, c, d = random.sample(range(n), 4)

        # BFS 1: from a
        dist_a = bfs_until(neigh, a, [b, c, d], max_dist)
        if b not in dist_a or c not in dist_a or d not in dist_a:
            continue

        # BFS 2: from b
        dist_b = bfs_until(neigh, b, [c, d], max_dist)
        if c not in dist_b or d not in dist_b:
            continue

        # BFS 3: from c
        dist_c = bfs_until(neigh, c, [d], max_dist)
        if d not in dist_c:
            continue

        # 计算 δ
        delta = gromov_delta(
            dist_a[b], dist_a[c], dist_a[d],
            dist_b[c], dist_b[d], dist_c[d]
        )
        deltas.append(delta)
        pbar.update(1)

    pbar.close()

    if not deltas:
        raise RuntimeError("采样不到有效四元组，请增大 max_dist。")

    deltas_tensor = torch.tensor(deltas, dtype=torch.float32)
    return deltas_tensor.mean().item(), deltas_tensor
