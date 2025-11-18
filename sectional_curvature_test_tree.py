import torch
from manify.curvature_estimation.sectional_curvature_strict_fast import sectional_curvature_gpu

import networkx as nx

def make_tree_graph(branch_factor=4, depth=6):
    edges = []
    node = 0
    next_node = 1
    for d in range(depth):
        for i in range(branch_factor**d):
            for b in range(branch_factor):
                edges.append((node, next_node))
                node += 1
                next_node += 1

    # 构造邻接矩阵
    edges = torch.tensor(edges, dtype=torch.long).T
    n = edges.max().item() + 1
    adj = torch.zeros((n, n))
    adj[edges[0], edges[1]] = adj[edges[1], edges[0]] = 1

    # ✅ 用 networkx 计算最短路径距离矩阵（严格图距离）
    G = nx.from_numpy_array(adj.numpy())
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    dists = torch.zeros((n, n))
    for i, row in lengths.items():
        for j, dist in row.items():
            dists[i, j] = dist

    return adj, dists


adj, dists = make_tree_graph()
curv = sectional_curvature_gpu(adj, dists, device="cpu", relative=False)
print("平均曲率:", curv.mean().item())
