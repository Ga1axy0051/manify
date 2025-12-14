import math
import random
from collections import deque

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===============================
# 0. 参数配置
# ===============================
PATH = "./manify/data/wordnet/wordnet_direct_graph.pt"

# landmark 数量（越大越准，越慢）
NUM_LANDMARKS = 1024

# 四元组抽样个数（建议几万）
NUM_QUADS = 80000

# 随机种子，保证可复现
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===============================
# 1. 加载 WordNet 图 & 建邻接表（使用原始边 edge_index）
# ===============================
print(f"📘 加载 WordNet 图: {PATH}")
data = torch.load(PATH)

print("可用字段:", data.keys())

# edge_index: shape [2, E]
edge_index = data["edge_index"]
row = edge_index[0].tolist()
col = edge_index[1].tolist()

# 节点数量（WordNet 通常有 82115）
N = int(max(max(row), max(col)) + 1)
print(f"✔ Loaded WordNet original graph: N={N}, E={len(row)}")

# 构建无向邻接表（Gromov δ 必须使用无向距离）
neighbors = [[] for _ in range(N)]
for u, v in zip(row, col):
    neighbors[u].append(v)
    neighbors[v].append(u)


# ===============================
# 2. 抽 landmark + 计算 landmark 间最短路（BFS）
# ===============================
print(f"\n 随机抽取 {NUM_LANDMARKS} 个 landmark 节点...")
landmarks = sorted(random.sample(range(N), NUM_LANDMARKS))
id_map = {node: i for i, node in enumerate(landmarks)}

# 距离矩阵 D_landmarks[i, j] = landmarks[i] 到 landmarks[j] 的最短路长度
D_landmarks = np.full((NUM_LANDMARKS, NUM_LANDMARKS), np.inf, dtype=np.float32)
np.fill_diagonal(D_landmarks, 0.0)


def bfs_from(source_node):
    """从 source_node 做 BFS，返回到所有节点的距离字典 dist[node] = d."""
    dist = {source_node: 0}
    q = deque([source_node])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in neighbors[u]:
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


print("\n🚶 对每个 landmark 做 BFS，收集 landmark 间最短路距离...")
for idx, node in enumerate(tqdm(landmarks, desc="BFS from landmarks", ncols=90)):
    dist_map = bfs_from(node)
    # 只记录到其他 landmark 的距离
    for other_node, d in dist_map.items():
        if other_node in id_map:
            j = id_map[other_node]
            D_landmarks[idx, j] = float(d)

# 检查是否有不可达情况（理论上 WordNet closure 应该是连通的）
num_inf = np.isinf(D_landmarks).sum()
if num_inf > 0:
    print(f"⚠ 警告：在 landmark 间距离矩阵中有 {num_inf} 个 inf（不可达对），后面会跳过这些四元组。")
else:
    print("✔ 所有 landmark 之间均可达，无 inf 距离。")
print("示例距离子矩阵：\n", D_landmarks[:10, :10])
print("距离取值统计：最小 =", D_landmarks.min(), 
      "最大 =", D_landmarks.max(), 
      "unique 数量 ≈", len(np.unique(D_landmarks)))




# ===============================
# 3. 定义 Gromov δ 计算函数（四点条件）
# ===============================
def delta_four_point(a, b, c, d, D):
    """
    对四个 landmark 下标 a,b,c,d 计算 Gromov δ：
    δ = 1/2 [ (d(a,c)+d(b,d)) - max( d(a,b)+d(c,d), d(a,d)+d(b,c) ) ]
    D 为 landmark 间距离矩阵。
    """
    dab = D[a, b]
    dac = D[a, c]
    dad = D[a, d]
    dbc = D[b, c]
    dbd = D[b, d]
    dcd = D[c, d]

    # 若有 inf，返回 None 表示不可用
    if np.isinf([dab, dac, dad, dbc, dbd, dcd]).any():
        return None

    s1 = dac + dbd          # AC + BD
    s2 = dab + dcd          # AB + CD
    s3 = dad + dbc          # AD + BC

    delta = 0.5 * (s1 - max(s2, s3))
    # δ 最小是 0（数值上如果有 -1e-6 这样的小负误差也截断成 0）
    if delta < 0:
        delta = 0.0
    return float(delta)

def brute_force_delta_on_first_k(D, k=20):
    import itertools
    max_delta = 0.0
    cnt_pos = 0
    for a, b, c, d in itertools.combinations(range(k), 4):
        delta = delta_four_point(a, b, c, d, D)
        if delta is None:
            continue
        if delta > 0:
            cnt_pos += 1
            if delta > max_delta:
                max_delta = delta
    print(f"[BruteForce] 在前 {k} 个 landmark 里：")
    print(f"  有 δ>0 的四元组数量: {cnt_pos}")
    print(f"  其中最大 δ = {max_delta}")
    return max_delta

# 在抽样前调用：
brute_force_delta_on_first_k(D_landmarks, k=20)



# ===============================
# 4. 抽样四元组，近似估计 δ
# ===============================
print(f"\n 抽样 {NUM_QUADS} 个四元组 (a,b,c,d)，估计 Gromov δ-hyperbolicity ...")

delta_values = []
max_delta_running = []
mean_delta_running = []
samples_count = []

L = NUM_LANDMARKS

# 为了保证不出现完全重复的四元组，这里简单用随机抽 + 去重
seen = set()

pbar = tqdm(total=NUM_QUADS, desc="Sampling quadruples", ncols=90)
while len(delta_values) < NUM_QUADS:
    # 抽 4 个不同的 landmark 下标
    quad = tuple(sorted(random.sample(range(L), 4)))
    if quad in seen:
        continue
    seen.add(quad)
    a, b, c, d = quad

    delta = delta_four_point(a, b, c, d, D_landmarks)
    if delta is None:
        # 有不可达对，跳过
        continue

    delta_values.append(delta)
    k = len(delta_values)
    max_delta_running.append(max(delta_values))
    mean_delta_running.append(sum(delta_values) / k)
    samples_count.append(k)
    pbar.update(1)

pbar.close()

print("\n 抽样完成。")
print(f"  样本数: {len(delta_values)}")
print(f"  估计的最大 δ (max over samples): {max_delta_running[-1]:.4f}")
print(f"  估计的平均 δ (mean over samples): {mean_delta_running[-1]:.4f}")


# ===============================
# 5. 可视化 δ 随抽样规模的收敛情况
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(samples_count, max_delta_running, label="max δ (running)")
plt.plot(samples_count, mean_delta_running, label="mean δ (running)")
plt.xlabel("number of sampled quadruples")
plt.ylabel("δ (Gromov four-point hyperbolicity)")
plt.title("Approximate Gromov δ-hyperbolicity on WordNet (landmark-based)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wordnet_delta_convergence.png", dpi=200)
print("\n📉 已保存收敛图: wordnet_delta_convergence.png")

print("\n✅ 完成！你可以查看命令行最后的 δ 估计值，以及当前目录下的收敛图 PNG。")
