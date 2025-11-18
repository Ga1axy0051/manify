r"""Sectional curvature estimation for graphs (GPU safe + tqdm + logging + checkpoint + dataset-based logs)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import os
import json
from tqdm import tqdm
from datetime import datetime

if TYPE_CHECKING:
    from jaxtyping import Float


def sectional_curvature(
    adjacency_matrix: Float[torch.Tensor, "n_points n_points"],
    distance_matrix: Float[torch.Tensor, "n_points n_points"],
    samples: int | None = None,
    relative: bool = True,
    show_progress: bool = True,
    device: str | torch.device = "cuda",
    dataset_name: str = "default",        #  新增：自动按数据集命名日志
    save_every: int = 100,
    resume: bool = True,
    force_restart: bool = False,
    base_log_dir: str = "./curvature_logs",  #  主日志目录
) -> Float[torch.Tensor, "n_points"] | Float[torch.Tensor, "samples"]:
    """
    GPU 加速 + tqdm 实时进度条 + 安全过滤（避免 NaN / Inf）+ 自动日志管理。
    """

    # =========================================================
    # 1️ 构造数据集专属日志目录
    # =========================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(base_log_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "curvature_log.txt")
    ckpt_path = os.path.join(save_dir, "curvature_checkpoint.pt")
    final_path = os.path.join(save_dir, "curvature_final.pt")

    # =========================================================
    # 2️ 日志写入函数
    # =========================================================
    def log(msg: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        print(msg)

    # =========================================================
    # 3️ 初始化 / 恢复检查点
    # =========================================================
    if force_restart and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        log(" 已清理旧的 checkpoint 文件。")

    start_index = 0
    node_curvatures = None
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        node_curvatures = ckpt.get("node_curvatures")
        start_index = ckpt.get("last_index", 0)
        log(f" 检测到未完成任务，恢复自节点 {start_index} ...")

    A = adjacency_matrix.float().to(device)
    D = distance_matrix.float().to(device)
    n = A.shape[0]
    if node_curvatures is None:
        node_curvatures = torch.zeros(n, dtype=torch.float32, device=device)

    log(f" 开始计算曲率，总节点数: {n}, 设备: {device}")

    # =========================================================
    # 4️ 曲率计算核心
    # =========================================================
    invalid_total = 0
    total_samples = 0

    iterator = tqdm(range(start_index, n), desc="Computing node curvatures", ncols=90) if show_progress else range(start_index, n)

    for m in iterator:
        neighbors = torch.where(A[m] == 1)[0]
        if len(neighbors) < 2:
            continue

        triangle_curvatures = []

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                b, c = neighbors[i], neighbors[j]
                a_indices = torch.arange(n, device=device)
                valid_a = a_indices[a_indices != m]

                D_am = D[valid_a, m]
                D_bc = D[b, c]
                D_ab = D[valid_a, b]
                D_ac = D[valid_a, c]

                #  数值安全过滤
                safe_mask = (D_am > 1e-9) & torch.isfinite(D_am)
                D_am_safe = D_am[safe_mask]
                if len(D_am_safe) == 0:
                    continue

                D_ab_safe = D_ab[safe_mask]
                D_ac_safe = D_ac[safe_mask]

                curvature_vals = (
                    D_am_safe**2 + (D_bc**2) / 4.0 - (D_ab_safe**2 + D_ac_safe**2) / 2.0
                ) / (2 * D_am_safe)

                # 去除 NaN / Inf
                finite_mask = torch.isfinite(curvature_vals)
                invalid_total += (~finite_mask).sum().item()
                total_samples += len(curvature_vals)

                curvature_vals = curvature_vals[finite_mask]
                if len(curvature_vals) == 0:
                    continue

                triangle_curvatures.append(curvature_vals.mean())

        if triangle_curvatures:
            node_curvatures[m] = torch.stack(triangle_curvatures).mean()

        # 定期保存
        if (m + 1) % save_every == 0 or (m + 1) == n:
            torch.save(
                {"node_curvatures": node_curvatures, "last_index": m + 1},
                ckpt_path
            )
            log(f" 已保存进度：节点 {m + 1}/{n}")

    # =========================================================
    # 5️ 后处理 + 安全归一化
    # =========================================================
    if relative:
        max_D = torch.max(D)
        if max_D > 0 and torch.isfinite(max_D):
            node_curvatures = node_curvatures / max_D
        else:
            log(" 跳过归一化：最大距离为 0 或 NaN。")

    # =========================================================
    # 6 结果保存 + 汇总统计
    # =========================================================
    torch.save(node_curvatures, final_path)
    log(" 曲率计算完成，结果已保存 curvature_final.pt")

    stats = {
        "mean": float(node_curvatures.mean().item()),
        "min": float(node_curvatures.min().item()),
        "max": float(node_curvatures.max().item()),
        "invalid_ratio": float(invalid_total / max(total_samples, 1)),
        "nodes": int(n),
        "device": str(device),
    }

    with open(os.path.join(save_dir, "curvature_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log(f" 结果统计：mean={stats['mean']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}, 无效比={stats['invalid_ratio']:.2%}")
    return node_curvatures
