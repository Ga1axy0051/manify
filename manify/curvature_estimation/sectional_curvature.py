r"""Sectional curvature estimation for graphs (GPU optimized + tqdm progress bar)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from tqdm import tqdm  # ✅ 新增

if TYPE_CHECKING:
    from jaxtyping import Float


def sectional_curvature(
    adjacency_matrix: Float[torch.Tensor, "n_points n_points"],
    distance_matrix: Float[torch.Tensor, "n_points n_points"],
    samples: int | None = None,
    relative: bool = True,
    show_progress: bool = True,
    device: str | torch.device = "cuda",
) -> Float[torch.Tensor, "n_points"] | Float[torch.Tensor, "samples"]:
    """
    GPU 加速 + tqdm 实时进度条版。
    """
    if not isinstance(adjacency_matrix, torch.Tensor) or not isinstance(distance_matrix, torch.Tensor):
        raise TypeError("Both adjacency_matrix and distance_matrix must be torch.Tensors")

    if adjacency_matrix.shape != distance_matrix.shape:
        raise ValueError("Adjacency matrix and distance matrix must have the same shape")

    A = adjacency_matrix.float().to(device)
    D = distance_matrix.float().to(device)

    if samples is not None:
        return _sample_curvatures(D, samples, relative, device)
    else:
        return _compute_node_curvatures(A, D, relative, device)


def _sample_curvatures(
    D: Float[torch.Tensor, "n_points n_points"],
    n_samples: int,
    relative: bool,
    device: str | torch.device,
) -> Float[torch.Tensor, "n_samples"]:
    """Sample random triangle configurations and compute curvature estimates."""
    n = D.shape[0]
    indices = torch.randint(0, n, (n_samples, 4), device=device)
    a, b, c, m = indices.T

    valid_mask = a != m
    curvatures = torch.zeros(n_samples, dtype=torch.float32, device=device)

    if valid_mask.any():
        valid_a, valid_b, valid_c, valid_m = a[valid_mask], b[valid_mask], c[valid_mask], m[valid_mask]
        curvature_values = (
            D[valid_a, valid_m] ** 2
            + (D[valid_b, valid_c] ** 2) / 4.0
            - (D[valid_a, valid_b] ** 2 + D[valid_a, valid_c] ** 2) / 2.0
        ) / (2 * D[valid_a, valid_m])
        curvatures[valid_mask] = curvature_values

    if relative:
        curvatures = curvatures / torch.max(D)

    return curvatures


def _compute_node_curvatures(
    A: Float[torch.Tensor, "n_points n_points"],
    D: Float[torch.Tensor, "n_points n_points"],
    relative: bool,
    device: str | torch.device,
) -> Float[torch.Tensor, "n_points"]:
    """Compute curvature for each node by averaging over neighbor triangles, with tqdm progress bar."""
    n = A.shape[0]
    node_curvatures = torch.zeros(n, dtype=torch.float32, device=device)

    # ✅ 使用 tqdm 监控循环
    for m in tqdm(range(n), desc="Computing node curvatures", ncols=90):
        neighbors = torch.where(A[m] == 1)[0]
        if len(neighbors) < 2:
            continue

        triangle_curvatures = []

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                b, c = neighbors[i], neighbors[j]

                # 预先构建索引向量 (批处理)
                a_indices = torch.arange(n, device=device)
                valid_a = a_indices[a_indices != m]

                D_am = D[valid_a, m]
                D_bc = D[b, c]
                D_ab = D[valid_a, b]
                D_ac = D[valid_a, c]

                # 向量化曲率计算
                curvature_vals = (D_am**2 + (D_bc**2) / 4.0 - (D_ab**2 + D_ac**2) / 2.0) / (2 * D_am)
                triangle_curvatures.append(curvature_vals.mean())

        if triangle_curvatures:
            node_curvatures[m] = torch.stack(triangle_curvatures).mean()
    
    if relative:
       node_curvatures = node_curvatures / torch.max(D)

    return node_curvatures

