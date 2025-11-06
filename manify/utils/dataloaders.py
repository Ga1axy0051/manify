"""# Dataloaders Submodule.

The dataloaders module allows users to load datasets from Manify's datasets repo [on Hugging Face](https://huggingface.co/manify).

We provide a summary of the data types available, and their original sources, here.

Earlier versions of Manify included scripts to process raw data, which we have replaced with a single, centralized Hugging Face repo and the function `load_hf`. For transparency, we have preserved the data generation code in [the Dataset-Generation branch of Manify](https://github.com/pchlenski/manify/tree/Dataset-Generation).

| Dataset | Task | Distance Matrix | Features | Labels | Adjacency Matrix | Source/Citation |
|---------|------|----------------|----------|--------|-----------------|-----------------|
| cities | none | ✅ | ❌ | ❌ | ❌ | [Network Repository: Cities](https://networkrepository.com/Cities.php) |
| cs_phds | regression | ✅ | ❌ | ✅ | ✅ | [Network Repository: CS PhDs](https://networkrepository.com/CSphd.php) |
| polblogs | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polblogs](https://networkrepository.com/polblogs.php) |
| polbooks | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polbooks](https://networkrepository.com/polbooks.php) |
| cora | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Cora](https://networkrepository.com/cora.php) |
| citeseer | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Citeseer](https://networkrepository.com/citeseer.php) |
| karate_club | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Karate](https://networkrepository.com/karate.php) |
| lesmis | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Lesmis](https://networkrepository.com/lesmis.php) |
| adjnoun | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Adjnoun](https://networkrepository.com/adjnoun.php) |
| football | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Football](https://networkrepository.com/football.php) |
| dolphins | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Dolphins](https://networkrepository.com/dolphins.php) |
| blood_cells | classification | ❌ | ✅ | ✅ | ❌ | See datasets from Zheng et al (2017): Massively parallel digital transcriptional profiling of single cells.<br>- [CD8+ Cytotoxic T-cells](https://www.10xgenomics.com/datasets/cd-8-plus-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD8+/CD45RA+ Naive Cytotoxic T Cells](https://www.10xgenomics.com/datasets/cd-8-plus-cd-45-r-aplus-naive-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD56+ Natural Killer Cells](https://www.10xgenomics.com/datasets/cd-56-plus-natural-killer-cells-1-standard-1-1-0)<br>- [CD4+ Helper T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-helper-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RO+ Memory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-oplus-memory-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RA+/CD25- Naive T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-aplus-cd-25-naive-t-cells-1-standard-1-1-0)<br>- [CD4+/CD25+ Regulatory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-25-plus-regulatory-t-cells-1-standard-1-1-0)<br>- [CD34+ Cells](https://www.10xgenomics.com/datasets/cd-34-plus-cells-1-standard-1-1-0)<br>- [CD19+ B Cells](https://www.10xgenomics.com/datasets/cd-19-plus-b-cells-1-standard-1-1-0)<br>- [CD14+ Monocytes](https://www.10xgenomics.com/datasets/cd-14-plus-monocytes-1-standard-1-1-0) |
| lymphoma | classification | ❌ | ✅ | ✅ | ❌ | See datasets from 10x Genomics:<br>- [Hodgkin's Lymphoma](https://www.10xgenomics.com/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0)<br>- [Healthy Donor PBMCs](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0) |
| cifar_100 | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) |
| mnist | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| temperature | regression | ❌ | ✅ | ✅ | ❌ | [Citation] |
| landmasses | classification | ❌ | ✅ | ✅ | ❌ | Generated using [basemap.is_land](https://matplotlib.org/basemap/stable/api/basemap_api.html#mpl_toolkits.basemap.Basemap.is_land) |
| neuron_33 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| neuron_46 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| traffic | regression | ❌ | ✅ | ✅ | ❌ | [Kaggle: Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) |
| qiita | none | ✅ | ✅ | ❌ | ❌ | [NeuroSEED Git Repo](https://github.com/gcorso/NeuroSEED) |
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from jaxtyping import Float, Real


def load_hf(
    name: str, namespace: str = "manify"
) -> tuple[
    Float[torch.Tensor, "n_points ..."] | None,  # features
    Float[torch.Tensor, "n_points n_points"] | None,  # pairwise dists
    Float[torch.Tensor, "n_points n_points"] | None,  # adjacency
    Real[torch.Tensor, "n_points"] | None,  # labels
]:
    """
    Load a dataset from HuggingFace Hub at {namespace}/{name}, or from PyG if name='pubmed'.
    """
    # ✅ 新增分支：PubMed 数据集
    if name.lower() == "pubmed":
        print("📘 Loading PubMed dataset using PyTorch Geometric ...")
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_dense_adj
        import time

        start_time = time.time()
        dataset = Planetoid(root="data/PubMed", name="PubMed")
        data = dataset[0]

        features = data.x
        labels = data.y
        adj = to_dense_adj(data.edge_index)[0]

        print(f"✅ Loaded raw PubMed tensors: features {features.shape}, adj {adj.shape}, labels {labels.shape}")

        # 计算 pairwise 欧式距离矩阵
        with torch.no_grad():
            try:
                print(" Computing pairwise distance matrix...")
                dists = torch.cdist(features, features)
            except RuntimeError:
                subset = 1000
                print(f" 内存不足，抽样前 {subset} 个节点计算距离矩阵")
                features = features[:subset]
                labels = labels[:subset]
                adj = adj[:subset, :subset]
                dists = torch.cdist(features, features)

        elapsed = time.time() - start_time
        print(f" PubMed dataset loaded in {elapsed:.2f} seconds")
        print(f"节点数: {features.shape[0]}, 特征维度: {features.shape[1]}, 类别数: {len(torch.unique(labels))}\n")

        return features, dists, adj, labels
    
    #COMPUTERS dataset
    elif name.lower() == "computers":
        print("📘 Loading Amazon Computers dataset using PyTorch Geometric ...")
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root="data/Computers", name="Computers")
        data = dataset[0]

    # adjacency matrix
        adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
        edges = data.edge_index
        adj[edges[0], edges[1]] = 1
        adj[edges[1], edges[0]] = 1  # 无向图

    # 计算 pairwise 距离矩阵（简单版：用特征欧氏距离）
        print("Computing pairwise distance matrix (features-based)...")
        features = data.x
        dists = torch.cdist(features, features, p=2)

        features = features.float()
        labels = data.y.long()
        print(" Amazon Computers dataset loaded successfully!")
        return features, dists, adj, labels


    # ✅ 原始逻辑（Hugging Face 数据集）
    ds = load_dataset(f"{namespace}/{name}")
    data = ds.get("train", ds)  # use "train" split if available, else the only split
    row = data[0]

    def to_tensor(key: str, dtype: torch.dtype) -> torch.Tensor | None:
        vals = row.get(key, [])
        if not vals:
            return None
        return torch.tensor(vals, dtype=dtype)

    dists = to_tensor("distances", torch.float32)
    feats = to_tensor("features", torch.float32)
    adj = to_tensor("adjacency", torch.float32)

    cls_ls = row.get("classification_labels", [])
    reg_ls = row.get("regression_labels", [])
    if cls_ls:
        labels = torch.tensor(cls_ls, dtype=torch.int64)
    elif reg_ls:
        labels = torch.tensor(reg_ls, dtype=torch.float32)
    else:
        labels = None

    return feats, dists, adj, labels