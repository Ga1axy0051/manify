import torch
import os

# 自动获取 test.py 所在目录
root = os.path.dirname(os.path.abspath(__file__))

# 拼出 telecom_graph.pt 的绝对路径
graph_path = os.path.join(root, "data", "telecom", "telecom_graph.pt")

print("Loading:", graph_path)

data = torch.load(graph_path, map_location="cpu")

print("Loaded keys:", data.keys())
