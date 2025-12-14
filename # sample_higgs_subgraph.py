# sample_higgs_subgraph.py
import networkx as nx
import random

edge_path = "/home/guoquanjiang/WXY/manify/data/twitter/higgs-retweet_network.edgelist"
print("Loading Higgs edge list ...")
G = nx.read_edgelist(edge_path, nodetype=int, data=False)

print(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 这里随便来一个简单版：随机采 2000 个节点
num_sample = 2000
nodes = list(G.nodes())
sample_nodes = random.sample(nodes, num_sample)

H = G.subgraph(sample_nodes).copy()
print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

# 把子图写回一个新的 edgelist
sub_path = "/home/guoquanjiang/WXY/manify/data/twitter/higgs_sub2k.edgelist"
with open(sub_path, "w") as f:
    for u, v in H.edges():
        f.write(f"{u} {v}\n")

print(f"Saved subgraph to {sub_path}")
