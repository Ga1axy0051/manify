import torch

data = torch.load("./manify/data/wordnet/wordnet_direct_graph.pt")

print("==== wordnet_direct_graph.pt ====")
for k, v in data.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, type(v))
