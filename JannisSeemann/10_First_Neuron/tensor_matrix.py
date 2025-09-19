import torch

X = torch.tensor([
    [10],
    [38],
    [100],
    [150]
])

# print(X)
# print(X.shape)
# print(X.size())
# print(X.size(0))
# print(X.size(1))

print(X[:, 0])
