import torch
from torch import nn

# turning C into F

X = torch.tensor([
    [10.0],
    [38.0],
    [100.0],
    [150.0]
])

model = nn.Linear(1, 1)

model.bias = nn.Parameter(torch.tensor([32.0]))
model.weight = nn.Parameter(torch.tensor([[1.8]]))

print(f"Model bias: {model.bias}")
print(f"Model weight: {model.weight}")

y_pred = model(X)

print(f"Predicted values: {y_pred}")
