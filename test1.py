import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

devNumber = torch.cuda.current_device()
print(f"Current CUDA device number: {devNumber}")

devName = torch.cuda.get_device_name(devNumber)
print(f"GPU name is: {devName}")

# create a tensor on CPU

T1 = torch.randn(4, 4)
print("CPU Tensor")
print(T1)

# this will convert a CPU tensor with pinned memory to a CUDA tensor
T2 = T1.to(device)
print("GPU Tensor")
print(T2)
