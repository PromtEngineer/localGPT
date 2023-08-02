import torch

print(torch.version.cuda)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU.")
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} - CUDA device name:", torch.cuda.get_device_name(i))
else:
    print("PyTorch is using CPU.")

# Free up GPU memory
torch.cuda.empty_cache()

# Check if cache is freed
allocated_memory = torch.cuda.memory_allocated()
max_allocated_memory = torch.cuda.max_memory_allocated()

print("Allocated GPU memory after empty_cache():", allocated_memory)
print("Max allocated GPU memory after empty_cache():", max_allocated_memory)