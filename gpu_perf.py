import torch

# ... Your code ...

# Check initial GPU memory usage
initial_allocated_memory = torch.cuda.memory_allocated()
print("Initial allocated GPU memory:", initial_allocated_memory)

# ... Perform your computations ...

# Check GPU memory usage after computations
current_allocated_memory = torch.cuda.memory_allocated()
max_allocated_memory = torch.cuda.max_memory_allocated()

print("Current allocated GPU memory:", current_allocated_memory)
print("Max allocated GPU memory:", max_allocated_memory)
