import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"{torch.cuda.get_device_name(0)}")

