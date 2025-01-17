import torch
print(torch.cuda.is_available())  # Powinno zwrócić True
print(torch.cuda.get_device_name(0))  # Powinno wypisać "NVIDIA GeForce GTX 1060"

