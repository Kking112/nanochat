# from nanochat.engine import device
import torch

# Get GPU Name

device_name = torch.cuda.get_device_name(0)
print(device_name)