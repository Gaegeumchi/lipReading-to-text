import torch
print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"✅ 현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
