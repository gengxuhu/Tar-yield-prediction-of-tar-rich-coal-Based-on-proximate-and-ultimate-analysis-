import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.cuda.device_count())  # 显示可用GPU数量