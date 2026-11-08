import torch, triton
print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())
