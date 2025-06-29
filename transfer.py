import torch

# 载入整个.pt模型（结构+权重）
model = torch.load('./user_data/model_pt/best.pt', map_location='cpu')  

# 提取 state_dict（可能嵌套在 'model' 键中）
if 'model' in model:
    state_dict = model['model'].state_dict()
else:
    state_dict = model.state_dict()  # 有时直接是模型

# 保存 state_dict 到新文件
torch.save(state_dict, 'model_best.pth')