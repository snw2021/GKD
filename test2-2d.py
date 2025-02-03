import torch, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

class CustomGradientFunction(Function):
    @staticmethod
    def forward(ctx, input, scale_factor, truth_index, confidence):
        ctx.scale_factor = scale_factor
        ctx.confidence = confidence
        ctx.truth_index = truth_index
        return input  # 直接返回输入，因为前向传播无需改动

    @staticmethod  # 重点！
    def backward(ctx, grad_output):
        print('src_logit_grad:', grad_output)
        mask = torch.ones_like(grad_output)
        mask[:, ctx.truth_index] = 0 
        scaled_grad_output = grad_output * (mask * ctx.scale_factor * ctx.confidence + (1 - mask))
        print('scaled_logit_grad:', scaled_grad_output)
        print('truth_index, confidence, scale_factor:', ctx.truth_index, ctx.confidence, ctx.scale_factor)
        return grad_output, None, None, None

# 定义一个三层全连接神经网络
class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc1.register_backward_hook(self.scale_gradients)
        self.fc2.register_backward_hook(self.scale_gradients)
        self.fc3.register_backward_hook(self.scale_gradients)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层不加激活函数，因为交叉熵损失内部会对最后一层的输出应用Softmax
        return x
    
    def scale_gradients(self, module, grad_input, grad_output):
        print(grad_output)
        #modify = tuple(g * 0.5 if g is not None else g for g in grad_input)
        # 假设我们想将梯度缩放至原来的十分之一
        #return modify
    
# 实例化网络、损失函数和优化器
input_size = 10  # 假定输入特征维度为100
hidden_size = 50
output_size = 5  # 假定分类任务有10个类别
scale_factor = 1.0 # 梯度缩放因子
model = SimpleFCN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
kl_loss = nn.KLDivLoss(reduction="batchmean")
# 假设的输入数据和目标标签
inputs = torch.randn(1, input_size)  # 一批3个样本，每个样本有10个特征
targets = torch.tensor([[0.7,0.1,0.1,0.05,0.05]]) # 对应的目标类别，每个样本一个类别
truth_index = 0
# 前向传播
outputs = model(inputs)

# 修改梯度
custom_grad = CustomGradientFunction.apply
modified_outputs = custom_grad(outputs, scale_factor, truth_index, targets[0][truth_index])
# 计算损失
prob = F.softmax(modified_outputs, dim=1)
print('pred prob:', prob)
print('true prob:', targets)
print('calculat man:', prob-targets)
#loss = -torch.sum(targets * torch.log(prob), dim=1)
loss = kl_loss(torch.log(prob), targets)
print('loss:', loss)
# 反向传播和优化
loss.backward()
optimizer.step()

# 清零梯度为下一次迭代做准备
optimizer.zero_grad()