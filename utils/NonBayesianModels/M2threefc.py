import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from utils.BBBlayers import FlattenLayer
import torch
M = 2
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)
def Quantize(w, level, partition):
    w_=w.clone()
    M,c_num=level.shape[0],partition.shape[0]
    flag = torch.zeros(w.size()).cuda()
    flag4grad = torch.zeros(w.size()).cuda()
    for i in range (M):
        tmp=torch.ones(w.size()).cuda().mul(level[i]).cuda()
        if (i==0):
            w_=torch.where(w_>partition[i],w_,tmp)
            flag = torch.where(w_>partition[i],flag,torch.Tensor([i+1]).cuda())
            flag4grad = torch.where(w_>level[i],flag4grad,torch.Tensor([i+1]).cuda())
        elif(i==M-1):
            w_=torch.where(w_<=partition[i-1],w_,tmp)
            flag = torch.where(w_<=partition[i-1],flag,torch.Tensor([i+1]).cuda())
            flag4grad = torch.where(w_<=level[i-1],flag4grad,torch.Tensor([i+1]).cuda())
        else:
            w_=torch.where((w_<partition[i-1])| (w_>partition[i]),w_,tmp)
            flag = torch.where((w_<partition[i-1])| (w_>partition[i]),flag,torch.Tensor([i+1]).cuda())
            flag4grad = torch.where((w_<level[i-1])| (w_>level[i]),flag4grad,torch.Tensor([i+1]).cuda())
    return w_,flag,flag4grad

class ThreeFC(nn.Module):
    def __init__(self, num_classes, inputs=1):
        super(ThreeFC, self).__init__()
        self.flat = FlattenLayer(784)
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.levels1 = Parameter(torch.Tensor(M,))
        self.index1 = torch.Tensor(self.fc1.weight.size())
        self.index2 = torch.Tensor(self.fc2.weight.size())
        self.index3 = torch.Tensor(self.fc3.weight.size())
        self.partitions1 = Parameter(torch.Tensor(M-1,))
        self.levels2 = Parameter(torch.Tensor(M,))
        self.partitions2 = Parameter(torch.Tensor(M-1,))
        self.levels3 = Parameter(torch.Tensor(M,))
        self.partitions3 = Parameter(torch.Tensor(M-1,))
        
    def forward(self, x):
        w1, flag1, flag4grad1 = Quantize(self.fc1.weight.data, self.levels1.data,self.partitions1.data)
        self.index1 = flag1
#         self.fc1.weight.org.copy_(self.fc1.weight.data.clamp_(-1,1))
        self.fc1.weight.org = self.fc1.weight.data.clone()
        self.fc1.weight.data = w1
        
        w, flag2, flag4grad2 = Quantize(self.fc2.weight.data, self.levels2.data,self.partitions2.data)
        self.index2 = flag2
#         self.fc2.weight.org.copy_(self.fc2.weight.data.clamp_(-1,1))
        self.fc2.weight.org = self.fc2.weight.data.clone()
        self.fc2.weight.data = w
        
        w, flag, flag4grad = Quantize(self.fc3.weight.data, self.levels3.data,self.partitions3.data)
        self.index3 = flag
#         self.fc3.weight.org.copy_(self.fc3.weight.data.clamp_(-1,1))
        self.fc3.weight.org = self.fc3.weight.data.clone()
        self.fc3.weight.data = w
        out =  x.view(-1, 784)
        out = self.relu(self.bn1(self.fc1(out)))
#         out.data=Binarize(out.data)
        out = self.relu(self.bn2(self.fc2(out)))
#         out.data=Binarize(out.data)
        out = self.fc3(out)
     
        return self.logsoftmax(out)
