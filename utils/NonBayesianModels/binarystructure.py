import torch.nn as nn
from utils.BBBlayers import FlattenLayer
import torch.nn.functional as F

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class ThreeFC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(ThreeFC, self).__init__()
        self.flat = FlattenLayer(784)
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, outputs)
        

    def forward(self, x):
#         print(x.size())
        out =  x.view(-1, 784)
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out