from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn import Parameter
import os
import sys
import time
import argparse

from torch.autograd import Variable
  
from utils.NonBayesianModels import conv_init
from utils.NonBayesianModels.resnet import ResNet
from utils.NonBayesianModels.AlexNet import AlexNet
from utils.NonBayesianModels.LeNet import LeNet
from utils.NonBayesianModels.SqueezeNet import SqueezeNet
from utils.NonBayesianModels.wide_resnet import Wide_ResNet
from utils.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC
from utils.NonBayesianModels.M2threefc import ThreeFC
from utils.autoaugment import CIFAR10Policy

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning_rate')
parser.add_argument('--net_type', default='alexnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100/fashionmnist/stl10]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
best_acc = 0
# resize=32
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

transform_train = transforms.Compose([
#     transforms.Resize((resize, resize)),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
#     transforms.Resize((resize, resize)),
#     transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    #CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    inputs=3

elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100
    inputs = 3
    
elif(args.dataset == 'mnist'):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    inputs = 1
    
elif(args.dataset == 'fashionmnist'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    inputs = 1
elif (args.dataset == 'stl10'):
    print("| Preparing STL10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_test)
    num_classes = 10
    inputs = 3

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes,inputs)
        file_name = 'lenet'
    elif (args.net_type == 'alexnet'):
        net = AlexNet(num_classes,inputs)
        file_name = 'alexnet-'
    elif (args.net_type == '3conv3fc'):
        net = AlexNet(num_classes, inputs)
        file_name = 'ThreeConvThreeFC-'
    elif (args.net_type == '3fc'):
        net = ThreeFC(num_classes, inputs)
        file_name = 'ThreeFC-'
    elif (args.net_type == 'squeezenet'):
        net = SqueezeNet(num_classes,inputs)
        file_name = 'squeezenet-'
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes,inputs)
        file_name = 'resnet-' + str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes,inputs)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / AlexNet /SqueezeNet/ ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/nonBayes'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
#         net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#         cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(correct))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    net, file_name = getNetwork(args)
   
    
    checkpoint = torch.load('./checkpoint/nonBayes'+args.dataset+os.sep+file_name+'.t7')
    pretrain_net = checkpoint['net']
#     best_cor = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
    pretrain_dict = pretrain_net.state_dict()
    net_dict = net.state_dict()

    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
    # 2. overwrite entries in the existing state dict
    net_dict.update(pretrain_dict) 
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t",net.state_dict()[param_tensor].size())
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)
M=2
# lev=np.sort(np.random.rand(M)-0.5)
lev = np.array([-0.15,0.15])
print(lev)
part=np.zeros((M-1,))
for m in range(M-1):
    part[m]=(lev[m]+lev[m+1])/2

net.levels1=Parameter(torch.Tensor(lev))        
net.partitions1=Parameter(torch.Tensor(part))  
net.levels2=Parameter(torch.Tensor(lev))        
net.partitions2=Parameter(torch.Tensor(part))  
net.levels3=Parameter(torch.Tensor(lev))        
net.partitions3=Parameter(torch.Tensor(part)) 
net.levels4=Parameter(torch.Tensor(lev))        
net.partitions4=Parameter(torch.Tensor(part))  
net.levels5=Parameter(torch.Tensor(lev))        
net.partitions5=Parameter(torch.Tensor(part)) 

if use_cuda:
    net.cuda()
#     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
logfile = os.path.join('diagnostics_NonBayes{}_{}.txt'.format(args.net_type, args.dataset))
print(net)
# Training
quan_cor= 0
best_cor = 0
def train(epoch):
    global quan_cor
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    if epoch>100:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net.forward(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        '''lenet
        #layer1
        flag = net.index1
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.conv1.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels1.grad=grad_value.cuda()
        #layer2
        flag = net.index2
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.conv2.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels2.grad=grad_value.cuda()
        #layer 3
        flag = net.index3
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.fc1.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels3.grad=grad_value.cuda()
        #layer 4
        flag = net.index4
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.fc2.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels4.grad=grad_value.cuda()
        # layer5
        flag = net.index5
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            
            grad_weight_ste = (net.fc3.weight.grad*inx).data.clamp_(-1,1)
        #     print(grad_weight_ste[0,:])
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels5.grad=grad_value.cuda()
        '''
        #layer1
        flag = net.index1
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.fc1.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels1.grad=grad_value.cuda()
        #layer2
        flag = net.index2
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.fc2.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels2.grad=grad_value.cuda()
        #layer 3
        flag = net.index3
        grad_value=torch.zeros((M,)).cuda()
        for m in range(M):
            inx=torch.zeros(flag.size()).cuda()
            inx = torch.where((flag !=(m+1)),inx, torch.Tensor([1]).cuda())
            grad_weight_ste = (net.fc3.weight.grad*inx).data.clamp_(-1,1)
            grad_weight_ste = torch.where((abs(grad_weight_ste)!=1), grad_weight_ste, torch.Tensor([0]).cuda()) 
            grad_value[m]= torch.sum(grad_weight_ste)
        net.levels3.grad=grad_value.cuda()
#         for p in list(net.parameters()):
#             if hasattr(p,'org'):
#                 p.data.copy_(p.org)
#         net.fc1.weight.requires_grad = False
#         net.fc2.weight.requires_grad = False
#         net.fc3.weight.requires_grad = False
        optimizer.step() # Optimizer update
        
        sort, _ = torch.sort(net.levels1.data)
        net.levels1.data = sort.cuda()
        lev=torch.sum(sort)*0.5
        net.partitions1.data[0] = lev
        
        sort, _ = torch.sort(net.levels2.data)
        net.levels2.data = sort.cuda()
        lev=torch.sum(sort)*0.5
        net.partitions2.data[0] = lev
        
        sort, _ = torch.sort(net.levels3.data)
        net.levels3.data = sort.cuda()
        lev=torch.sum(sort)*0.5
        net.partitions3.data[0] = lev
        
        
#         sort, _ = torch.sort(net.levels4.data)
#         net.levels4.data = sort.cuda()
#         sort, _ = torch.sort(net.levels5.data)
#         net.levels5.data = sort.cuda()
        
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data, 100.*correct/total))
        sys.stdout.flush()
    acc =correct
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': 100*correct / total}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))
    if correct>=quan_cor :
        quan_cor = correct
        print('Top1:',quan_cor,'%')
        np.savez('nonbayes_param'+str(M)+'.npz',par0=net.partitions1.data.cpu(),lev0 =net.levels1.data.cpu()
                                           ,par1=net.partitions2.data.cpu(),lev1 =net.levels2.data.cpu()
                                           ,par2=net.partitions3.data.cpu(),lev2 =net.levels3.data.cpu()
                                           ,par3=net.partitions4.data.cpu(),lev3 =net.levels4.data.cpu()
                                           ,par4=net.partitions5.data.cpu(),lev4 =net.levels5.data.cpu())
        state = {
                'net':net if use_cuda else net,
                'correct':correct,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/nonBayes_quan'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+str(M)+'.t7')
    print('\n')
    print('layer1\t', net.levels1.data)
    print('layer2\t', net.levels2.data)
    print('layer3\t', net.levels3.data)
    print('layer4\t', net.levels4.data)
    print('layer5\t', net.levels5.data)
def test(epoch):
    global best_cor
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net(inputs_value)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data, correct))
    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data, 'Accuracy': correct}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if correct >= best_cor:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(correct))
        state = {
                'net':net if use_cuda else net,
                'correct':correct,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/nonBayes_quan'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_cor = correct

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))
torch.backends.cudnn.enabled=False
elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_cor))
