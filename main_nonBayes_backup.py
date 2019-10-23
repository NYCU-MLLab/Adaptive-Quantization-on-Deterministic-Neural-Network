from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from torch.autograd import Variable
  
from utils.NonBayesianModels import conv_init
from utils.NonBayesianModels.resnet import ResNet
from utils.NonBayesianModels.AlexNet import AlexNet
from utils.NonBayesianModels.backupLeNetbackup import LLeNet
from utils.NonBayesianModels.SqueezeNet import SqueezeNet
from utils.NonBayesianModels.wide_resnet import Wide_ResNet
from utils.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC
from utils.NonBayesianModels.binarystructure import ThreeFC
from utils.autoaugment import CIFAR10Policy

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
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
        net = LLeNet(num_classes,inputs)
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
#     checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    checkpoint = torch.load('./checkpoint/lenetcifar10/lenet10.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

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
    print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/nonBayes'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t",net.state_dict()[param_tensor].size())
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
#     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
logfile = os.path.join('diagnostics_NonBayes{}_{}.txt'.format(args.net_type, args.dataset))
print(net)
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        if use_cuda:
            inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs_value, targets = Variable(inputs_value), Variable(targets)
        outputs = net.forward(inputs_value)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Cor@1: %.3f%% \tTotal%.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data, correct,total))
        sys.stdout.flush()
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': 100*correct / total}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))

def test(epoch):
    global best_acc
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
    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data, 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/nonBayes'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))
torch.backends.cudnn.enabled=False
elapsed_time = 0
for epoch in range(start_epoch, 150):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
