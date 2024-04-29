import torch
import torch.nn as nn
from dataset import My_Dataset
from dataset import My_Dataset_test
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
def test_GoogleNet(model):
    # 批量数目
    batch_size_test = 10
    # 预测正确个数
    correct = 0
    # 加载数据
    test_set = My_Dataset_test(r'D:\NTU_AI_FINAL\test', transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size_test, shuffle=False)
    # 开始
    for batch_data,batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction, aux_logits2, aux_logits1 = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    GoogleNet_test_accuracy= correct / 2494
    print('Googlenet: [test-accuracy]:%.2f %%' % (100 * GoogleNet_test_accuracy)) # 因为总共2494个测试数据

def test_Resnet50(model):
    # 批量数目
    batch_size_test = 10
    # 预测正确个数
    correct = 0
    # 加载数据
    test_set = My_Dataset_test(r'D:\NTU_AI_FINAL\test', transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size_test, shuffle=False)
    # 开始
    for batch_data,batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print('Resnet50: [test-accuracy]:%.2f %%' % (100 * correct / 2494)) # 因为总共2494个测试数据

def test_VIT(model):
    # 批量数目
    batch_size_test = 5
    # 预测正确个数
    correct = 0
    # 加载数据
    test_set = My_Dataset_test(r'D:\NTU_AI_FINAL\test', transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size_test, shuffle=False)
    # 开始
    for batch_data,batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print('VIT: [test-accuracy]:%.2f %%' % (100 * correct / 2494)) # 因为总共2494个测试数据

def test_Pre_VIT(model):
    # 批量数目
    batch_size_test = 5
    # 预测正确个数
    correct = 0
    # 加载数据
    test_set = My_Dataset_test(r'D:\NTU_AI_FINAL\test', transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size_test, shuffle=False)
    # 开始
    for batch_data,batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print('Pre_VIT: [test-accuracy]:%.2f %%' % (100 * correct / 2494)) # 因为总共2494个测试数据