import torch
import torch.nn as nn
import os
from model.GoogleNet_kidney import GoogLeNet
from model.VIT_kidney import VIT
from model.Pre_VIT_kidney import Pre_VIT
from model.Resnet50_kidney import ResNet50
from data.dataset import My_Dataset,My_Dataset_test
from Test import test_GoogleNet,test_Resnet50,test_VIT,test_Pre_VIT
# import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
os.environ['CUDA_LAUNCH_BLOCKING']='0'


#-------------------------------save para------------------------------



def save_checkpoint_GoogLeNet(model, epoch):  # save model function
    model_out_path = 'Weights_Googlenet' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

def save_checkpoint_Resnet50(model, epoch):  # save model function
    model_out_path = 'Weights_Resnet50' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

def save_checkpoint_VIT(model, epoch):  # save model function
    model_out_path = 'Weights_VIT' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

def save_checkpoint_Pre_VIT(model, epoch):  # save model function
    model_out_path = 'Weights_Pre_VIT' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)



#-----------------------------------------------train----------------------------------




def train_Googlenet():
    model_path = r'Weights_Googlenet/100.pth'
    batch_size = 64 # 批量训练大小
    model = GoogLeNet()
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
        print('Googlenet is Successfully Loaded from %s' % (model_path))

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(),lr=0.001)
    # 加载数据
    train_set = My_Dataset(r'D:\NTU_AI_FINAL\train',transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # 训练20次
    # train_loss_list = []
    # train_acc_list = []
    #
    # test_acc_list = []
    writer = SummaryWriter('logs_GoogleNet')
    for epoch in range(100):

        loss_temp = 0
        correct = 0# 临时变量
        for j,(batch_data,batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data,batch_label = batch_data.to(device),batch_label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            logits, aux_logits2, aux_logits1 = model(batch_data)
            # 损失值
            loss0 = loss_func(logits,batch_label)
            loss1= loss_func(aux_logits2, batch_label)
            loss2 = loss_func(aux_logits1, batch_label)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(logits.data, 1)[1]
            # 获取准确个数
            correct += (predicted == batch_label).sum()
        # 打印一次损失值
        loss=loss_temp/len(train_loader)
        # train_loss_list .append(loss)
        train_accuracy=correct / 8710
        print('Googlenet: [%d:epoch] loss:%.3f' % (epoch+1,loss))
        print('Googlenet: [train-accuracy]:%.2f %%' % (100 * train_accuracy))
        # train_acc_list.append(train_accuracy)
        # if (i+1) % 20 == 0:
        #      save_checkpoint_GoogLeNet(model, epoch+1)

#-----------------------------------------test----------------------------------------
        # 批量数目
        batch_size_test = 10
        # 预测正确个数
        correct = 0
        # 加载数据
        test_set = My_Dataset_test(r'D:\NTU_AI_FINAL\test', transform=transforms.ToTensor())
        test_loader = DataLoader(test_set, batch_size_test, shuffle=False)
        # 开始
        for batch_data, batch_label in test_loader:
            # 放入GPU中
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            # 预测
            prediction, aux_logits2, aux_logits1 = model(batch_data)
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(prediction.data, 1)[1]
            # 获取准确个数
            correct += (predicted == batch_label).sum()
        test_accuracy = correct / 2494
        # test_acc_list.append(test_accuracy)
        print('Googlenet: [test-accuracy]:%.2f %%' % (100 * test_accuracy))  # 因为总共2494个测试数据


#---------------------------------------------draw--------------------------------------------
        # writer.add_scalar('train-accuracy',train_accuracy,global_step=epoch+1)
        # writer.add_scalar('train-loss',loss,global_step=epoch+1)
        # writer.add_scalar('test-accuracy',test_accuracy ,global_step=epoch+1)
        print('---------------------------------------------------------')
    # # 绘制loss曲线图
    # plt.subplot(1,2,1)
    # plt.plot(train_loss_list, label='train')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('loss-graph')
    #
    # # 绘制accuracy曲线图
    # plt.subplot(1,2,2)
    # plt.plot(train_acc_list, label='train')
    # plt.plot(test_acc_list, label='test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy-graph')
    # plt.tight_layout()
    # plt.show()

def train_Resnet50():
    model_path = r'/Weights_Resnet100'
    batch_size = 32 # 批量训练大小
    model = ResNet50()
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
        print('Resnet50 is Successfully Loaded from %s' % (model_path))

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(),lr=0.001)
    # 加载数据
    train_set = My_Dataset(r'D:\NTU_AI_FINAL\train',transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # 训练20次
    for i in range(100):
        loss_temp = 0  # 临时变量
        correct = 0
        for j,(batch_data,batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data,batch_label = batch_data.to(device),batch_label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            prediction = model(batch_data)
            # 损失值
            loss = loss_func(prediction,batch_label)
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(prediction.data, 1)[1]
            # 获取准确个数
            correct += (predicted == batch_label).sum()
        # 打印一次损失值
        print('Resnet50: [%d:epoch] loss:%.3f' % (i+1,loss_temp/len(train_loader)))
        print('Resnet50: [train-accuracy]:%.2f %%' % (100 * correct / 8710))
        if (i+1) % 20 == 0:
             save_checkpoint_Resnet50(model, i+1)
        test_Resnet50(model)
        print('---------------------------------------------------------')

def train_VIT():
    model_path = r'/Weights_VIT_100'
    batch_size = 10 # 批量训练大小
    model = VIT(num_classes=4)
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
        print('VIT is Successfully Loaded from %s' % (model_path))

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(),lr=0.001)
    # 加载数据
    train_set = My_Dataset(r'D:\NTU_AI_FINAL\train',transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # 训练20次
    for i in range(100):
        loss_temp = 0  # 临时变量
        correct = 0
        for j,(batch_data,batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data,batch_label = batch_data.to(device),batch_label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            prediction = model(batch_data)
            # 损失值
            loss = loss_func(prediction,batch_label)
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(prediction.data, 1)[1]
            # 获取准确个数
            correct += (predicted == batch_label).sum()
        # 打印一次损失值
        print('VIT: [%d:epoch] loss:%.3f' % (i+1,loss_temp/len(train_loader)))
        print('VIT: [train-accuracy]:%.2f %%' % (100 * correct / 8710))
        if (i+1) % 20 == 0:
             save_checkpoint_VIT(model, i+1)
        test_VIT(model)
        print('---------------------------------------------------------')

def train_Pre_VIT():
    model_path = r'/Weights_Pre_VIT_100'
    batch_size = 10 # 批量训练大小
    model = Pre_VIT()
    # 将模型放入GPU中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
        print('Pre_VIT is Successfully Loaded from %s' % (model_path))

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=model.parameters(),lr=0.001)
    # 加载数据
    train_set = My_Dataset(r'D:\NTU_AI_FINAL\train',transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # 训练20次
    for i in range(100):
        loss_temp = 0  # 临时变量
        correct = 0
        for j,(batch_data,batch_label) in enumerate(train_loader):
            # 数据放入GPU中
            batch_data,batch_label = batch_data.to(device),batch_label.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 模型训练
            prediction = model(batch_data)
            # 损失值
            loss = loss_func(prediction,batch_label)
            loss_temp += loss.item()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 将预测值中最大的索引取出，其对应了不同类别值
            predicted = torch.max(prediction.data, 1)[1]
            # 获取准确个数
            correct += (predicted == batch_label).sum()
        # 打印一次损失值
        print('Pre_VIT: [%d:epoch] loss:%.3f' % (i+1,loss_temp/len(train_loader)))
        print('Pre_VIT: [train-accuracy]:%.2f %%' % (100 * correct / 8710))
        if (i+1) % 20 == 0:
             save_checkpoint_Pre_VIT(model, i+1)
        test_Pre_VIT(model)
        print('---------------------------------------------------------')









if __name__ == "__main__":
    train_Googlenet()
    #train_Resnet50()
    # train_VIT()
    #train_Pre_VIT()
