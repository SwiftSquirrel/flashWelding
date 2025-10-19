# -*- coding: utf-8 -*-
# @Author: zzt
# @Date:   2022-10-19 19:21:41
# @Last Modified by:   zzt
# @Last Modified time: 2022-11-29 21:18:03

#LSTM网络用于提取时间信息
import imp
import nntplib
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import Linear,Embedding,LSTM,CrossEntropyLoss

class LSTMnet(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,dropout):
        """
        hidden_dim: lstm神经元数量
        layer_dim: lstm层数input
        output_dim: 输出维度（分类数量）
        """
        super(LSTMnet,self).__init__()
        torch.manual_seed(77)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTM(input_dim, hidden_dim, layer_dim, batch_first = True, dropout = dropout)


    def forward(self,x):
        r_out, (h_n, h_c) = self.lstm(x,None)  #零初始化
        out = r_out[:,-1,:] #选取lstm网络最后一个时间点的输出
        return out



#ResNet18网络用于提取空间信息
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1): # 默认输出通道数一致，不使用1x1卷积层
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        # 第一个模块stride=1
        # 其余模块的第一个残差层strides=2，通道加倍，高宽减半
        if i == 0 and not first_block: 
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2)) 
        else: 
            blk.append(Residual(num_channels, num_channels))
    return blk

# RseNet hierarchical
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)

        #针对原子节点空间坐标(coordinate)的卷积操作
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                   nn.BatchNorm2d(64), nn.ReLU(), 
                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        b2 = nn.Sequential(*resnet_block(None, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.convs= nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveMaxPool2d((1,1)),
                    nn.Flatten())
        
    def forward(self,x):  
        out1 = self.convs(x)
        return out1



class MultiModalModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_layer_dim, 
                 output_dim, dropout):
        super(MultiModalModel, self).__init__()
        torch.manual_seed(77)

        # 初始化 LSTM 模块
        self.lstm_model = LSTMnet(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            layer_dim=lstm_layer_dim,
            dropout=dropout
        )

        # 初始化 ResNet18 模块
        self.resnet_model = ResNet18()

        # 融合层：将两个特征向量拼接后送入全连接层
        self.fusion_layer = nn.Sequential(
            nn.Linear(3*lstm_hidden_dim + 512, 256),  #512为残差网络输出维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_lstm1,x_lstm2,x_lstm3, x_resnet):
        # 提取 LSTM 特征
        out_lstm1 = self.lstm_model(x_lstm1)  # shape: [batch_size, lstm_hidden_dim]
        out_lstm2 = self.lstm_model(x_lstm2)  # shape: [batch_size, lstm_hidden_dim]
        out_lstm3 = self.lstm_model(x_lstm3)  # shape: [batch_size, lstm_hidden_dim]
        # 提取 ResNet 特征
        out_resnet = self.resnet_model(x_resnet.unsqueeze(1))  # shape: [batch_size, resnet_output_dim]

        # 特征拼接
        combined = torch.cat((out_lstm1,out_lstm2,out_lstm3, out_resnet), dim=1)  # shape: [batch_size, lstm_hidden_dim + resnet_output_dim]

        # 通过融合层
        output = self.fusion_layer(combined)

        return output
