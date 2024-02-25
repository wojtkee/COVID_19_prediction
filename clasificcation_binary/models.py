import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import os
# from os import listdir
from torch.utils.data import  Dataset, DataLoader
from PIL import Image
import time 
import matplotlib.pyplot as plt
from collections import Counter
import gc
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import csv




# Model

class CNN_V1(nn.Module):
    def __init__(self):
        super(CNN_V1, self).__init__()
        
        self.conv1=nn.Conv2d(1,32,
                  kernel_size=2)
        
        self.conv2=nn.Conv2d(32,64,
                  kernel_size=2)
        
        self.conv3=nn.Conv2d(64,128,
                  kernel_size=2)

        self.pool=nn.MaxPool2d(2)
        
        self.flat=nn.Flatten()
        self.lin1=nn.Linear(359552,128) #128*61*61
        self.drop=nn.Dropout(0.5)
        self.lin2=nn.Linear(128,2)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=self.flat(x)
        # print(x.shape)
        x=F.relu(self.lin1(x))
        x=self.drop(x)
        x=self.lin2(x)
        return x

    


class CNN_V2(nn.Module):
    def __init__(self):
        super(CNN_V2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2)
        
        
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(346112, 128)  
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = self.flat(x)
        # print(x.shape)
        # x= torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = self.lin2(x)
        return x

class UNET_v3(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(UNET_v3,self).__init__()

        # down layers
        # self.input_conv=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(inplace=True))
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.down_conv1=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(64,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True))
        
        self.down_conv2=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(128,256,kernel_size=3,padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        
        self.down_conv3=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(256,512,kernel_size=3,padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(inplace=True))
        

        self.down_conv4=nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(512,1024,kernel_size=3,padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True))


        # up layers

        self.up_conv41=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv31=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up_conv21=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up_conv11=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)


        self.up_conv42=nn.Sequential(nn.Conv2d(1024,512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        
        self.up_conv32=nn.Sequential(nn.Conv2d(512,256, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
    
        self.up_conv22=nn.Sequential(nn.Conv2d(256,128, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        
        self.up_conv12=nn.Sequential(nn.Conv2d(128,64, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        

        self.out_conv=nn.Conv2d(64,1,kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Warstwy klasyfikujące dla całego obrazu
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(1024, 128)  # Przykładowy rozmiar warstwy liniowej
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, out_channels)  # out_channels - liczba klas

    def forward(self, x):
        xinput = self.input_conv(x)
        x1 = self.down_conv1(xinput)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)

        # Przekształcenie wyjścia za pomocą warstwy GAP
        x = self.pool(x4)
        x = x.view(x.size(0), -1)

        # Warstwy klasyfikujące dla całego obrazu
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = self.lin2(x)

        return x