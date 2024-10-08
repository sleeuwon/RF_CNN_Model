import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from modelsunshining import SParamCNNsunniest
from predictor import predicting

class SParamCNN(nn.Module):
    def __init__(self):
        super(SParamCNN, self).__init__()
        self.encoderconv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.encoderconv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.encoderconv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoderpool = nn.MaxPool2d(2, stride=2,return_indices=True)
        self.encoderfc1 = nn.Linear(64 * 5 * 5, 1024)
        self.encoderfc2 = nn.Linear(1024, 12 * 301 * 2)  # Adjusted output size
        #self.downfc1 = nn.Linear(3*40*40, 3*40*40)
        #self.downfc2 = nn.Linear(3*40*40, 3*40*40)  # Adjusted output size

        self.decoderfc2 = nn.Linear(12 * 301 * 2,1024)  # Adjusted output size
        self.decoderfc1 = nn.Linear(1024, 64 * 5 * 5)
        """
        self.decoderpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decoderpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decoderpool3 = nn.Upsample(scale_factor=2, mode='bilinear')
        """
        self.decoderpool1 = nn.ConvTranspose2d(64,64, kernel_size=2,padding=3,stride=(3,3))
        self.decoderpool2 = nn.ConvTranspose2d(32,32, kernel_size=2,padding=1,stride=(3,3))
        self.decoderpool3 = nn.ConvTranspose2d(16,16, kernel_size=2,padding=1,stride=(2,2))
        self.decoderconv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.decoderconv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoderconv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)



        
    def forward(self, x):
        """
        x=F.relu(self.encoderconv1(x))
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        #print(len(x[0][0][0]))
        #print(len(x[0][0][0][0]))

        x,i1 = self.encoderpool(x)
        x,i2 = self.encoderpool(F.relu(self.encoderconv2(x)))
        x,i3 = self.encoderpool(F.relu(self.encoderconv3(x)))
        """
        """
        print(i1)
        print(i2)
        print(i3)
        print(len(i1))  #batch size
        print(len(i1[0])) #second level depth
        print(len(i1[0][0])) #pixmap size?
        print(len(i1[0][0][0])) #pixmap size?
        print(len(i2))  #batch size
        print(len(i2[0])) #second level depth
        print(len(i2[0][0])) #pixmap size?
        print(len(i2[0][0][0])) #pixmap size?
        print(len(i3))  #batch size
        print(len(i3[0])) #second level depth
        print(len(i3[0][0])) #pixmap size?
        print(len(i3[0][0][0])) #pixmap size?
        print(len(i3[0][0][0][0]))
        """
        """
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.encoderfc1(x))
        #print(len(x))
        #print(len(x[0]))
        x = self.encoderfc2(x)
        #print(len(x[0]))
        #print(len(x))
        x = x.view(-1, 12, 301, 2)  # Reshape to [batch_size, num_of_sparameters, frequency_points, 2]
        """
        #x = x.view(-1, 12, 301, 2)  # Reshape to [batch_size, num_of_sparameters, frequency_points, 2]

        y = x.view(-1,7224)
        #print(len(y))
        #print(len(y[0]))
        y = self.decoderfc2(y)
        #print(len(y))
        y = F.relu(self.decoderfc1(y))
        y = y.view(-1,64,5,5) #10,64,5,5
        #verified layers until here
        y = self.decoderpool1(y)
        y = F.relu(y)
        y = self.decoderconv1(y)
        y = self.decoderpool2(y)
        y = F.relu(y)
        y = self.decoderconv2(y)
        y = self.decoderpool3(y)
        y = F.relu(y)
        y = self.decoderconv3(y)
        #print(len(y))
        #print(len(y[0]))
        #print(len(y[0][0]))
        #print(len(y[0][0][0]))
        #y = F.relu(self.downfc1(y))
        #print(len(x))
        #print(len(x[0]))
        #y = self.downfc2(y)
        #y=F.sigmoid(y)
        #y=torch.round(y)
        #y=torch.tensor((y.clone().detach().requires_grad_(True)>.5).float()).requires_grad_(True)
        #y=np.around(y.detach().to("cpu"),decimals=0).to("cuda").requires_grad_(True)
        y = y.view(-1,3,40,40)
        
        x=predicting(y)
        
        return x,y
