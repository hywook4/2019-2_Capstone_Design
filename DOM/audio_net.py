import torch, os
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


class AudioConvNet(nn.Module):
    def __init__(self):
        super(AudioConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.norm2 = nn.BatchNorm2d(64)


        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.norm4 = nn.BatchNorm2d(128)


        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        self.norm6 = nn.BatchNorm2d(256)


        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        

        self.maxpool2  = nn.MaxPool2d((48, 36), stride=(48, 36))
        
        self.fc1    = nn.Linear(512, 128)
        self.fc2    = nn.Linear(128, 128)
        self.normf = nn.BatchNorm1d(128)


    def forward(self, i):
        o = F.relu(self.conv1(i))
        o = F.relu(self.norm2(self.conv2(o)))
        o = self.avgpool(o)

        o = F.relu(self.conv3(o))
        o = F.relu(self.norm4(self.conv4(o)))
        o = self.avgpool(o)

        o = F.relu(self.conv5(o))
        o = F.relu(self.norm6(self.conv6(o)))
        o = self.maxpool(o)

        o = F.relu(self.conv7(o))
        o = F.relu(self.conv8(o))

        
        o = self.maxpool2(o).squeeze(2).squeeze(2)
        o = F.relu(self.fc1(o))
        o = self.fc2(o)
        o = self.normf(o)

        return o

    
    def loss(self, output):
        return (output.mean())**2



if __name__ == "__main__":

    model = AudioConvNet().cpu()
    audio = Variable(torch.rand(2, 1, 480, 640)).cpu()


    optim = SGD(model.parameters(), lr=1e-4)
    for i in range(100):
        optim.zero_grad()
        c = model(audio)
        print("output shape:")
        print(c.shape)
        print("input shape:")
        print(audio.shape)
        print("loss:")
        loss = model.loss(c)
        loss.backward()
        print(loss)



