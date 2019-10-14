import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np
import random
import math
import time
from PIL import Image

from audio_net import *



class DOMNet(nn.Module):
    def __init__(self):
        super(DOMNet, self).__init__()


        # video net
        ## TODO
		
        # audio net 
        self.audio_net = AudioConvNet()


		# Combining layers
        self.mse     = F.mse_loss
        self.fc3     = nn.Linear(1, 2)
        self.softmax = F.softmax


    def forward(self, video, audio):
		# Video
        ## TODO
        vid = None
        

		# Audio
        ad = audio_net(audio)

		# Join them 
        mse = self.mse(vid, aud, reduce=False).mean(1).unsqueeze(1)
        out = self.fc3(mse)
        out = self.softmax(out, 1)

        return out, vid, aud



