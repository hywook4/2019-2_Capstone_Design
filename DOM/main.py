import os
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




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: ", device)

    #set varables

    num_epoch = 50
    learning_rate = 0.0001
    momentum = 0.9

    