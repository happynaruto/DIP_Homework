import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

vgg16 = models.vgg16(pretrained = True)
vgg16.fc = nn.Linear(4096, 43)
nn.init.eye_(vgg16.fc.weight)
nn.init.zeros_(vgg16.fc.bias)

print(vgg16)