import cv2
from torchvision import  models
import torch.nn as nn

# model = models.resnet50(pretrained=True)
# # 重定义第一层卷积的输入通道数
# # model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
# print(model.conv1)

# from PIL import Image
#
f='/media/DATA3/swenb/dataset/LIVE/databaserelease2/refimgs/woman.bmp'
# img = Image.open(f)
# print(type(img))
# print(img.shape)

# b = cv2.dct(img[0])
# img[0]=cv2.dct(img[2])
# img[1]=cv2.dct(img[1])
# img[2]=b
# img = cv2.imread(f,0).resize(224).astype('float')
img = cv2.imread(f,0)
img1 = cv2.resize(img,(224,224)).astype('float32')
print(img)
print(img1)