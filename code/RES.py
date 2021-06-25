import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

BATCH_SIZE=128
EPOCHS=50
DEVICE = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"
print(DEVICE)

transform=transforms.Compose([
    transforms.Resize(224),  # 缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224),  # 从中间切出 224*224的图片
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])

train_data=ImageFolder('/opt/mnt/yl/DIP/dataset/GTSRB1/train/',transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

testpath = '/opt/mnt/yl/DIP/dataset/GTSRB1/test/'
img_path = []
img_label = []

for i in os.listdir(testpath):
    if i.split('.')[-1] == 'jpg':
        img_path.append(testpath + i)

with open("/opt/mnt/yl/DIP/dataset/GTSRB1/GT.csv") as file:
    for (i, line) in enumerate(file):
        if i != 0:
            img_label.append(int(line.strip().split(',')[-1]))


def default_loader(path):
    img_pil = Image.open(path)
    # img_pil = img_pil.resize((224, 224))
    img_tensor = transform(img_pil)
    return img_tensor


class test_set(Dataset):
    def __init__(self, loader=default_loader):
        self.images = img_path
        self.target = img_label
        self.loader = loader

    def __getitem__(self, index):
        file_path = self.images[index]
        img = self.loader(file_path)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)


test_data = test_set()
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

resnet50_feature_extractor = models.resnet50(pretrained = True)
resnet50_feature_extractor.fc = nn.Linear(2048, 43)
nn.init.eye_(resnet50_feature_extractor.fc.weight)
nn.init.zeros_(resnet50_feature_extractor.fc.bias)
model = resnet50_feature_extractor.to(DEVICE)
learning_rate = 0.001
momentum = 0.9
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-4)

criteria = nn.CrossEntropyLoss()


def test(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            #             print(type(data))

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criteria(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    #     loss.append(test_loss)
    #     accuracy.append(correct / len(testloader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

def train(model, device, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
#         print(output.shape)
#         print(target.shape)
#         loss = F.nll_loss(output, target)
#         _,predict=outputs.max(1)
#         loss = nn.CrossEntropyLoss(output, target)
        loss=criteria(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%40 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)