import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dense = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 43)
        )
        self.initialize_weights()

    def forward(self, x):
        pool1 = self.maxpool1(x)

        pool2 = self.maxpool2(pool1)

        pool3 = self.maxpool3(pool2)

        pool4 = self.maxpool4(pool3)

        pool5 = self.maxpool5(pool4)

        flat = pool5.view(pool5.size(0), -1)

        output = self.dense(flat)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

BATCH_SIZE=128
EPOCHS=50
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(DEVICE)

transform=transforms.Compose([
    transforms.Resize(224),  # 缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224),  # 从中间切出 224*224的图片
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])

train_data=ImageFolder('/opt/mnt/swb/dataset/GTSRB-Training_fixed/GTSRB/Training/',transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

testpath = '/opt/mnt/swb/dataset/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/'
img_path = []
img_label = []

for i in os.listdir(testpath):
    if i.split('.')[-1] == 'ppm':
        img_path.append(testpath + i)

with open("/opt/mnt/swb/dataset/GTSRB_Final_Test_GT/GT-final_test.csv") as file:
    for (i, line) in enumerate(file):
        if i != 0:
            img_label.append(int(line.strip().split(';')[-1]))


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

model = VGG16().to(DEVICE)
learning_rate = 0.001
momentum = 0.9
optimizer = torch.optim.Adam(model.parameters())

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