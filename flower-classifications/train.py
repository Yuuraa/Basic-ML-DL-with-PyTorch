# Image classification practice
# Source from https://medium.com/towards-artificial-intelligence/image-classification-using-deep-learning-pytorch-a-case-study-with-flower-image-data-80a18554df63

# View data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_image(path):
    img = Image.open(path)
    image_arr = np.array(rose_img)
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))

# Preprocessing data
# Resize & Normalization
import torchvision.datasets as dsets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

transformations = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = dsets.ImageFolder("./data/flowers", transform = transformations)
dataset_loader = DataLoader(dataset = total_dataset, batch_size = 100)
items = iter(dataset_loader)
image, label = items.next()

def show_transformed_image(image):
    np_image = image.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(np_image, (1, 2, 0)))

# Building the model
# Convolution & Max-pooling layer -> extract features
from torch.utils.data import random_split

train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size
# TODO: random_split에서 데이터셋의 분포는 고려하지 않아도 되는지
train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

train_data_loader = DataLoader(dataset = train_dataset, batch_size = 100)
test_data_loader = DataLoader(dataset = test_dataset, batch_size = 100)

import torch.nn as nn


class FlowerClassifier(nn.Module):
    
    def __init(self, num_classes=5):
        super(FlowerClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.lf = nn.Linear(in_features=32*32*24, out_features=num_classes)
    

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.maxpool1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = output.view(-1, 32*32*24)
        output = self.lf(output)

        return output

# Train the model
from torch.optim import Adam
import torch

# Hyperparameters:
NUM_EPOCHES = 200


if __name__ == '__main__':
    model = FlowerClassifier()
    optimizer = Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHES):
        model.train()
        for i, (images, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        # TODO: Epoch 별 출력문
        print("Epoch: %d" %epoch)
        # print(f"Epoch: {epoch}")
        # print(f'Epoch: {epoch}  loss: {loss}')


    model.eval()
    test_acc_count = 0
    for k, (test_images, test_labels) in enumearte(test_data_loader):
        test_outputs = model(test_images)
        _, prediction = torch.max(test_outputs, 1)
        test_acc_count += torch.sum(prediction == test_labels.data).item()

    test_accuracy = test_acc_count / len(test_dataset)