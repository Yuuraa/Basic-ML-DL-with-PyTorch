import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from util import *
from model import *
from dataset import *
import torchvision.transforms as transforms

# Hyperparameters & Directories
DATA_DIR = '/content/dataset' # 참조: 새로 dataset 폴더를 만들고 그 안으로 .tif 파일들을 옮겼음
CKPT_DIR = '/content/drive/MyDrive/UNet/checkpoint'
LOG_DIR = '/content/drive/MyDrive/UNet/logs'
RESULT_DIR = '/content/drive/MyDrive/UNet/results'

LEARNING_RATE = 1e-3
BATCH_SIZE = 4
NUM_EPOCH = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)


# Load datset
train_transform = transforms.Compose([
                                        ToFloat32Tensors(),
                                        Normalizes(mean=0.5, std=0.5),
                                        RandomFlips()
                                      ])
train_dataset = Dataset(data_dir=os.path.join(DATA_DIR, 'train'), transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_transform = transforms.Compose([
                                        ToFloat32Tensors(),
                                        Normalizes(mean=0.5, std=0.5),
                                        RandomFlips(),
                                      ])
val_dataset = Dataset(data_dir=os.path.join(DATA_DIR, 'val'), transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# Train the model
model = UNet().to(device)

loss_func = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCH):
    model.train()
    loss_arr = []

    for batch, data in enumerate(train_loader, 1):
        # Forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        # print(input.shape)

        y_pred = model(input)

        # Backward pass
        optimizer.zero_grad()
        
        loss = loss_func(y_pred, label)
        loss.backward()

        optimizer.step()

        loss_arr += [loss.item()]

        print(f"Train: Epoch {epoch:04d}/{num_epoch:04d} | batch {batch:04d} | loss: {np.mean(loss_arr):.4f}")

    with torch.no_grad():
        model.eval()
        loss_arr = []
        for batch, data in enumerate(val_loader, 1):
            input = data['input'].to(device)
            label = data['label'].to(device)

            y_pred = model(input)
            idx = np.random.randint(len(input))
            show_tensor_to_img(input[idx], label[idx], y_pred[idx])
            loss = loss_func(y_pred, label)
            loss_arr += [loss.item()]

        print(f"Eval: Epoch {epoch:04d}/{num_epoch:04d} | Validation loss: {np.mean(loss_arr):.4f}")