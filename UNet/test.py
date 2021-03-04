import numpy as np
import os

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


model = UNet().to(device)
# load(model, CKPT_DIR)

test_transform  = transforms.Compose([
                                        ToFloat32Tensors(),
                                        Normalizes(mean=0.5, std=0.5),
                                      ])
test_dataset = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

with torch.no_grad():
    model.eval()
    loss_arr = []

    for i, data in enumerate(test_loader, 1):
        input = data['input'].to(device)
        label = data['label'].to(device)

        y_pred = model(input)

        loss = loss_func(y_pred, label)

        print(f'TEST: {batch:d} LOSS: {loss.item():.4f}')
    print(f'TEST: AVERAGE LOSS: {np.mean(loss_arr)}')