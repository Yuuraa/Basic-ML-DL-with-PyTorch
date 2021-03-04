import torch
import torch.nn as nn
import torchvision

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Download dataset
data_dir = '/content/dataset' # 참조: 새로 dataset 폴더를 만들고 그 안으로 .tif 파일들을 옮겼음

train_label = 'train-labels.tif'
train_input = 'train-volume.tif'

img_label = Image.open(os.path.join(data_dir, train_label))
img_input = Image.open(os.path.join(data_dir, train_input))

ny, nx = img_label.size # 512 512
nframe = img_label.n_frames # 30 n_frame은 이미지의 갯수를 의미함

# nframe_train = 24
# nframe_val = 3
# nframe_test = 3

# Train, test, validate 데이터셋 저장 폴더 만들기
train_save_dir = os.path.join(data_dir, 'train')
val_save_dir = os.path.join(data_dir, 'val')
test_save_dir = os.path.join(data_dir, 'test')

if not os.path.exists(train_save_dir):
    os.makedirs(train_save_dir)
if not os.path.exists(val_save_dir):
    os.makedirs(val_save_dir)
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)


total_img = []
total_label = []
for i in range(30):
    # Seek i는 i 번째 프레임에 있는 이미지를 읽어주는 듯 하다
    img_label.seek(i)
    img_input.seek(i)
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    total_img.append(input_)
    total_label.append(label_)


# Train-test split
train_img, test_img, train_label, test_label = train_test_split(total_img, total_label, test_size=0.1, random_state=1)
train_img, val_img, train_label, val_label = train_test_split(train_img, train_label, train_size=0.89, test_size=0.11)

for i in range(len(train_img)):
    np.save(os.path.join(train_save_dir, f'input_{i:03d}.npy'), train_img[i])
    np.save(os.path.join(train_save_dir, f'label_{i:03d}.npy'), train_label[i])

for i in range(len(val_img)):
    np.save(os.path.join(val_save_dir, f'input_{i:03d}.npy'), val_img[i])
    np.save(os.path.join(val_save_dir, f'label_{i:03d}.npy'), val_label[i])

for i in range(len(test_img)):
    np.save(os.path.join(test_save_dir, f'input_{i:03d}.npy'), test_img[i])
    np.save(os.path.join(test_save_dir, f'label_{i:03d}.npy'), test_label[i])


# Visualize Image
idx = np.random.randint(0, high=len(train_img))
print(idx)
plt.subplot(121)
plt.imshow(train_img[idx], cmap='gray')
plt.title('input')

plt.subplot(122)
plt.imshow(train_label[idx], cmap='gray')
plt.title('label')

plt.show()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        data_list = os.listdir(data_dir)

        labels = [f for f in data_list if f.startswith('label')]
        inputs = [f for f in data_list if f.startswith('input')]

        labels.sort()
        inputs.sort()

        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = np.load(os.path.join(self.data_dir, self.labels[idx]))
        input = np.load(os.path.join(self.data_dir, self.inputs[idx]))

        # Normalize
        label = label / 255.0
        input = input / 255.0 

        # PyTorch에 들어가기 위해서는 x, y, channel 3개의 차원의 입력이 필요
        if label.ndim == 2:
            label = label[:,:,np.newaxis]
        if input.ndim == 2:
            input = input[:,:,np.newaxis]


        data = {'input': input, 'label': label}
        if self.transform:
            data = self.transform(data)

        return data

# Define Transforms
import torchvision.transforms.functional as F
class RandomFlips(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        input_img = data['input']
        label_img = data['label']

        if torch.rand(1) < self.p:
            input_img = F.hflip(input_img)
            label_img = F.hflip(label_img)

        if torch.rand(1) < self.p:
            input_img = F.vflip(input_img)
            label_img = F.vflip(label_img)
        
        data['input'] = input_img
        data['label'] = label_img

        return data

class ToFloat32Tensors(nn.Module):
    def __call__(self, data):
        data['input'] = F.to_tensor(data['input']).float()
        data['label'] = F.to_tensor(data['label']).float()
        return data

class Normalizes(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def forward(self, data):
        """ data is a dictionary, composed of two tensor images: 'input' and 'output'
            Returns dictionary of two normalized tensors """
        data['input'] = F.normalize(data['input'], self.mean, self.std, self.inplace)
        data['label'] = F.normalize(data['label'], self.mean, self.std, self.inplace)
        return data

