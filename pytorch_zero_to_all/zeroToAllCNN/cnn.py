# Import libraries
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Use GPU if available
isgpu = torch.cuda.is_available()
device = torch.device('cuda' if isgpu else 'cpu')

torch.manual_seed(777) # Random value를 고정해주는 부분
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        # 이거 빼먹지 않도록 유의할 것1
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Layer2에서 나온 것을 펼친 값
        self.fc = nn.Linear(7*7*64, 10, bias=True) # bias를 넣어줌
        torch.nn.init.xavier_uniform_(self.fc.weight) # 초기화 해줌

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_batch = len(data_loader) # 전체 batch size를 알 수 있게 됨
print("======= Training phase =========\n")

for epoch in range(training_epochs):
    avg_cost = 0
    # X 는 이미지, Y는 라벨
    for X, Y in data_loader:
        # cuda 연산을 진행하기 위함 -> torch.tensor가 아니라 torch.cuda_tensor가 되어야 함
        X = X.to(device)
        Y = Y.to(device)
        # 매우 중요! 역전파 단계를 실행하기 전에 변화도를 0으로 만듬
        optimizer.zero_grad()
        hypothesis = model(X)

        # Cross entropy로 cost를 계산한다
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        

