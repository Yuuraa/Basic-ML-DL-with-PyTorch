import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
num_epochs = 2000
learning_rate = 0.01

# Data
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])


class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticModel()
criterion = nn.BCELoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    y_pred = model(x_data)
    l = criterion(y_pred, y_data)
    print(f"Epoch: {epoch}, loss: {l.item()}")
    
    # Zero gradients
    optimizer.zero_grad()
    # Perform backward pass
    l.backward()
    # Update the weights
    optimizer.step()


print("After training: ")
test_data = torch.tensor([1.0])
print(f"Predict 1 hour: {model(test_data).item()}")
test_data = torch.tensor([7.0])
print(f"Predict 7 hour: {model(test_data).item()}")
