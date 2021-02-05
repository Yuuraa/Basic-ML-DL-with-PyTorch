import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)

# Hyperparameters
learning_rate = 0.01
num_epoch = 10


# Forward pass
def forward(x):
    return w * x


# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2


# Before training
for epoch in range(num_epoch):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1. Forward pass
        l = loss(y_pred, y_val) # 2. Calculate Loss
        l.backward() # Backpropagation to calculate gradients
        print(f"\t grad: {w.grad.item()} x: {x_val}, y: {y_val}")
        w.data = w.data - learning_rate * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print(f"Prediction (after training), input: 4, \
    accuracy: {1 - loss(forward(4).item(), 8)} ")

    