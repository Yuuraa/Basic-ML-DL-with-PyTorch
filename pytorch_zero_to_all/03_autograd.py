import torch
from torch.autograd import Variable

# initial weight and hyper parameters
# TODO: Variable 쓰지 않고도 requires_grad 지정하는 법 있었던 것 같음. 조사
w = Variable(torch.Tensor([1.0]), requires_grad=True)
learning_rate = 0.01
num_epoch = 100

# Training Data & Test data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
x_test = [4.0]
y_test = [8.0]

# Initialize the weight, w
def init_weight(val = 1.0):
    global w
    w = val


# Our model for the forward pass
def forward(x):
    return x * w


# Loss function
def MSE_loss(y_pred, y):
    return (y_pred - y) ** 2


# Train the model to find w
for epoch in range(num_epoch):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = MSE_loss(y_pred, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        # gradient 값이 w.grad.data에 저장됨
        w = w - learning_rate * w.grad.data
    
    print("Progress: ", epoch, "w: ", w, "loss: ", l)


# Test the model and print accuracy
total_loss = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val)
    l = MSE_loss(y_pred, y_val)
    total_loss += l
print('Loss after training: ', total_loss / len(x_data))