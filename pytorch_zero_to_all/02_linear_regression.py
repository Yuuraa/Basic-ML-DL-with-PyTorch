import numpy as np

# initial weight and hyper parameters
w = 0.0
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


''' ### Compute the MSE loss for w
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = MSE_loss(x_val, y_val)
        l_sum += 1
        print("\t", x_val, y_val, y_pred_val, l)
    
    print("MSE=", l_sum / 3)
'''

# Gradient descent
def gradient(x, y):
    return 2 * x * (x * w - y)


# Train the model to find w
for epoch in range(num_epoch):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        y_pred = forward(x_val)
        w = w - learning_rate * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = MSE_loss(y_pred, y_val)
    
    print("Progress: ", epoch, "w: ", w, "loss: ", l)


# Test the model and print accuracy
total_loss = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val)
    l = MSE_loss(y_pred, y_val)
    total_loss += l
print('Loss after training: ', total_loss / len(x_data))
    