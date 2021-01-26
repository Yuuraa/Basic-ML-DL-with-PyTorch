import torch

# Hyperparmeters
num_epochs = 500
learning_rate = 0.01


# Data and
x_data = torch.tensor([1.0], [2.0], [3.0])
y_data = torch.tensor([2.0], [4.0], [6.0])

class Model(torch.nn.Module):
    # Initialize the model
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in and one out

    # forward function
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

# Training step
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad() # 이 과정 항상 반드시 해줘야 함!
    loss.backward()
    optimizer.step()

# TODO: optimizer.zero_grad, step 설명 loss.backward 설명
