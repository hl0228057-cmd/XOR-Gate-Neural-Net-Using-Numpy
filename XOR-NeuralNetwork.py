import numpy as np

np.random.seed(1)

# Dataset
X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

target_y = np.array([
    [0],
    [1],
    [1],
    [0]
])

n = len(X)

# Layer sizes
input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
b2 = np.zeros((1, output_size))


def relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def MSE(errors):
    return (1 / n) * np.sum(errors ** 2)

class NeuralNetwork:
    def __init__(self, w1, b1, w2, b2, epochs, lr):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.epochs = epochs
        self.lr = lr
    
    # forward pass
    def forward(self, X):
        # hidden layer
        z1 = np.dot(X, self.w1) + self.b1
        a1 = relu(z1)

        # output layer
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid(z2)

        return (z1, a1, z2, a2)

    # backpropagation
    def backward(self, errors, a2, a1, z1):
        dA2 = 2 * (errors) / n

        dA2_dZ2 = a2 * (1 - a2)
        dZ2 = dA2 * dA2_dZ2

        dw2 = a1.T @  dZ2
        db2 = np.sum(dZ2, axis = 0, keepdims = True)

        dA1 = dZ2 @ self.w2.T
        
        dA1_dZ1 = np.where(z1 > 0, 1, 0.01)
        dZ1 = dA1 * dA1_dZ1

        dw1 = X.T @  dZ1
        db1 = np.sum(dZ1, axis = 0, keepdims = True)

        return (dw1, db1, dw2, db2)

    # training loop
    def train(self):
        for epoch in range(self.epochs):
            # forward pass
            z1, a1, z2, a2 = self.forward(X)

            errors = a2 - target_y

            # mean squared error loss
            loss = MSE(errors)
            if epoch % (self.epochs / 10) == 0:
                print(loss)

            gradientLoss = self.backward(errors, a2, a1, z1)

            self.w1 -= self.lr * gradientLoss[0]
            self.b1 -= self.lr * gradientLoss[1]
            self.w2 -= self.lr * gradientLoss[2]
            self.b2 -= self.lr * gradientLoss[3]
    
nn = NeuralNetwork(W1, b1, W2, b2, epochs=20000, lr=0.5)
nn.train()
print(np.round(nn.forward(X)[3], 3))