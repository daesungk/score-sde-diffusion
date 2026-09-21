import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.normal(0, np.sqrt(2 / input_dim), (output_dim, input_dim))
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        self.dW = grad.T @ self.x
        self.db = np.sum(grad, axis=0)
        return grad @ self.W


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class SiLU:
    def forward(self, x):
        self.x = x
        self.sigmoid = 1.0 / (1.0 + np.exp(-x))
        return x * self.sigmoid

    def backward(self, grad):
        derivative = self.sigmoid + self.x * self.sigmoid * (1.0 - self.sigmoid)
        return grad * derivative
