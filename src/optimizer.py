import numpy as np

from . import config


class SGD:
    def __init__(self, learning_rate: float = config.LEARNING_RATE):
        self.learning_rate = learning_rate

    def step(self, parameters):
        for param, grad in parameters:
            param -= self.learning_rate * grad


class Adam:
    def __init__(
        self, learning_rate=config.LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def step(self, parameters):
        if self.m is None:
            self.m = [np.zeros_like(param) for param, grad in parameters]
            self.v = [np.zeros_like(param) for param, grad in parameters]

        self.t += 1

        for i, (param, grad) in enumerate(parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
