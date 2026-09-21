import json
from dataclasses import asdict

import numpy as np

from . import layers


class Network:
    def __init__(self, x_dim, hidden_dim, activation_cls=layers.SiLU):
        self.layer1 = layers.Linear(x_dim + 1, hidden_dim)
        self.activation1 = activation_cls()

        self.layer2 = layers.Linear(hidden_dim, hidden_dim)
        self.activation2 = activation_cls()

        self.layer3 = layers.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        network_input = np.concatenate([x, t[:, None]], axis=1)

        z1 = self.layer1.forward(network_input)
        u1 = self.activation1.forward(z1)

        z2 = self.layer2.forward(u1)
        u2 = self.activation2.forward(z2)

        v = self.layer3.forward(u2)

        return v

    def backward(self, grad_output):
        grad_u2 = self.layer3.backward(grad_output)
        grad_z2 = self.activation2.backward(grad_u2)

        grad_u1 = self.layer2.backward(grad_z2)
        grad_z1 = self.activation1.backward(grad_u1)

        grad_input = self.layer1.backward(grad_z1)

        return grad_input

    def parameters(self):
        return [
            (self.layer1.W, self.layer1.dW),
            (self.layer1.b, self.layer1.db),
            (self.layer2.W, self.layer2.dW),
            (self.layer2.b, self.layer2.db),
            (self.layer3.W, self.layer3.dW),
            (self.layer3.b, self.layer3.db),
        ]

    def save(self, filename):
        data = {
            "W1": self.layer1.W,
            "b1": self.layer1.b,
            "W2": self.layer2.W,
            "b2": self.layer2.b,
            "W3": self.layer3.W,
            "b3": self.layer3.b,
        }

        if hasattr(self, "training_config"):
            data["config"] = json.dumps(asdict(self.training_config))

        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)

        self.layer1.W[...] = data["W1"]
        self.layer1.b[...] = data["b1"]

        self.layer2.W[...] = data["W2"]
        self.layer2.b[...] = data["b2"]

        self.layer3.W[...] = data["W3"]
        self.layer3.b[...] = data["b3"]
