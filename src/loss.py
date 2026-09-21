import numpy as np


class DSM_Loss:
    def forward(self, prediction, noise, sigma):
        self.prediction = prediction  # shape = (batch_size, 2)
        self.noise = noise
        self.sigma = sigma

        diff = sigma[:, None] * prediction + noise
        self.diff = diff

        return np.mean(np.sum(diff**2, axis=1))

    def backward(self):
        batch_size = self.prediction.shape[0]

        return 2 * self.sigma[:, None] * self.diff / batch_size
