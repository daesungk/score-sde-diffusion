from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VPSDE:
    # parameters
    beta_min: float = 0.1
    beta_max: float = 20.0
    a: float = 0.008
    T: float = 1.0

    # schedule
    def beta(self, t: np.ndarray | float, k: int) -> np.ndarray | float:
        if k > 0:  # polynomial growth of degree k
            return self.beta_min + (self.beta_max - self.beta_min) * np.power(t, k)
        elif k == 0:  # tanget growth
            return (
                np.pi / (1 + self.a) * np.tan((np.pi / 2) * (t + self.a) / (1 + self.a))
            )

    def beta_int(self, t: np.ndarray | float, k: int) -> np.ndarray | float:
        if k > 0:  # polynomial growth of degree k
            return self.beta_min * t + (1 / (k + 1)) * (
                self.beta_max - self.beta_min
            ) * np.power(t, k + 1)
        elif k == 0:  # tanget growth
            return (-2) * np.log(np.cos((np.pi / 2) * (t + self.a) / (1 + self.a)))

    # marginal distribution
    def marginal_dist(
        self, x0: np.ndarray, t: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # marginal distribution is Gaussian
        # returns mean and variance
        t = np.asarray(t, dtype=x0.dtype)
        B = self.beta_int(t, k)
        alpha_bar = np.exp(-B)
        mean_coef = np.sqrt(alpha_bar)
        var = 1.0 - alpha_bar
        if x0.ndim > mean_coef.ndim:
            mean_coef = mean_coef.reshape(
                mean_coef.shape + (1,) * (x0.ndim - mean_coef.ndim)
            )
            var = var.reshape(var.shape + (1,) * (x0.ndim - var.ndim))
        return mean_coef * x0, var * np.ones_like(x0)

    def sample_marginal(
        self, x0: np.ndarray, t: np.ndarray, rng: np.random.Generator, k: int
    ) -> np.ndarray:
        # Gaussian sampling
        mean, var = self.marginal_dist(x0, t, k)
        eps = rng.standard_normal(size=x0.shape).astype(x0.dtype)
        return mean + np.sqrt(var) * eps

    def true_score(
        self, x0: np.ndarray, x_t: np.ndarray, t: np.ndarray, k: int
    ) -> np.ndarray:
        mean, var = self.marginal_dist(x0, t, k)
        return -(x_t - mean) / (var)

    # SDE trajectory simulation
    def drift(self, x: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
        beta_t = self.beta(t, k)
        if x.ndim > beta_t.ndim:
            beta_t = beta_t.reshape(beta_t.shape + (1,) * (x.ndim - beta_t.ndim))
        return -0.5 * beta_t * x

    def diffusion(self, t: np.ndarray, k: int) -> np.ndarray:
        return np.sqrt(self.beta(t, k))

    def sample_trajectory(
        self,
        x0: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        k: int,
        t0: float = 0.0,
        t1: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Sample SDE trajectory starting from x0 at time between t0 and t1 using Euler-Maruyama.
        # Returns trajectory and ts where trajectory has shape (n_steps + 1, x0.shape) and ts has shape (n_steps + 1,).

        t1 = self.T if t1 is None else t1
        dt = (t1 - t0) / n_steps
        ts = t0 + dt * np.arange(n_steps + 1)

        traj = np.empty((n_steps + 1, *x0.shape), dtype=x0.dtype)
        traj[0] = x0
        x = x0.copy()
        for i in range(n_steps):
            t = ts[i]
            noise = rng.standard_normal(size=x.shape).astype(x.dtype)
            x = (
                x
                + self.drift(x, t, k) * dt
                + self.diffusion(t, k) * np.sqrt(dt) * noise
            )
            traj[i + 1] = x
        return traj, ts
