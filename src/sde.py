from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class VPSDE:
    k: int
    beta_min: float = config.BETA_MIN
    beta_max: float = config.BETA_MAX
    a: float = config.ADJ_COEF
    T: float = config.T

    def __post_init__(self):
        if self.k < 0:
            raise ValueError("k must be a nonnegative integer")

    def beta(self, t: np.ndarray | float) -> np.ndarray | float:
        if self.k > 0:
            return self.beta_min + (self.beta_max - self.beta_min) * np.power(t, self.k)

        return np.pi / (1 + self.a) * np.tan((np.pi / 2) * (t + self.a) / (1 + self.a))

    def beta_int(self, t: np.ndarray | float) -> np.ndarray | float:
        if self.k > 0:
            return self.beta_min * t + (self.beta_max - self.beta_min) / (
                self.k + 1
            ) * np.power(t, self.k + 1)

        # tangent schedule
        return -2.0 * np.log(np.cos((np.pi / 2) * (t + self.a) / (1 + self.a)))

    def marginal_dist(
        self, x0: np.ndarray, t: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:

        t = np.asarray(t, dtype=x0.dtype)
        if t.ndim == 0:
            t = np.full(x0.shape[0], t, dtype=x0.dtype)
        B = self.beta_int(t)
        alpha_bar = np.exp(-B)
        var = 1.0 - alpha_bar

        mean_coef = np.sqrt(alpha_bar)
        mean = mean_coef[:, None] * x0

        return mean, var

    def sample_marginal(
        self, x0: np.ndarray, t: np.ndarray | float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:

        mean, var = self.marginal_dist(x0, t)
        eps = rng.standard_normal(size=x0.shape).astype(x0.dtype)
        xt = mean + np.sqrt(var)[:, None] * eps

        return xt, eps

    def true_score(
        self, x0: np.ndarray, x_t: np.ndarray, t: np.ndarray | float
    ) -> np.ndarray:

        mean, var = self.marginal_dist(x0, t)
        return -(x_t - mean) / var

    def drift(
        self,
        x: np.ndarray,
        t: np.ndarray | float,
    ) -> np.ndarray:

        beta_t = self.beta(t)
        beta_t = np.asarray(beta_t)

        if x.ndim > beta_t.ndim:
            beta_t = beta_t.reshape(beta_t.shape + (1,) * (x.ndim - beta_t.ndim))

        return -0.5 * beta_t * x

    def diffusion(self, t: np.ndarray | float) -> np.ndarray | float:

        return np.sqrt(self.beta(t))

    def sample_trajectory(
        self,
        x0: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        t0: float = 0.0,
        t1: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        t1 = self.T if t1 is None else t1

        dt = (t1 - t0) / n_steps
        ts = t0 + dt * np.arange(n_steps + 1)

        trajectory = np.empty(
            (n_steps + 1, *x0.shape),
            dtype=x0.dtype,
        )

        trajectory[0] = x0

        x = x0.copy()

        for i in range(n_steps):
            t = ts[i]

            noise = rng.standard_normal(size=x.shape).astype(x0.dtype)

            x = x + self.drift(x, t) * dt + self.diffusion(t) * np.sqrt(dt) * noise

            trajectory[i + 1] = x

        return trajectory, ts
