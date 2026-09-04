"""Variance-preserving (VP) forward SDE and its closed-form Gaussian marginal.

Forward SDE (Song & Ermon 2021, continuous-time limit of DDPM):

    dx = -1/2 * beta(t) * x dt + sqrt(beta(t)) dW,   t in [0, T]

with a linear noise schedule beta(t) = beta_min + t * (beta_max - beta_min).

Because beta(t) is linear in t, B(t) = int_0^t beta(s) ds has a closed form,
which makes the marginal p_t(x_t | x_0) an exact Gaussian:

    x_t | x_0 ~ N( alpha_bar(t)^0.5 * x_0,  (1 - alpha_bar(t)) * I )

where alpha_bar(t) = exp(-B(t)). This closed form is the ground truth that
the Euler-Maruyama simulation below is checked against in tests/test_sde.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VPSDE:
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0

    # ---- schedule ----------------------------------------------------

    def beta(self, t: np.ndarray | float) -> np.ndarray | float:
        """Instantaneous noise rate beta(t), linear in t."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: np.ndarray | float) -> np.ndarray | float:
        """B(t) = integral_0^t beta(s) ds, in closed form since beta is linear."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    # ---- closed-form marginal p_t(x_t | x_0) --------------------------

    def marginal_prob(
        self, x0: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of the Gaussian marginal x_t | x_0, in closed form.

        x0: (..., d) array. t: scalar or (...) array broadcastable against x0's
        leading dimensions. Returns (mean, std), each broadcastable to x0's shape.
        """
        t = np.asarray(t, dtype=x0.dtype)
        B = self.integral_beta(t)
        alpha_bar = np.exp(-B)
        mean_coef = np.sqrt(alpha_bar)
        std = np.sqrt(1.0 - alpha_bar)
        # Broadcast mean_coef/std (shape (...,)) against x0's trailing data dim.
        mean_coef = mean_coef[..., None] if mean_coef.ndim == x0.ndim - 1 else mean_coef
        std = std[..., None] if std.ndim == x0.ndim - 1 else std
        return mean_coef * x0, std * np.ones_like(x0)

    def sample_marginal(
        self, x0: np.ndarray, t: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw x_t | x_0 exactly, via the closed-form reparameterization."""
        mean, std = self.marginal_prob(x0, t)
        eps = rng.standard_normal(size=x0.shape).astype(x0.dtype)
        return mean + std * eps

    def true_score(self, x0: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Analytic score of the conditional p_t(x_t | x_0): grad_{x_t} log p(x_t|x0).

        For a Gaussian N(mean, std^2 I), grad log p(x_t) = -(x_t - mean) / std^2.
        This is exactly what denoising score matching (Week 2) regresses against.
        """
        mean, std = self.marginal_prob(x0, t)
        return -(x_t - mean) / (std**2)

    # ---- forward-process simulation (Euler-Maruyama) -------------------

    def drift(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """f(x, t) = -1/2 beta(t) x."""
        beta_t = self.beta(t)
        beta_t = beta_t[..., None] if np.ndim(beta_t) == x.ndim - 1 else beta_t
        return -0.5 * beta_t * x

    def diffusion(self, t: np.ndarray) -> np.ndarray:
        """g(t) = sqrt(beta(t))."""
        return np.sqrt(self.beta(t))

    def simulate_forward(
        self,
        x0: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
        t0: float = 0.0,
        t1: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the forward SDE from x0 at t0 to t1 via Euler-Maruyama.

        Returns (trajectory, ts) where trajectory has shape
        (n_steps + 1, *x0.shape) and ts has shape (n_steps + 1,).
        """
        t1 = self.T if t1 is None else t1
        dt = (t1 - t0) / n_steps
        ts = t0 + dt * np.arange(n_steps + 1)

        traj = np.empty((n_steps + 1, *x0.shape), dtype=x0.dtype)
        traj[0] = x0
        x = x0.copy()
        for i in range(n_steps):
            t = ts[i]
            g = self.diffusion(t)
            noise = rng.standard_normal(size=x.shape).astype(x.dtype)
            x = x + self.drift(x, t) * dt + g * np.sqrt(dt) * noise
            traj[i + 1] = x
        return traj, ts
