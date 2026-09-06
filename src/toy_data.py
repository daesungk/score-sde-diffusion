from __future__ import annotations

import numpy as np


def two_moons(n: int = 2000, noise: float = 0.06, seed: int | None = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n_up = n // 2
    n_down = n - n_up

    t_up = np.linspace(0, np.pi, n_up)
    x_up = np.cos(t_up)
    y_up = np.sin(t_up)

    t_down = np.linspace(0, np.pi, n_down)
    x_down = 1 - np.cos(t_down)
    y_down = 1 - np.sin(t_down) - 0.5

    x = np.concatenate([x_down, x_up])
    y = np.concatenate([y_down, y_up])

    data = np.stack([x, y], axis=1)
    data += rng.normal(scale=noise, size=data.shape)

    data -= data.mean(axis=0, keepdims=True)
    data /= data.std()

    return data.astype(np.float32)


if __name__ == "__main__":
    d = two_moons(10)
    print(d.shape, d.dtype)
    print(d)
