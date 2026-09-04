"""2D toy datasets for validating the score-based diffusion pipeline.

Kept dependency-free (pure numpy) so Week 1 doesn't need scikit-learn.
"""

from __future__ import annotations

import numpy as np


def two_moons(
    n_samples: int = 2000, noise: float = 0.06, seed: int | None = 0
) -> np.ndarray:
    """Generate the classic two-interleaving-moons 2D dataset.

    Same construction as sklearn.datasets.make_moons, reimplemented here
    so the project has no dependency on scikit-learn.

    Returns an (n_samples, 2) float32 array, roughly centered at the origin.
    """
    rng = np.random.default_rng(seed)

    n_out = n_samples // 2
    n_in = n_samples - n_out

    outer_t = np.linspace(0, np.pi, n_out)
    outer_x = np.cos(outer_t)
    outer_y = np.sin(outer_t)

    inner_t = np.linspace(0, np.pi, n_in)
    inner_x = 1 - np.cos(inner_t)
    inner_y = 1 - np.sin(inner_t) - 0.5

    x = np.concatenate([outer_x, inner_x])
    y = np.concatenate([outer_y, inner_y])

    data = np.stack([x, y], axis=1)
    data += rng.normal(scale=noise, size=data.shape)

    # Center and scale roughly to unit-ish variance so the VP-SDE's noise
    # schedule (tuned for data with O(1) scale) behaves sensibly.
    data -= data.mean(axis=0, keepdims=True)
    data /= data.std()

    return data.astype(np.float32)


if __name__ == "__main__":
    d = two_moons(10)
    print(d.shape, d.dtype)
    print(d)
