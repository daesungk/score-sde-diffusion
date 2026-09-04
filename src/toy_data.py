"""2D toy datasets for validating the score-based diffusion pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons


def two_moons(
    n_samples: int = 2000, noise: float = 0.06, seed: int | None = 0
) -> np.ndarray:
    """Generate the classic two-interleaving-moons 2D dataset.

    Thin wrapper around sklearn.datasets.make_moons: draws the two crescents,
    adds Gaussian noise, then centers and rescales to roughly unit variance
    so the VP-SDE's noise schedule (tuned for O(1)-scale data) behaves
    sensibly. Labels (which moon each point came from) are discarded since
    the diffusion model is unsupervised.

    Returns an (n_samples, 2) float32 array.
    """
    data, _labels = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    data = data - data.mean(axis=0, keepdims=True)
    data = data / data.std()

    return data.astype(np.float32)


if __name__ == "__main__":
    d = two_moons(10)
    print(d.shape, d.dtype)
    print(d)
