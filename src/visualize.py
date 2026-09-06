from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.sde import VPSDE
from src.toy_data import two_moons


def plot_forward_diffusion(
    save_path: str = "notebooks/figures/vpsde_two_moons.png",
    ks: tuple[int, ...] = (0, 1, 2),
) -> None:
    sde = VPSDE()
    x0 = two_moons(2000, seed=0)

    t_snapshots = [0.0, 0.1, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(
        len(ks), len(t_snapshots), figsize=(4 * len(t_snapshots), 4 * len(ks))
    )

    for row, k in enumerate(ks):
        # Same seed reused per row so all three schedules diffuse using the
        # "same" noise draws -- a fairer visual comparison across k.
        rng = np.random.default_rng(0)
        for col, t in enumerate(t_snapshots):
            ax = axes[row, col]
            if t == 0.0:
                xt = x0
            else:
                xt = sde.sample_marginal(x0, np.array(t, dtype=x0.dtype), rng, k)
            ax.scatter(xt[:, 0], xt[:, 1], s=3, alpha=0.4, color="#1f6f5c")
            ax.set_title(f"k={k}, t={t:.2f}")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")

    fig.suptitle(
        "Forward VP-SDE starting from two-moons data, across schedules k=0,1,2"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved {save_path}")


if __name__ == "__main__":
    plot_forward_diffusion()
