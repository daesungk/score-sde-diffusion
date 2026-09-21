import matplotlib.pyplot as plt
import numpy as np

from . import config, network, sde


class Sampler:
    def __init__(
        self,
        model=None,
        model_file=None,
        schedule_type=None,
        n_samples=None,
        n_steps=None,
        seed=0,
    ):
        self.n_samples = config.N_SAMPLES if n_samples is None else n_samples
        self.n_steps = config.N_STEPS if n_steps is None else n_steps
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.diffusion = None
        if model is not None:
            self.load_model(model, schedule_type)
        elif model_file is not None:
            self.load_model_from_file(model_file, schedule_type)

    def load_model(self, model, schedule_type):
        self.model = model
        self.schedule_type = schedule_type
        self.diffusion = sde.VPSDE(k=schedule_type)

    def load_model_from_file(self, filename, schedule_type):
        model = network.Network(config.X_DIM, config.HIDDEN_DIM)
        model.load(filename)
        self.load_model(model, schedule_type)

    def sample(self, snapshot_times=None):
        if self.model is None:
            raise RuntimeError("No model has been loaded.")

        x_dim = config.X_DIM

        x = self.rng.standard_normal(size=(self.n_samples, x_dim)).astype(np.float32)

        dt = self.diffusion.T / self.n_steps

        if snapshot_times is None:
            return_snapshots = False
            snapshot_times = []
        else:
            return_snapshots = True

        snapshot_steps = {round((self.diffusion.T - t) / dt): t for t in snapshot_times}

        snapshots = []

        if 0 in snapshot_steps:
            snapshots.append((self.diffusion.T, x.copy()))

        for i in range(self.n_steps):
            t = self.diffusion.T - i * dt
            t_batch = np.full(self.n_samples, t, dtype=np.float32)
            beta_t = self.diffusion.beta(t)

            # Learned score
            score = self.model.forward(x, t_batch)

            # Reverse drift
            drift = -0.5 * beta_t * x - beta_t * score

            noise = self.rng.standard_normal(size=x.shape).astype(np.float32)

            x += drift * (-dt)
            x += np.sqrt(beta_t * dt) * noise

            step = i + 1
            if step in snapshot_steps:
                snapshot_t = snapshot_steps[step]
                snapshots.append((snapshot_t, x.copy()))

        if return_snapshots:
            return snapshots

        return x

    def plot_samples(self, samples):
        plt.scatter(samples[:, 0], samples[:, 1], s=5)

        plt.axis("equal")
        plt.show()

    def plot_snapshots(self, snapshots):
        n = len(snapshots)

        _, axes = plt.subplots(1, n, figsize=(3 * n, 3))

        if n == 1:
            axes = [axes]

        for ax, (t, samples) in zip(axes, snapshots):
            ax.scatter(samples[:, 0], samples[:, 1], s=5)

            ax.set_title(f"t = {t:.2f}")
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sampler = Sampler(model_file=config.MODEL_FILE, schedule_type=config.SCHEDULE_K)
    samples = Sampler.sample()
    sampler.plot_samples(samples)
