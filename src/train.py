import matplotlib.pyplot as plt
import numpy as np

from . import config, layers, sde
from . import loss as ls
from . import network as nw
from . import optimizer as opt
from . import toy_data as data


def train_model(
    num_iterations=None,
    schedule_type=None,
    activation_cls=layers.SiLU,
    optimizer=None,
    seed=0,
):
    # Configuration
    x_dim = config.X_DIM
    hidden_dim = config.HIDDEN_DIM

    if num_iterations is None:
        num_iterations = config.NUM_ITERATIONS
    if schedule_type is None:
        schedule_type = config.SCHEDULE_K
    num_data = config.NUM_DATA
    t_min = config.T_MIN
    batch_size = config.BATCH_SIZE

    rng = np.random.default_rng(seed)

    # Create objects
    # dataset = data.two_moons(num_data)
    dataset = data.two_moons_full(num_data)
    diffusion = sde.VPSDE(schedule_type)
    network = nw.Network(x_dim, hidden_dim, activation_cls)
    loss_function = ls.DSM_Loss()

    if optimizer is None:
        optimizer = opt.Adam()

    # Save training configuration in the model
    training_config = config.TrainingConfig(
        schedule_type=schedule_type,
        activation=activation_cls.__name__,
        optimizer=optimizer.__class__.__name__,
        learning_rate=optimizer.learning_rate,
        hidden_dim=hidden_dim,
        x_dim=x_dim,
        beta_min=diffusion.beta_min,
        beta_max=diffusion.beta_max,
        adj_coef=diffusion.a,
        T=diffusion.T,
    )

    network.training_config = training_config

    # Training
    loss_history = []
    for iteration in range(num_iterations):
        # Sample
        indices = rng.integers(0, num_data, size=batch_size)
        x0 = dataset[indices]
        t = rng.uniform(t_min, diffusion.T, size=batch_size).astype(np.float32)

        mean, var = diffusion.marginal_dist(x0, t)
        eps = rng.standard_normal(size=x0.shape).astype(x0.dtype)
        sigma = np.sqrt(var)
        xt = mean + sigma[:, None] * eps

        # Network
        prediction = network.forward(xt, t)

        # Loss
        loss = loss_function.forward(prediction, eps, sigma)

        # Backpropagation
        gradient = loss_function.backward()
        network.backward(gradient)

        # Optimization
        optimizer.step(network.parameters())

        # Print loss
        if iteration % 1000 == 0:
            loss_history.append((iteration, loss))

    print(
        f"Training completed: {num_iterations} iterations. Final batch loss = {loss:.4f}"
    )
    iterations, losses = zip(*loss_history)

    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    return network, loss_history


def save_model(model, filename=None):
    if filename is None:
        filename = config.MODEL_FILE

    model.save(filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    model, _ = train_model(
        activation_cls=layers.ReLU, optimizer=opt.Adam(learning_rate=1e-4)
    )
    save_model(model)
