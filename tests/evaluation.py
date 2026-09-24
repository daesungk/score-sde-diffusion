import numpy as np
import ot


def WassersteinDistance(samples, data):
    n = len(samples)
    m = len(data)

    u_sample = np.ones(n) / n
    u_data = np.ones(m) / m

    cost = ot.dist(samples, data, metric="sqeuclidean")

    w2_squared = ot.emd2(u_sample, u_data, cost)

    return np.sqrt(w2_squared)


def Gaussian_pdf(x, mu, sigma):
    # x:  (B, d)
    # mu: (N, d)
    # output: (B, N)

    dim = x.shape[1]

    diff = x[:, None, :] - mu[None, :, :]

    coeff = (2 * np.pi * sigma**2) ** (-dim / 2)
    exponent = np.sum(diff**2, axis=2) / (2 * sigma**2)

    return coeff * np.exp(-exponent)


def ExactScore(x, t, diffusion, data):
    beta_int = diffusion.beta_int(t)
    alpha_bar = np.exp(-beta_int)

    mean_coeff = np.sqrt(alpha_bar)
    var = 1 - alpha_bar
    sigma = np.sqrt(var)

    # Means: (N, d)
    mu = mean_coeff * data

    # Gaussian densities: (B, N)
    pdf = Gaussian_pdf(x, mu, sigma)

    # Mixture weights: (B, N)
    weight = pdf / np.sum(pdf, axis=1, keepdims=True)

    # Component scores: (B, N, d)
    component_score = (mu[None, :, :] - x[:, None, :]) / var

    # Weighted sum over mixture components: (B, d)
    score = np.sum(
        weight[:, :, None] * component_score,
        axis=1,
    )

    return score


def ScoreMSE(model, x, t, diffusion, data):
    exact_score = ExactScore(x, t, diffusion, data)

    t_batch = np.full(
        x.shape[0],
        t,
        dtype=np.float32,
    )

    predicted_score = model.forward(x, t_batch)

    return np.mean(np.sum((predicted_score - exact_score) ** 2, axis=1))
