import numpy as np
import pytest

from src.sde import VPSDE
from src.toy_data import two_moons


def test_two_moons_shape_and_finite():
    data = two_moons(500, seed=0)
    assert data.shape == (500, 2)
    assert np.all(np.isfinite(data))


def test_beta_is_linear_and_matches_endpoints():
    sde = VPSDE(beta_min=0.1, beta_max=20.0, T=1.0)
    assert sde.beta(0.0) == pytest.approx(0.1)
    assert sde.beta(1.0) == pytest.approx(20.0)
    assert sde.beta(0.5) == pytest.approx((0.1 + 20.0) / 2)


def test_integral_beta_matches_numerical_quadrature():
    sde = VPSDE()
    from scipy.integrate import quad

    for t in [0.1, 0.3, 0.7, 1.0]:
        numeric, _ = quad(sde.beta, 0.0, t)
        assert sde.integral_beta(t) == pytest.approx(numeric, rel=1e-6)


def test_marginal_prob_matches_hand_computation_at_t0():
    # At t=0, alpha_bar(0) = exp(0) = 1, so mean == x0 and std == 0.
    sde = VPSDE()
    x0 = np.array([[1.0, -2.0], [0.5, 0.5]], dtype=np.float32)
    mean, std = sde.marginal_prob(x0, np.array(0.0, dtype=np.float32))
    np.testing.assert_allclose(mean, x0, atol=1e-6)
    np.testing.assert_allclose(std, np.zeros_like(x0), atol=1e-6)


def test_marginal_prob_approaches_standard_normal_at_T():
    # At t=T=1 with beta_max=20, alpha_bar(1) is essentially 0, so the
    # marginal should be very close to N(0, I) regardless of x0.
    sde = VPSDE()
    x0 = np.array([[3.0, -5.0]], dtype=np.float32)
    mean, std = sde.marginal_prob(x0, np.array(1.0, dtype=np.float32))
    np.testing.assert_allclose(mean, np.zeros_like(x0), atol=0.05)
    np.testing.assert_allclose(std, np.ones_like(x0), atol=1e-2)


def test_euler_maruyama_matches_closed_form_marginal():
    """The core rigor check: simulate the forward SDE many times from a
    fixed x0 and confirm the empirical marginal (mean, std) at several t
    matches the closed-form Gaussian marginal within Monte Carlo error.
    """
    sde = VPSDE()
    rng = np.random.default_rng(42)
    data = two_moons(200, seed=1)
    x0_single = data[7]

    n_particles = 20000
    x0_batch = np.tile(x0_single, (n_particles, 1)).astype(np.float32)
    traj, ts = sde.simulate_forward(x0_batch, n_steps=1000, rng=rng)

    # Monte Carlo std error of a mean estimate from n_particles samples of
    # a unit-scale Gaussian is ~1/sqrt(n_particles) ~ 0.007; use a generous
    # tolerance so the test isn't flaky.
    for t_check in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = int(np.argmin(np.abs(ts - t_check)))
        sim = traj[idx]
        mean_cf, std_cf = sde.marginal_prob(
            x0_single[None, :], np.array(ts[idx], dtype=np.float32)
        )

        np.testing.assert_allclose(sim.mean(axis=0), mean_cf[0], atol=0.05)
        np.testing.assert_allclose(sim.std(axis=0), std_cf[0], atol=0.05)


def test_true_score_is_consistent_with_marginal_prob():
    # For a Gaussian N(mean, std^2), the score at x_t = mean is exactly 0.
    sde = VPSDE()
    x0 = np.array([[1.0, 1.0]], dtype=np.float32)
    t = np.array(0.5, dtype=np.float32)
    mean, _ = sde.marginal_prob(x0, t)
    score_at_mean = sde.true_score(x0, mean, t)
    np.testing.assert_allclose(score_at_mean, 0.0, atol=1e-6)
