import numpy as np
from scipy.integrate import quad

from hawkes import (
    calculate_integrated_intensity,
    calculate_log_likelihood,
    hawkes_intensity,
)
from model_comparison import calculate_poisson_log_likelihood


def test_integrated_intensity_matches_numerical_quadrature():
    """
    The closed-form integrated intensity used inside the
    log-likelihood must agree with a brute-force numerical
    integration of the same intensity function.
    """

    event_times = np.array([0.5, 1.0, 1.3, 2.7, 5.0, 9.3])
    mu, alpha, beta = 1.0, 0.5, 2.0
    T = event_times[-1]

    analytic = calculate_integrated_intensity(event_times, mu, alpha, beta)

    numerical, _ = quad(
        lambda t: hawkes_intensity(t, event_times, mu, alpha, beta),
        0,
        T,
        points=list(event_times),
        limit=200
    )

    assert abs(analytic - numerical) < 1e-6


def test_zero_alpha_reduces_to_poisson_log_likelihood():
    """
    With alpha = 0 there is no self-excitation, so the Hawkes
    log-likelihood must reduce exactly to the homogeneous Poisson
    log-likelihood at rate mu.
    """

    event_times = np.array([0.5, 1.0, 1.3, 2.7, 5.0, 9.3])
    mu = 1.5

    hawkes_ll = calculate_log_likelihood(event_times, mu, alpha=0.0, beta=1.0)
    poisson_ll = calculate_poisson_log_likelihood(event_times, mu)

    assert abs(hawkes_ll - poisson_ll) < 1e-9
