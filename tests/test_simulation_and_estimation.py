import numpy as np
from scipy.stats import kstest

from hawkes import (
    calculate_branching_ratio,
    calculate_time_rescaled_intervals,
    estimate_hawkes_parameters,
    is_hawkes_stable,
    simulate_hawkes,
)


def test_mle_recovers_known_parameters_from_simulated_data():
    """
    Simulating from a known (mu, alpha, beta) and re-fitting via MLE
    should recover parameters close to the truth. Tolerances are
    generous on purpose -- this is a stochastic simulation, not a
    deterministic check -- but a badly broken likelihood or
    optimizer would fail this by a wide margin, not a narrow one.
    """

    true_mu, true_alpha, true_beta = 1.0, 2.0, 5.0

    simulated = np.array(
        simulate_hawkes(
            true_mu, true_alpha, true_beta,
            observation_time=5000.0,
            seed=123
        )
    )
    assert len(simulated) > 500

    result = estimate_hawkes_parameters(simulated)
    mu, alpha, beta = result.x

    true_n = true_alpha / true_beta
    fitted_n = alpha / beta

    assert abs(fitted_n - true_n) < 0.15
    assert abs(mu - true_mu) / true_mu < 0.3


def test_estimator_always_returns_a_stable_fit():
    """
    estimate_hawkes_parameters is reparameterized so that n < 1 by
    construction (see hawkes.py) -- this should hold regardless of
    the input data.
    """

    simulated = np.array(
        simulate_hawkes(1.0, 2.0, 5.0, observation_time=3000.0, seed=42)
    )

    result = estimate_hawkes_parameters(simulated)
    mu, alpha, beta = result.x

    assert is_hawkes_stable(alpha, beta)


def test_time_rescaled_intervals_are_approximately_exp1():
    """
    For data simulated from a known Hawkes model, time-rescaling
    with the TRUE parameters should produce intervals that behave
    like i.i.d. Exp(1) draws -- this is the theoretical basis for
    the goodness-of-fit diagnostics used throughout the project.
    """

    mu, alpha, beta = 1.0, 2.0, 5.0

    simulated = np.array(
        simulate_hawkes(mu, alpha, beta, observation_time=20000.0, seed=7)
    )

    rescaled = calculate_time_rescaled_intervals(simulated, mu, alpha, beta)[1:]

    assert abs(rescaled.mean() - 1.0) < 0.1
    assert abs(rescaled.var() - 1.0) < 0.2

    ks_statistic, _ = kstest(rescaled, "expon", args=(0, 1))
    assert ks_statistic < 0.03


def test_branching_ratio_and_stability_formulas():

    assert calculate_branching_ratio(10.0, 20.0) == 0.5
    assert is_hawkes_stable(5.0, 10.0) is True
    assert is_hawkes_stable(15.0, 10.0) is False
