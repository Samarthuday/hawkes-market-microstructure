import numpy as np

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    estimate_hawkes_parameters,
    negative_log_likelihood,
    prepare_hawkes_event_times,
)
from time_window_robustness import run_time_window_robustness
from trade_processing import process_trade_data


def numerical_hessian(f, x, relative_step=1e-4):
    """
    Estimate the Hessian of a scalar function f at x using central
    finite differences, with a per-coordinate step size scaled to
    that coordinate's own magnitude:

        step_k = relative_step * max(1, |x_k|)

    A single fixed step size would be a poor choice here because
    mu, alpha, and beta live on very different scales (mu ~ 1,
    alpha ~ 10, beta ~ 60 for this dataset); a relative step keeps
    the finite-difference error roughly comparable across
    coordinates.
    """

    n = len(x)
    steps = relative_step * np.maximum(1.0, np.abs(x))

    hessian = np.zeros((n, n))
    f0 = f(x)

    f_plus = np.zeros(n)
    for i in range(n):
        x_i = x.copy()
        x_i[i] += steps[i]
        f_plus[i] = f(x_i)

    for i in range(n):
        for j in range(i, n):

            x_ij = x.copy()
            x_ij[i] += steps[i]
            x_ij[j] += steps[j]

            second_derivative = (
                f(x_ij) - f_plus[i] - f_plus[j] + f0
            ) / (steps[i] * steps[j])

            hessian[i, j] = second_derivative
            hessian[j, i] = second_derivative

    return hessian


def calculate_asymptotic_standard_errors(event_times, mu, alpha, beta):
    """
    Estimate asymptotic standard errors for (mu, alpha, beta) via the
    inverse of the numerical Hessian of the negative log-likelihood
    at the MLE (the observed Fisher information), then propagate to
    the branching ratio n = alpha / beta and half-life
    t_1/2 = ln(2) / beta via the delta method:

        Var(g(theta)) ~= grad(g)^T * Cov(theta) * grad(g)

    These standard errors reflect *sampling noise given the
    exponential-kernel Hawkes model is correctly specified* -- they
    do NOT capture the additional uncertainty from potential model
    misspecification (e.g. the nonstationarity documented in
    nonstationarity.py). See calculate_block_based_spread below for
    a complementary, model-free uncertainty estimate that does
    reflect that.
    """

    x = np.array([mu, alpha, beta])

    hessian = numerical_hessian(
        lambda params: negative_log_likelihood(params, event_times),
        x
    )

    covariance = np.linalg.inv(hessian)
    standard_errors = np.sqrt(np.diag(covariance))

    se_mu, se_alpha, se_beta = standard_errors

    # Delta method for n = alpha / beta.
    grad_n = np.array([0.0, 1.0 / beta, -alpha / beta ** 2])
    var_n = grad_n @ covariance @ grad_n
    se_n = np.sqrt(var_n)

    # Delta method for t_1/2 = ln(2) / beta.
    d_half_life_d_beta = -np.log(2) / beta ** 2
    var_half_life = (d_half_life_d_beta ** 2) * covariance[2, 2]
    se_half_life = np.sqrt(var_half_life)

    return {
        "mu": (mu, se_mu),
        "alpha": (alpha, se_alpha),
        "beta": (beta, se_beta),
        "branching_ratio": (
            calculate_branching_ratio(alpha, beta), se_n
        ),
        "half_life": (np.log(2) / beta, se_half_life),
    }


def calculate_block_based_spread(event_times, number_of_blocks=5):
    """
    A model-free, complementary uncertainty estimate: the mean and
    standard deviation of the branching ratio across independently
    re-fit time blocks (reusing time_window_robustness.py's block
    partition). Because nonstationarity.py already established that
    mu, alpha, and beta genuinely drift over this sample, the
    block-to-block spread reflects real variation over time, not
    just estimation noise -- it is expected to be, and should be,
    larger than the asymptotic standard error above.
    """

    block_results = run_time_window_robustness(
        event_times, number_of_blocks=number_of_blocks
    )

    return {
        "branching_ratio_mean": block_results["branching_ratio"].mean(),
        "branching_ratio_std": block_results["branching_ratio"].std(),
        "n_blocks": len(block_results),
    }


def format_ci(estimate, se, z=1.96):
    return f"{estimate:.6f} +/- {z * se:.6f} (95% CI)"


if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )
    processed_data = process_trade_data(data)
    event_times = prepare_hawkes_event_times(processed_data)

    print("=" * 60)
    print("PARAMETER UNCERTAINTY: ASYMPTOTIC (HESSIAN-BASED)")
    print("=" * 60)

    fit_result = estimate_hawkes_parameters(event_times)
    mu, alpha, beta = fit_result.x

    print(f"\nPoint estimates: mu={mu:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")

    print("\nComputing numerical Hessian and asymptotic standard errors...")
    asymptotic = calculate_asymptotic_standard_errors(
        event_times, mu, alpha, beta
    )

    for name, (estimate, se) in asymptotic.items():
        print(f"{name:>18}: {format_ci(estimate, se)}")

    print("\n" + "=" * 60)
    print("PARAMETER UNCERTAINTY: BLOCK-BASED SPREAD (MODEL-FREE)")
    print("=" * 60)

    block_spread = calculate_block_based_spread(event_times, number_of_blocks=5)

    print(
        f"\nBranching ratio across {block_spread['n_blocks']} independent "
        f"time blocks: mean={block_spread['branching_ratio_mean']:.4f}, "
        f"std={block_spread['branching_ratio_std']:.4f}"
    )

    asymptotic_n_se = asymptotic["branching_ratio"][1]
    print(
        f"\nAsymptotic (sampling-noise-only) branching ratio SE: "
        f"{asymptotic_n_se:.4f}"
    )
    print(
        f"Block-to-block (nonstationarity-driven) branching ratio std: "
        f"{block_spread['branching_ratio_std']:.4f}"
    )
    print(
        f"Ratio (block std / asymptotic SE): "
        f"{block_spread['branching_ratio_std'] / asymptotic_n_se:.2f}x"
    )

    if block_spread["branching_ratio_std"] > asymptotic_n_se:
        print(
            "\nINTERPRETATION: the branching ratio varies across time "
            "blocks by more than pure sampling noise would predict -- "
            "consistent with the nonstationarity already documented in "
            "nonstationarity.py. Reporting only the asymptotic SE would "
            "understate the true uncertainty in the branching ratio."
        )
