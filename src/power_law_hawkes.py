import numpy as np
from scipy.optimize import minimize

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    calculate_log_likelihood,
    estimate_hawkes_parameters,
    prepare_hawkes_event_times,
)
from trade_processing import process_trade_data

# How far into the past the power-law kernel's contribution is treated
# as negligible and truncated to zero. This is a computational
# approximation, not part of the model itself: an exact power-law
# Hawkes likelihood has no Markov/recursive shortcut (unlike the
# exponential kernel in hawkes.py), so summing over *all* previous
# events for *every* event is O(n^2) and intractable for the ~2*10^5
# events in this dataset.
#
# The exponential-kernel fit in hawkes.py found beta in the range of
# roughly 40-140 across sub-samples of this data (see
# nonstationarity.py), i.e. a decay half-life on the order of
# 0.005-0.02 seconds. A 5-second lookback window is therefore several
# hundred to a thousand times longer than the exponential kernel's
# characteristic timescale, and comfortably covers the lag range
# (10^-4 to 1 second) over which aftershock_analysis.py finds
# excitation to be concentrated. The fitted power-law parameters are
# checked against this window after fitting (see main block) to
# confirm the truncation is still a good approximation post-fit.
LOOKBACK_WINDOW_SECONDS = 5.0


def power_law_kernel(t, alpha, c, p):
    """
    Evaluate the power-law (Omori-Utsu style) excitation kernel.

        g(t) = alpha / (t + c)^p,   t >= 0

    where:

        alpha > 0   overall excitation strength
        c     > 0   offset avoiding the singularity at t = 0
        p     > 1   tail decay exponent (p > 1 is required for the
                    kernel to be integrable, i.e. for a finite
                    branching ratio)

    Unlike the exponential kernel alpha*exp(-beta*t), this kernel
    decays as a power law and therefore has a "fat" tail: excitation
    persists longer than an exponential kernel with a comparable
    initial value.
    """

    return alpha / (t + c) ** p


def calculate_power_law_branching_ratio(alpha, c, p):
    """
    Calculate the branching ratio of the power-law kernel.

        n = integral_0^inf g(t) dt
          = integral_0^inf alpha / (t + c)^p dt
          = alpha * c^(1-p) / (p - 1),   for p > 1

    As with the exponential kernel, a stationary process requires
    n < 1.
    """

    return alpha * c ** (1 - p) / (p - 1)


def is_power_law_stable(alpha, c, p):
    """
    Check the stability condition n = alpha * c^(1-p) / (p-1) < 1.
    """

    return calculate_power_law_branching_ratio(alpha, c, p) < 1


def calculate_integrated_intensity_power_law(event_times, mu, alpha, c, p):
    """
    Calculate the integrated power-law Hawkes intensity over [0, T].

    For event t_i, the exact contribution of its excitation to the
    integral over [t_i, T] has a closed form (no truncation needed
    here -- this sum is O(n) via vectorized numpy regardless of the
    lookback-window approximation used for the log-intensity term):

        integral_{t_i}^{T} alpha / (s - t_i + c)^p ds
            = alpha / (p - 1) * [ c^(1-p) - (T - t_i + c)^(1-p) ]

    The total integrated intensity is the baseline term mu*T plus the
    sum of this excitation contribution across every event.
    """

    T = event_times[-1]

    baseline_integral = mu * T

    excitation_integral = (alpha / (p - 1)) * np.sum(
        c ** (1 - p) - (T - event_times + c) ** (1 - p)
    )

    return baseline_integral + excitation_integral


def build_lookback_pairs(event_times, lookback_window):
    """
    Precompute, once, every (event, prior-event-within-window) pair
    needed to evaluate the truncated log-intensity sum.

    This precomputation depends only on the event times and the fixed
    lookback window -- NOT on mu, alpha, c, or p -- so it only needs
    to be done once per dataset, no matter how many times the
    log-likelihood is subsequently evaluated during MLE optimization.

    For each event i, every prior event j with
    event_times[i] - lookback_window <= event_times[j] < event_times[i]
    is included. The result is returned as two flat arrays:

        group_id[k]  = the index i of the "current" event for pair k
        delta_t[k]   = event_times[i] - event_times[j] > 0 for pair k

    so that, for any kernel parameters, the excitation received by
    event i can be recovered as the sum of delta_t entries whose
    group_id equals i (see calculate_log_intensity_sum_power_law).
    """

    n = len(event_times)

    # Lower index bound of the lookback window for every event,
    # found in one vectorized O(n log n) pass.
    lower_bounds = np.searchsorted(
        event_times,
        event_times - lookback_window
    )

    indices = np.arange(n)

    # Number of prior events within the lookback window, for every
    # event. This can be zero (e.g. the very first events).
    counts = indices - lower_bounds

    total_pairs = int(counts.sum())

    # Flatten the ragged (variable-length) per-event windows into a
    # single pair of arrays, entirely with vectorized numpy ops (no
    # Python-level loop over events).
    group_id = np.repeat(indices, counts)

    # Position of each pair within its own group: 0, 1, ..., counts[i]-1.
    group_offsets = np.repeat(
        np.cumsum(counts) - counts,
        counts
    )
    position_in_group = np.arange(total_pairs) - group_offsets

    prior_indices = np.repeat(lower_bounds, counts) + position_in_group

    delta_t = event_times[group_id] - event_times[prior_indices]

    return group_id, delta_t, n


def calculate_log_intensity_sum_power_law(precomputed_pairs, mu, alpha, c, p):
    """
    Calculate Sum_i log(lambda(t_i)) for the truncated power-law
    Hawkes model, using precomputed (group_id, delta_t) pairs.

    For each event i:

        lambda(t_i) = mu + Sum_{j: pair (i,j) precomputed} g(delta_t_ij)

    np.bincount aggregates the per-pair contributions back into a
    per-event excitation total in one vectorized pass; events with no
    prior events inside the lookback window correctly receive an
    excitation of 0 (bincount, unlike np.add.reduceat, handles empty
    groups without special-casing).
    """

    group_id, delta_t, n = precomputed_pairs

    contributions = power_law_kernel(delta_t, alpha, c, p)

    excitation = np.bincount(group_id, weights=contributions, minlength=n)

    intensities = mu + excitation

    return np.sum(np.log(intensities))


def calculate_log_likelihood_power_law(
    event_times,
    precomputed_pairs,
    mu,
    alpha,
    c,
    p
):
    """
    Calculate the (truncated) power-law Hawkes log-likelihood.

        log L = Sum_i log(lambda(t_i)) - integral_0^T lambda(t) dt
    """

    if mu <= 0 or alpha < 0 or c <= 0 or p <= 1:
        raise ValueError(
            "Parameters mu and c must be positive, p must be "
            "greater than 1, and alpha must be non-negative."
        )

    return (
        calculate_log_intensity_sum_power_law(
            precomputed_pairs,
            mu,
            alpha,
            c,
            p
        )
        - calculate_integrated_intensity_power_law(
            event_times,
            mu,
            alpha,
            c,
            p
        )
    )


def negative_log_likelihood_power_law(params, event_times, precomputed_pairs):

    mu, alpha, c, p = params

    return -calculate_log_likelihood_power_law(
        event_times,
        precomputed_pairs,
        mu,
        alpha,
        c,
        p
    )


def estimate_power_law_hawkes_parameters(
    event_times,
    lookback_window=LOOKBACK_WINDOW_SECONDS
):
    """
    Estimate power-law Hawkes parameters (mu, alpha, c, p) using
    maximum likelihood, truncating the excitation kernel's memory to
    `lookback_window` seconds for tractability.

    Returns the scipy optimization result.
    """

    precomputed_pairs = build_lookback_pairs(event_times, lookback_window)

    initial_params = [
        1.0,   # mu
        1.0,   # alpha
        0.01,  # c
        1.5,   # p
    ]

    bounds = [
        (1e-6, None),   # mu > 0
        (0, None),       # alpha >= 0
        (1e-6, None),    # c > 0
        (1 + 1e-3, None),  # p > 1
    ]

    result = minimize(
        negative_log_likelihood_power_law,
        initial_params,
        args=(event_times, precomputed_pairs),
        bounds=bounds
    )

    return result


if __name__ == "__main__":

    # ==========================================================
    # TOY EXAMPLE: validate the vectorized bincount-based
    # log-intensity sum against a brute-force O(n^2) reference.
    # ==========================================================

    toy_event_times = np.array([1.0, 2.0, 4.0, 4.5, 9.0])
    toy_mu, toy_alpha, toy_c, toy_p = 1.0, 0.5, 0.1, 1.5

    def brute_force_log_intensity_sum(event_times, mu, alpha, c, p):
        total = 0.0
        for i, t in enumerate(event_times):
            past = event_times[:i]
            excitation = np.sum(power_law_kernel(t - past, alpha, c, p))
            total += np.log(mu + excitation)
        return total

    brute_force_value = brute_force_log_intensity_sum(
        toy_event_times, toy_mu, toy_alpha, toy_c, toy_p
    )

    # Use a lookback window larger than the entire toy series so that
    # nothing is truncated -- this should exactly match the brute
    # force reference.
    toy_pairs = build_lookback_pairs(toy_event_times, lookback_window=100.0)

    vectorized_value = calculate_log_intensity_sum_power_law(
        toy_pairs, toy_mu, toy_alpha, toy_c, toy_p
    )

    print("=" * 60)
    print("TOY VALIDATION: brute force vs vectorized log-intensity sum")
    print("=" * 60)
    print("Brute force:", brute_force_value)
    print("Vectorized: ", vectorized_value)
    print("Difference: ", brute_force_value - vectorized_value)

    toy_integrated = calculate_integrated_intensity_power_law(
        toy_event_times, toy_mu, toy_alpha, toy_c, toy_p
    )
    print("\nToy integrated intensity:", toy_integrated)

    toy_branching_ratio = calculate_power_law_branching_ratio(
        toy_alpha, toy_c, toy_p
    )
    print("Toy branching ratio:", toy_branching_ratio)

    # A truncated lookback window (shorter than the series) should
    # differ from the untruncated brute-force value, but only because
    # some real excitation is being dropped -- sanity check that it's
    # in the same ballpark, not wildly different.
    toy_pairs_truncated = build_lookback_pairs(
        toy_event_times, lookback_window=2.0
    )
    truncated_value = calculate_log_intensity_sum_power_law(
        toy_pairs_truncated, toy_mu, toy_alpha, toy_c, toy_p
    )
    print("\nTruncated (window=2.0) log-intensity sum:", truncated_value)

    # ==========================================================
    # REAL BTCUSDT DATA
    # ==========================================================

    print("\n" + "=" * 60)
    print("REAL BTCUSDT DATA")
    print("=" * 60)

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    event_times = prepare_hawkes_event_times(processed_data)

    print("\nNumber of event times:", len(event_times))
    print(f"Observation period: {event_times[-1]:.2f} seconds")

    import time

    print(
        f"\nBuilding lookback pairs "
        f"(window = {LOOKBACK_WINDOW_SECONDS}s)..."
    )
    start = time.time()
    precomputed_pairs = build_lookback_pairs(
        event_times, LOOKBACK_WINDOW_SECONDS
    )
    build_seconds = time.time() - start
    group_id, delta_t, n = precomputed_pairs
    print(f"Built {len(delta_t):,} pairs in {build_seconds:.2f}s")
    print(f"Average prior events per event within window: "
          f"{len(delta_t) / n:.2f}")

    print("\nFitting power-law Hawkes model via MLE...")
    start = time.time()
    result = estimate_power_law_hawkes_parameters(
        event_times, lookback_window=LOOKBACK_WINDOW_SECONDS
    )
    fit_seconds = time.time() - start
    print(f"Fit completed in {fit_seconds:.2f}s")
    print("\nOptimization result:")
    print(result)

    pl_mu, pl_alpha, pl_c, pl_p = result.x

    print("\nEstimated power-law parameters:")
    print(f"mu    = {pl_mu:.6f}")
    print(f"alpha = {pl_alpha:.6f}")
    print(f"c     = {pl_c:.6f}")
    print(f"p     = {pl_p:.6f}")

    pl_branching_ratio = calculate_power_law_branching_ratio(
        pl_alpha, pl_c, pl_p
    )
    print("\nBranching ratio:", pl_branching_ratio)
    print("Stable:", is_power_law_stable(pl_alpha, pl_c, pl_p))

    # Confirm the lookback-window truncation is still a good
    # approximation for the *fitted* parameters: how small is the
    # kernel at the edge of the window relative to its peak value?
    edge_ratio = (
        power_law_kernel(LOOKBACK_WINDOW_SECONDS, pl_alpha, pl_c, pl_p)
        / power_law_kernel(0.0, pl_alpha, pl_c, pl_p)
    )
    print(
        f"\ng(window_edge) / g(0) at the fitted parameters: "
        f"{edge_ratio:.2e} "
        f"(smaller is a better-justified truncation)"
    )

    pl_log_likelihood = calculate_log_likelihood_power_law(
        event_times, precomputed_pairs, pl_mu, pl_alpha, pl_c, pl_p
    )
    print("\nPower-law Hawkes log-likelihood:", pl_log_likelihood)

    # ----------------------------------------------------------
    # For reference, refit the exponential-kernel model on the
    # exact same event series (full dataset, no subsampling was
    # needed for the power-law fit above).
    # ----------------------------------------------------------

    print("\nFitting exponential-kernel Hawkes model for comparison...")
    exp_result = estimate_hawkes_parameters(event_times)
    exp_mu, exp_alpha, exp_beta = exp_result.x
    exp_log_likelihood = calculate_log_likelihood(
        event_times, exp_mu, exp_alpha, exp_beta
    )
    exp_branching_ratio = calculate_branching_ratio(exp_alpha, exp_beta)

    print(f"mu={exp_mu:.6f}, alpha={exp_alpha:.6f}, beta={exp_beta:.6f}")
    print(f"Exponential branching ratio: {exp_branching_ratio:.6f}")
    print(f"Exponential log-likelihood: {exp_log_likelihood:.6f}")

    print("\n" + "=" * 60)
    print("EXPONENTIAL vs POWER-LAW (same full dataset)")
    print("=" * 60)
    print(f"Exponential log-likelihood: {exp_log_likelihood:.6f} (3 params)")
    print(f"Power-law log-likelihood:   {pl_log_likelihood:.6f} (4 params)")
    print(
        f"Power-law - exponential:    "
        f"{pl_log_likelihood - exp_log_likelihood:.6f}"
    )
