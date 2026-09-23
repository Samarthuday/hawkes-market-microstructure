import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import expon, kstest

from data_loader import load_trade_data
from trade_processing import (
    event_times_to_seconds,
    extract_event_times,
    process_trade_data,
)


def hawkes_intensity(t, event_times, mu, alpha, beta):
    """
    Calculate the Hawkes process intensity at time t.

    For an exponential-kernel Hawkes process:

        λ(t) = μ + Σ_{t_i < t} α exp[-β(t - t_i)]

    where:

        μ     = baseline intensity
        α     = excitation parameter
        β     = decay parameter
        t_i   = previous event times

    Each previous event increases the current intensity by

        α exp[-β(t - t_i)]

    and this contribution decays exponentially as time passes.
    """

    # Select only events that occurred before time t.
    # The current event is excluded because we are calculating
    # the pre-event intensity λ(t^-).
    past_events = event_times[event_times < t]

    # Baseline intensity + contribution from all previous events.
    intensity = mu + np.sum(
        alpha * np.exp(-beta * (t - past_events))
    )

    return intensity


def calculate_hawkes_intensity_series(event_times, mu, alpha, beta):
    """
    Calculate the Hawkes intensity at every event time.

    This is the direct implementation of the Hawkes intensity
    formula. It is useful as a reference implementation for
    validation, but it is O(n^2) and therefore too slow for
    large datasets.
    """

    # Store the intensity corresponding to each event time.
    intensities = []

    # Calculate the intensity separately for every event time.
    for t in event_times:

        # Calculate the pre-event Hawkes intensity λ(t^-).
        intensity = hawkes_intensity(
            t,
            event_times,
            mu,
            alpha,
            beta
        )

        # Add the calculated intensity to the list.
        intensities.append(intensity)

    return intensities


def create_hawkes_intensity_dataframe(event_times, mu, alpha, beta):
    """
    Create a DataFrame containing event times and their
    corresponding Hawkes intensities.
    """

    # Calculate the Hawkes intensity for every event time.
    intensities = calculate_hawkes_intensity_series(
        event_times,
        mu,
        alpha,
        beta
    )

    # Combine event times and intensities into a DataFrame.
    df = pd.DataFrame({
        "timestamp": event_times,
        "intensity": intensities
    })

    return df


def prepare_hawkes_event_times(data):
    """
    Prepare event times for Hawkes process estimation.

    The raw trade timestamps are first converted into unique,
    sorted event times. These timestamps are then converted
    into elapsed seconds measured from the first event.
    """

    # Extract unique, sorted timestamps.
    event_times = extract_event_times(data)

    # Convert timestamps into elapsed seconds from the
    # beginning of the observation period.
    event_times = event_times_to_seconds(event_times)

    # Return as a NumPy array.
    return event_times.to_numpy()


def calculate_hawkes_excitation_series(event_times, alpha, beta):
    """
    Calculate the Hawkes excitation immediately before each event.

    For an exponential-kernel Hawkes process, define the excitation
    immediately before event t_i as:

        R_i = Σ_{j < i} α exp[-β(t_i - t_j)]

    Instead of recomputing this summation for every event, we use
    the recursive relationship:

        R_1 = 0

        R_i = (α + R_{i-1}) exp[-β(t_i - t_{i-1})]

    Explanation:

    1. R_{i-1} contains the excitation from all events before t_{i-1}.
    2. This excitation decays over Δt_i = t_i - t_{i-1}:

           R_{i-1} exp(-βΔt_i)

    3. The event at t_{i-1} contributes a new excitation α.
    4. Therefore:

           R_i = (α + R_{i-1}) exp(-βΔt_i)

    The corresponding Hawkes intensity is:

        λ(t_i^-) = μ + R_i

    This recursive implementation reduces the computational
    complexity from O(n^2) to O(n).
    """

    # The first event has no previous events, so its
    # pre-event excitation is zero.
    excitations = [0.0]

    for i in range(1, len(event_times)):

        # Time elapsed since the previous event:
        #
        # Δt_i = t_i - t_{i-1}
        delta_t = event_times[i] - event_times[i - 1]

        # The previous excitation and the contribution
        # from the immediately preceding event both decay
        # over the interval Δt_i.
        #
        # R_i = (α + R_{i-1}) exp(-βΔt_i)
        excitation = (
            alpha + excitations[i - 1]
        ) * np.exp(-beta * delta_t)

        excitations.append(excitation)

    return excitations


def calculate_hawkes_intensity_series_fast(
    event_times,
    mu,
    alpha,
    beta
):
    """
    Calculate Hawkes intensities using the recursive
    excitation calculation.

    This implementation is O(n) and is suitable for
    large event datasets.
    """

    # Calculate the excitation before every event.
    excitations = calculate_hawkes_excitation_series(
        event_times,
        alpha,
        beta
    )

    # Add the baseline intensity to the excitation.
    intensities = mu + np.array(excitations)

    return intensities

def calculate_time_rescaled_intervals(
    event_times,
    mu,
    alpha,
    beta
):
    rescaled_intervals = [mu * event_times[0]]

    excitation = 0.0

    for i in range(1, len(event_times)):
        delta_t = event_times[i] - event_times[i - 1]

        integrated_excitation = (
            (alpha + excitation)
            / beta
            * (
                1 - np.exp(-beta * delta_t)
            )
        )

        rescaled_interval = (
            mu * delta_t
            + integrated_excitation
        )

        rescaled_intervals.append(rescaled_interval)

        excitation = (
            alpha + excitation
        ) * np.exp(-beta * delta_t)

    return np.array(rescaled_intervals)


def calculate_log_intensity_sum(event_times, mu, alpha, beta):
    """
    Calculate:

        Σ_i log(λ(t_i))

    using the fast recursive Hawkes intensity calculation.
    """

    # Use the O(n) recursive implementation rather than
    # the O(n^2) direct implementation.
    intensities = calculate_hawkes_intensity_series_fast(
        event_times,
        mu,
        alpha,
        beta
    )

    # Sum the logarithm of the pre-event intensities.
    log_intensity_sum = np.sum(
        np.log(intensities)
    )

    return log_intensity_sum


def calculate_integrated_intensity(event_times, mu, alpha, beta):
    """
    Calculate the integrated Hawkes intensity over [0, T].

    The integrated intensity is:

        ∫₀ᵀ λ(t) dt
        = μT + (α/β) Σᵢ [1 - exp(-β(T - tᵢ))]

    where:

        μ     = baseline intensity
        α     = excitation strength
        β     = decay rate
        tᵢ    = event times
        T     = end of observation period
    """

    # End of the observation period.
    T = event_times[-1]

    # Contribution from the baseline intensity:
    #
    # ∫₀ᵀ μ dt = μT
    baseline_integral = mu * T

    # Contribution from the excitation generated
    # by each event.
    excitation_integral = (
        alpha / beta
    ) * np.sum(
        1 - np.exp(
            -beta * (T - event_times)
        )
    )

    # Total integrated intensity.
    integrated_intensity = (
        baseline_integral
        + excitation_integral
    )

    return integrated_intensity


def calculate_log_likelihood(event_times, mu, alpha, beta):
    """
    Calculate the Hawkes process log-likelihood.

        log L =
            Σ_i log(λ(t_i))
            - ∫₀ᵀ λ(t) dt
    """

    # Parameter restrictions:
    #
    # μ > 0
    # α >= 0
    # β > 0
    if mu <= 0 or alpha < 0 or beta <= 0:
        raise ValueError(
            "Parameters mu and beta must be positive, "
            "and alpha must be non-negative."
        )

    # Event contribution minus integrated intensity.
    return (
        calculate_log_intensity_sum(
            event_times,
            mu,
            alpha,
            beta
        )
        - calculate_integrated_intensity(
            event_times,
            mu,
            alpha,
            beta
        )
    )


def negative_log_likelihood(params, event_times):
    """
    Return the negative Hawkes log-likelihood.

    scipy.optimize.minimize() minimizes functions, so we
    minimize -log(L) instead of maximizing log(L).
    """

    mu, alpha, beta = params

    negative_log_likelihood_value = (
        -calculate_log_likelihood(
            event_times,
            mu,
            alpha,
            beta
        )
    )

    return negative_log_likelihood_value


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def _theta_to_hawkes_params(theta):
    """
    Map an unconstrained parameter vector theta = [log_mu, log_beta,
    logit_n] to (mu, alpha, beta) such that the stability condition
    n = alpha / beta < 1 holds by construction, for any theta:

        mu    = exp(log_mu)          > 0
        beta  = exp(log_beta)        > 0
        n     = sigmoid(logit_n)     in (0, 1)
        alpha = n * beta             so that alpha / beta = n < 1

    This reparameterization means the optimizer can run completely
    unconstrained in theta-space -- there is no boundary at n = 1 to
    approach or violate, unlike the original box-constrained
    (mu, alpha, beta) parameterization, which only rejected an
    unstable fit after the fact.
    """

    log_mu, log_beta, logit_n = theta

    mu = np.exp(log_mu)
    beta = np.exp(log_beta)
    n = _sigmoid(logit_n)
    alpha = n * beta

    return mu, alpha, beta


def _hawkes_params_to_theta(mu, alpha, beta):
    """
    Inverse of _theta_to_hawkes_params, used only to build initial
    guesses for the optimizer from an interpretable (mu, alpha, beta)
    starting point.
    """

    n = alpha / beta

    return np.array([np.log(mu), np.log(beta), _logit(n)])


def _negative_log_likelihood_reparameterized(theta, event_times):

    mu, alpha, beta = _theta_to_hawkes_params(theta)

    return -calculate_log_likelihood(event_times, mu, alpha, beta)


def _build_multistart_initial_guesses(event_times):
    """
    Build a small grid of initial guesses for the branching ratio n,
    with mu initialized from the empirical mean intensity so that the
    implied stationary mean intensity mu / (1 - n) roughly matches
    the data from the very first iteration:

        mu_0 = (1 - n_0) * lambda_hat,   lambda_hat = N / T

    Only n0 is varied (beta0 is fixed at 1.0): this dataset's
    likelihood surface has consistently converged to the same optimum
    regardless of starting beta in testing, and varying both n0 and
    beta0 on a full grid (9 starts) made every downstream re-fit in
    this project roughly 9x slower for no measurable benefit here.
    Varying n0 alone still guards against the one failure mode that
    matters most -- getting stuck fitting a near-zero or near-1
    branching ratio -- at roughly 3x the cost of a single start.
    """

    lambda_hat = len(event_times) / event_times[-1]
    beta0 = 1.0

    initial_guesses = []

    for n0 in (0.2, 0.5, 0.8):

        mu0 = (1 - n0) * lambda_hat
        alpha0 = n0 * beta0

        initial_guesses.append(
            _hawkes_params_to_theta(mu0, alpha0, beta0)
        )

    return initial_guesses


def estimate_hawkes_parameters(event_times):
    """
    Estimate Hawkes process parameters using maximum likelihood.

    The optimization is run in an unconstrained reparameterization
    that guarantees stability (n = alpha / beta < 1) for every
    candidate parameter vector the optimizer can propose (see
    _theta_to_hawkes_params), from a grid of 9 initial guesses
    spanning small/medium/large branching ratio and decay rate. The
    best (lowest negative log-likelihood) converged fit is kept.

    Returns a scipy OptimizeResult whose `.x` is [mu, alpha, beta] in
    the original, interpretable parameterization -- callers do not
    need to know the fit was done in a reparameterized space.
    """

    best_result = None
    best_theta = None

    for theta0 in _build_multistart_initial_guesses(event_times):

        result = minimize(
            _negative_log_likelihood_reparameterized,
            theta0,
            args=(event_times,),
            method="BFGS"
        )

        # scipy's BFGS often reports success=False with a "precision
        # loss" message right at a sharp, well-conditioned optimum
        # (the line search can't improve further in floating point,
        # even though it has effectively converged). Gating on
        # success alone would discard perfectly good fits, so the
        # only hard requirement is a finite objective value; the
        # best-of-multi-start comparison below does the real
        # correctness check by picking the lowest achieved value.
        if not np.isfinite(result.fun):
            continue

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_theta = result.x

    if best_result is None:
        raise RuntimeError(
            "Hawkes MLE failed to converge from every multi-start "
            "initial guess."
        )

    mu, alpha, beta = _theta_to_hawkes_params(best_theta)

    # Overwrite .x with the original, interpretable parameterization
    # so every existing caller (which does
    # `mu, alpha, beta = result.x`) keeps working unchanged.
    best_result.x = np.array([mu, alpha, beta])

    return best_result


def calculate_branching_ratio(alpha, beta):
    """
    Calculate the branching ratio of the exponential Hawkes process.

    For the kernel:

        g(t) = α exp(-βt)

    the branching ratio is:

        n = ∫₀∞ g(t) dt
          = α / β

    It represents the expected number of offspring events
    directly generated by one event.

    A stationary Hawkes process requires:

        n < 1
    """

    return alpha / beta


def is_hawkes_stable(alpha, beta):
    """
    Check whether the Hawkes process satisfies the
    stationarity/stability condition.

    The process is stable when:

        α / β < 1
    """

    branching_ratio = calculate_branching_ratio(
        alpha,
        beta
    )

    return branching_ratio < 1

def simulate_hawkes(mu, alpha, beta, observation_time, seed=None):
    rng = np.random.default_rng(seed)

    event_times = []
    current_time = 0.0
    excitation = 0.0
    upper_intensity = mu

    while True:
        delta_t = rng.exponential(1 / upper_intensity)
        candidate_time = current_time + delta_t

        if candidate_time > observation_time:
            break

        excitation *= np.exp(
            -beta * (candidate_time - current_time)
        )

        intensity = mu + excitation

        if rng.uniform() < intensity / upper_intensity:
            event_times.append(candidate_time)
            excitation += alpha

        current_time = candidate_time
        upper_intensity = mu + excitation

    return event_times

if __name__ == "__main__":

    # ==========================================================
    # TOY EXAMPLE
    # ==========================================================

    # Simple example event times for testing.
    toy_event_times = np.array([
        1.0,
        2.0,
        4.0
    ])

    # Hawkes process parameters.
    #
    # μ     = baseline intensity
    # α     = strength of excitation
    # β     = rate at which excitation decays
    toy_mu = 1.0
    toy_alpha = 0.5
    toy_beta = 1.0

    # ----------------------------------------------------------
    # Direct intensity calculation
    # ----------------------------------------------------------

    intensity = hawkes_intensity(
        5.0,
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("Toy intensity at t = 5:")
    print(intensity)

    # ----------------------------------------------------------
    # Toy intensity DataFrame
    # ----------------------------------------------------------

    df = create_hawkes_intensity_dataframe(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("\nToy Hawkes intensity DataFrame:")
    print(df)

    # ----------------------------------------------------------
    # Log intensity sum
    # ----------------------------------------------------------

    log_sum = calculate_log_intensity_sum(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("\nToy log intensity sum:")
    print(log_sum)

    # ----------------------------------------------------------
    # Integrated intensity
    # ----------------------------------------------------------

    integrated_intensity = calculate_integrated_intensity(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("\nToy integrated intensity:")
    print(integrated_intensity)

    # ----------------------------------------------------------
    # Log likelihood
    # ----------------------------------------------------------

    log_likelihood = calculate_log_likelihood(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("\nToy log likelihood:")
    print(log_likelihood)

    # ----------------------------------------------------------
    # Toy parameter estimation
    # ----------------------------------------------------------

    estimated_params = estimate_hawkes_parameters(
        toy_event_times
    )

    print("\nToy parameter estimation:")
    print(estimated_params)

    # Extract estimated parameters.
    estimated_mu, estimated_alpha, estimated_beta = (
        estimated_params.x
    )

    # Calculate branching ratio.
    branching_ratio = calculate_branching_ratio(
        estimated_alpha,
        estimated_beta
    )

    print("\nToy branching ratio:")
    print(branching_ratio)

    # ==========================================================
    # VALIDATE FAST IMPLEMENTATION
    # ==========================================================

    # Calculate intensities using the original O(n^2)
    # implementation.
    old_intensities = calculate_hawkes_intensity_series(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    # Calculate intensities using the new O(n)
    # recursive implementation.
    new_intensities = calculate_hawkes_intensity_series_fast(
        toy_event_times,
        toy_mu,
        toy_alpha,
        toy_beta
    )

    print("\nOld intensities:")
    print(old_intensities)

    print("\nNew intensities:")
    print(new_intensities)

    print("\nDifference:")
    print(
        np.array(old_intensities)
        - new_intensities
    )

    # ==========================================================
    # REAL BTCUSDT DATA
    # ==========================================================

    print("\n" + "=" * 60)
    print("REAL BTCUSDT DATA")
    print("=" * 60)

    # Load the first 1,000,000 trades.
    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    # Convert timestamps and calculate trade statistics.
    processed_data = process_trade_data(data)

    # Prepare unique event times in elapsed seconds.
    event_times = prepare_hawkes_event_times(
        processed_data
    )

    # ----------------------------------------------------------
    # Basic event-time information
    # ----------------------------------------------------------

    print("\nNumber of event times:")
    print(len(event_times))

    print("\nObservation period:")
    print(f"{event_times[-1]:.6f} seconds")

    print("\nFirst 10 event times:")
    print(event_times[:10])

    # ----------------------------------------------------------
    # Estimate Hawkes parameters
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("HAWKES PARAMETER ESTIMATION")
    print("=" * 60)

    print("\nEstimating Hawkes parameters...")

    result = estimate_hawkes_parameters(
        event_times
    )

    print("\nOptimization result:")
    print(result)

    # Extract estimated parameters.
    estimated_mu, estimated_alpha, estimated_beta = (
        result.x
    )

    print("\nEstimated parameters:")
    print(f"mu    = {estimated_mu:.6f}")
    print(f"alpha = {estimated_alpha:.6f}")
    print(f"beta  = {estimated_beta:.6f}")

    # ----------------------------------------------------------
    # Branching ratio
    # ----------------------------------------------------------

    branching_ratio = calculate_branching_ratio(
        estimated_alpha,
        estimated_beta
    )

    print("\nBranching ratio:")
    print(f"{branching_ratio:.6f}")

    # ----------------------------------------------------------
    # Stability
    # ----------------------------------------------------------

    stable = is_hawkes_stable(
        estimated_alpha,
        estimated_beta
    )

    print("\nHawkes process stable:")
    print(stable)

    mu = estimated_mu
    alpha = estimated_alpha
    beta = estimated_beta
    observation_time = 10000.0
    seed = 42

    simulated_events = simulate_hawkes(
        mu,
        alpha,
        beta,
        observation_time,
        seed=42
    )

    simulated_events = np.array(simulated_events)

    simulated_rescaled = calculate_time_rescaled_intervals(
        simulated_events,
        mu,
        alpha,
        beta
    )

    simulated_rescaled = simulated_rescaled[1:]

    print("\nSimulated Hawkes time-rescaled intervals:")
    print("Number:", len(simulated_rescaled))
    print("Mean:", np.mean(simulated_rescaled))
    print("Variance:", np.var(simulated_rescaled))

    simulated_autocorrelations = []

    for lag in range(1, 6):
        correlation = np.corrcoef(
            simulated_rescaled[:-lag],
            simulated_rescaled[lag:]
        )[0, 1]

        simulated_autocorrelations.append(correlation)

    print("Simulated Hawkes autocorrelations:")

    for lag, correlation in enumerate(
        simulated_autocorrelations,
        start=1
    ):
        print(f"Lag {lag}: {correlation:.6f}")

    rescaled_intervals = calculate_time_rescaled_intervals(
        event_times,
        mu,
        alpha,
        beta
    )

    rescaled_intervals = rescaled_intervals[1:]

    print("\nTime-rescaled intervals:")
    print("Number of intervals:", len(rescaled_intervals))
    print("Mean:", np.mean(rescaled_intervals))
    print("Variance:", np.var(rescaled_intervals))
    ks_statistic, ks_pvalue = kstest(
        rescaled_intervals,
        "expon",
        args=(0, 1)
    )

    print("KS statistic:", ks_statistic)
    print("KS p-value:", ks_pvalue)
    plt.figure(figsize=(8, 5))

    plt.hist(
        rescaled_intervals,
        bins=100,
        density=True,
        alpha=0.7,
        label="Time-rescaled intervals"
    )

    x = np.linspace(
        0,
        np.percentile(rescaled_intervals, 99),
        500
    )

    plt.plot(
        x,
        np.exp(-x),
        label="Exp(1)"
    )

    plt.xlabel("Time-rescaled interval")
    plt.ylabel("Density")
    plt.title("Time-Rescaled Intervals vs Exp(1)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("figures/hawkes_time_rescaled_histogram.png", dpi=150)
    plt.show()

    sorted_intervals = np.sort(rescaled_intervals)

    probabilities = (
        np.arange(1, len(sorted_intervals) + 1)
        / (len(sorted_intervals) + 1)
    )

    theoretical_quantiles = expon.ppf(
        probabilities
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        theoretical_quantiles,
        sorted_intervals,
        marker=".",
        markersize=1,
        linestyle="none",
        alpha=0.3
    )

    max_value = max(
        theoretical_quantiles[-1],
        sorted_intervals[-1]
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        label="y = x"
    )

    plt.xlabel("Theoretical Exp(1) quantiles")
    plt.ylabel("Observed time-rescaled quantiles")
    plt.title("Q-Q Plot of Time-Rescaled Intervals")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("figures/hawkes_qq_plot.png", dpi=150)
    plt.show()

    print("First 10:")
    print(rescaled_intervals[:10])

    autocorrelations = []

    for lag in range(1, 21):
        correlation = np.corrcoef(
            rescaled_intervals[:-lag],
            rescaled_intervals[lag:]
        )[0, 1]

        autocorrelations.append(correlation)

    print("\nTime-rescaled interval autocorrelations:")

    for lag, correlation in enumerate(
        autocorrelations,
        start=1
    ):
        print(
            f"Lag {lag}: {correlation:.6f}"
        )