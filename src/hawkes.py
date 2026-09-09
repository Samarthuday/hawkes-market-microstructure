import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def estimate_hawkes_parameters(event_times):
    """
    Estimate Hawkes process parameters using maximum likelihood.

    Returns the scipy optimization result.
    """

    # Initial parameter guess.
    initial_params = [
        1.0,   # mu
        0.5,   # alpha
        1.0    # beta
    ]

    # Parameter bounds:
    #
    # mu    > 0
    # alpha >= 0
    # beta  > 0
    bounds = [
        (1e-6, None),
        (0, None),
        (1e-6, None)
    ]

    # Minimize the negative log-likelihood.
    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(event_times,),
        bounds=bounds
    )

    return result


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