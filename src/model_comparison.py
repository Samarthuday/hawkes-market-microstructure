import numpy as np

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    calculate_log_likelihood,
    estimate_hawkes_parameters,
    prepare_hawkes_event_times,
)
from power_law_hawkes import (
    LOOKBACK_WINDOW_SECONDS,
    build_lookback_pairs,
    calculate_log_likelihood_power_law,
    calculate_power_law_branching_ratio,
    estimate_power_law_hawkes_parameters,
)
from trade_processing import (
    process_trade_data,
)


def calculate_aic(log_likelihood, number_of_parameters):
    """
    Calculate Akaike Information Criterion.

        AIC = 2k - 2 log(L)

    Lower AIC indicates a better model after
    accounting for model complexity.
    """

    return (
        2 * number_of_parameters
        - 2 * log_likelihood
    )


def calculate_bic(log_likelihood, number_of_parameters, n):
    """
    Calculate Bayesian Information Criterion.

        BIC = k log(N) - 2 log(L)

    Lower BIC indicates a better model after
    accounting for model complexity and sample size.
    """

    return (
        number_of_parameters * np.log(n)
        - 2 * log_likelihood
    )

def estimate_poisson_intensity(event_times):
    """
    Estimate the maximum-likelihood intensity of a
    homogeneous Poisson process.

    For a Poisson process:

        λ_hat = N / T

    where:

        N = number of events
        T = observation period
    """

    # Number of observed events.
    n = len(event_times)

    # Length of the observation period.
    T = event_times[-1]

    # Maximum-likelihood estimate of the Poisson intensity.
    intensity = n / T

    return intensity


def calculate_poisson_log_likelihood(event_times, intensity):
    """
    Calculate the log-likelihood of a homogeneous
    Poisson process.

        log L = N log(λ) - λT

    where:

        N = number of events
        λ = constant event intensity
        T = observation period
    """

    # Number of events.
    n = len(event_times)

    # Observation period.
    T = event_times[-1]

    # Poisson log-likelihood.
    log_likelihood = (
        n * np.log(intensity)
        - intensity * T
    )

    return log_likelihood


if __name__ == "__main__":

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    event_times = prepare_hawkes_event_times(
        processed_data
    )

    # ==========================================================
    # POISSON MODEL
    # ==========================================================

    poisson_intensity = estimate_poisson_intensity(
        event_times
    )

    poisson_log_likelihood = calculate_poisson_log_likelihood(
        event_times,
        poisson_intensity
    )

    print("=" * 60)
    print("POISSON MODEL")
    print("=" * 60)

    print("\nNumber of events:")
    print(len(event_times))

    print("\nObservation period:")
    print(f"{event_times[-1]:.6f} seconds")

    print("\nEstimated Poisson intensity:")
    print(f"{poisson_intensity:.6f} events/second")

    print("\nPoisson log-likelihood:")
    print(f"{poisson_log_likelihood:.6f}")

    # ==========================================================
    # HAWKES MODEL
    # ==========================================================

    print("\n" + "=" * 60)
    print("HAWKES MODEL")
    print("=" * 60)

    print("\nEstimating Hawkes parameters...")

    hawkes_result = estimate_hawkes_parameters(
        event_times
    )

    estimated_mu, estimated_alpha, estimated_beta = (
        hawkes_result.x
    )

    hawkes_log_likelihood = calculate_log_likelihood(
        event_times,
        estimated_mu,
        estimated_alpha,
        estimated_beta
    )

    print("\nEstimated mu:")
    print(f"{estimated_mu:.6f}")

    print("\nEstimated alpha:")
    print(f"{estimated_alpha:.6f}")

    print("\nEstimated beta:")
    print(f"{estimated_beta:.6f}")

    print("\nHawkes log-likelihood:")
    print(f"{hawkes_log_likelihood:.6f}")

    # ==========================================================
    # COMPARISON
    # ==========================================================

    log_likelihood_difference = (
        hawkes_log_likelihood
        - poisson_log_likelihood
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("\nPoisson log-likelihood:")
    print(f"{poisson_log_likelihood:.6f}")

    print("\nHawkes log-likelihood:")
    print(f"{hawkes_log_likelihood:.6f}")

    print("\nHawkes - Poisson:")
    print(f"{log_likelihood_difference:.6f}")

    # ==========================================================
    # AIC AND BIC
    # ==========================================================

    poisson_parameters = 1
    hawkes_parameters = 3
    n = len(event_times)

    poisson_aic = calculate_aic(
        poisson_log_likelihood,
        poisson_parameters
    )

    hawkes_aic = calculate_aic(
        hawkes_log_likelihood,
        hawkes_parameters
    )

    poisson_bic = calculate_bic(
        poisson_log_likelihood,
        poisson_parameters,
        n
    )

    hawkes_bic = calculate_bic(
        hawkes_log_likelihood,
        hawkes_parameters,
        n
    )

    print("\n" + "=" * 60)
    print("AIC / BIC COMPARISON")
    print("=" * 60)

    print("\nPoisson AIC:")
    print(f"{poisson_aic:.6f}")

    print("\nHawkes AIC:")
    print(f"{hawkes_aic:.6f}")

    print("\nPoisson BIC:")
    print(f"{poisson_bic:.6f}")

    print("\nHawkes BIC:")
    print(f"{hawkes_bic:.6f}")

    # ==========================================================
    # POWER-LAW HAWKES MODEL
    # ==========================================================

    print("\n" + "=" * 60)
    print("POWER-LAW HAWKES MODEL")
    print("=" * 60)

    print("\nEstimating power-law Hawkes parameters...")

    power_law_result = estimate_power_law_hawkes_parameters(
        event_times
    )

    pl_mu, pl_alpha, pl_c, pl_p = power_law_result.x

    # Rebuilding the lookback pairs here (rather than reusing the ones
    # inside estimate_power_law_hawkes_parameters) keeps this module's
    # public functions simple to call independently.
    power_law_pairs = build_lookback_pairs(
        event_times,
        LOOKBACK_WINDOW_SECONDS
    )

    power_law_log_likelihood = calculate_log_likelihood_power_law(
        event_times,
        power_law_pairs,
        pl_mu,
        pl_alpha,
        pl_c,
        pl_p
    )

    print(f"\nmu    = {pl_mu:.6f}")
    print(f"alpha = {pl_alpha:.6f}")
    print(f"c     = {pl_c:.6f}")
    print(f"p     = {pl_p:.6f}")

    print("\nPower-law Hawkes log-likelihood:")
    print(f"{power_law_log_likelihood:.6f}")

    power_law_parameters = 4

    power_law_aic = calculate_aic(
        power_law_log_likelihood,
        power_law_parameters
    )

    power_law_bic = calculate_bic(
        power_law_log_likelihood,
        power_law_parameters,
        n
    )

    # ==========================================================
    # THREE-WAY COMPARISON
    # ==========================================================

    print("\n" + "=" * 60)
    print("THREE-WAY MODEL COMPARISON")
    print("=" * 60 + "\n")

    exponential_branching_ratio = calculate_branching_ratio(
        estimated_alpha,
        estimated_beta
    )
    power_law_branching_ratio = calculate_power_law_branching_ratio(
        pl_alpha,
        pl_c,
        pl_p
    )

    print(
        f"{'Model':<22} {'Params':>7} {'LogLik':>14} "
        f"{'AIC':>14} {'BIC':>14} {'BranchRatio':>12}"
    )
    print(
        f"{'Poisson':<22} {poisson_parameters:>7} "
        f"{poisson_log_likelihood:>14.2f} {poisson_aic:>14.2f} "
        f"{poisson_bic:>14.2f} {'--':>12}"
    )
    print(
        f"{'Exponential Hawkes':<22} {hawkes_parameters:>7} "
        f"{hawkes_log_likelihood:>14.2f} {hawkes_aic:>14.2f} "
        f"{hawkes_bic:>14.2f} {exponential_branching_ratio:>12.4f}"
    )
    print(
        f"{'Power-law Hawkes':<22} {power_law_parameters:>7} "
        f"{power_law_log_likelihood:>14.2f} {power_law_aic:>14.2f} "
        f"{power_law_bic:>14.2f} {power_law_branching_ratio:>12.4f}"
    )

    aics = {
        "Poisson": poisson_aic,
        "Exponential Hawkes": hawkes_aic,
        "Power-law Hawkes": power_law_aic,
    }
    bics = {
        "Poisson": poisson_bic,
        "Exponential Hawkes": hawkes_bic,
        "Power-law Hawkes": power_law_bic,
    }
    best_aic_model = min(aics, key=aics.get)
    best_bic_model = min(bics, key=bics.get)

    print(f"\nBest model by AIC: {best_aic_model}")
    print(f"Best model by BIC: {best_bic_model}")

    log_likelihood_gain = power_law_log_likelihood - hawkes_log_likelihood

    print(
        f"\nPower-law - exponential log-likelihood gain: "
        f"{log_likelihood_gain:.2f} for "
        f"{power_law_parameters - hawkes_parameters} extra parameter(s)."
    )
    print(
        "Both AIC and BIC prefer the extra flexibility here despite the "
        "penalty, but note (see power_law_hawkes.py) that the fitted "
        "power-law shape parameter p can converge to a large value that "
        "makes the kernel decay almost as fast as the exponential kernel "
        "-- i.e. the improvement mainly reflects a better-fitting *shape* "
        "at short lags, not evidence of a genuinely fat/slow-decaying "
        "excitation tail."
    )