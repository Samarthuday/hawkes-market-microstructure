import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import prepare_hawkes_event_times, simulate_hawkes
from trade_processing import process_trade_data


def create_lag_bins(minimum_lag, maximum_lag, number_of_bins):
    """
    Create logarithmically spaced lag bins for Hawkes process analysis.

    Parameters:
        minimum_lag (float): The minimum lag value (in seconds).
        maximum_lag (float): The maximum lag value (in seconds).
        number_of_bins (int): The number of bins to create.
    """
    lag_bins = np.logspace(
        np.log10(minimum_lag),
        np.log10(maximum_lag),
        number_of_bins + 1
    )
    return lag_bins

def calculate_aftershock_counts_slow(event_times, lag_bins):
    counts = np.zeros(len(lag_bins) - 1, dtype=int)
    observation_end = event_times[-1]
    for event_time in event_times:
        for lag_index in range(len(lag_bins) - 1):
            lag_start = lag_bins[lag_index]
            lag_end = lag_bins[lag_index + 1]
            future_start = event_time + lag_start
            future_end = event_time + lag_end
            if future_end > observation_end:
                continue
            count = np.sum(
                (event_times >= future_start) &
                (event_times < future_end)
            )
            counts[lag_index] += count
    return counts

def calculate_aftershock_counts(event_times, lag_bins):
    counts = np.zeros(len(lag_bins) - 1, dtype=int)
    observation_end = event_times[-1]
    for event_time in event_times:
        valid_bins = event_time + lag_bins[1:] <= observation_end
        future_times = event_time + lag_bins
        indices = np.searchsorted(event_times, future_times)
        event_counts = np.diff(indices)
        counts[valid_bins] += event_counts[valid_bins]
    return counts

def calculate_aftershock_exposure(event_times, lag_bins):
    exposure = np.zeros(len(lag_bins) - 1, dtype=float)
    observation_end = event_times[-1]
    bin_widths = np.diff(lag_bins)
    for event_time in event_times:
        valid_bins = event_time + lag_bins[1:] <= observation_end
        exposure[valid_bins] += bin_widths[valid_bins]
    return exposure

def calculate_aftershock_rate(counts, exposure):
    return counts / exposure


if __name__ == "__main__":

    lag_bins = create_lag_bins(
        0.0001,
        1.0,
        50
    )

    print("Number of lag-bin edges:", len(lag_bins))
    print("Number of lag bins:", len(lag_bins) - 1)

    toy_event_times = np.array([
        0.0,
        0.001,
        0.003,
        0.010,
    ])

    toy_lag_bins = np.array([
        0.001,
        0.003,
        0.010,
    ])

    toy_counts = calculate_aftershock_counts(
        toy_event_times,
        toy_lag_bins
    )

    toy_exposure = calculate_aftershock_exposure(
        toy_event_times,
        toy_lag_bins
    )

    toy_rate = calculate_aftershock_rate(
        toy_counts,
        toy_exposure
    )

    print("\nToy aftershock counts:")
    print(toy_counts)

    print("\nToy aftershock exposure:")
    print(toy_exposure)

    print("\nToy aftershock rate:")
    print(toy_rate)

    slow_counts = calculate_aftershock_counts_slow(
        toy_event_times,
        toy_lag_bins
    )

    fast_counts = calculate_aftershock_counts(
        toy_event_times,
        toy_lag_bins
    )

    print("\nSlow counts:")
    print(slow_counts)

    print("Fast counts:")
    print(fast_counts)

    print("Counts identical:")
    print(np.array_equal(slow_counts, fast_counts))

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    data = process_trade_data(data)
    event_times = prepare_hawkes_event_times(data)
    print("Raw rows:", len(data))
    print("Event times:", len(event_times))
    print("Unique event times:", len(np.unique(event_times)))
    print("First 10 event times:", event_times[:10])
    print("Last 10 event times:", event_times[-10:])

    print("\nNumber of real event times:")
    print(len(event_times))

    real_counts = calculate_aftershock_counts(
        event_times,
        lag_bins
    )

    real_exposure = calculate_aftershock_exposure(
        event_times,
        lag_bins
    )

    real_rate = calculate_aftershock_rate(
        real_counts,
        real_exposure
    )

    lag_centers = np.sqrt(
    lag_bins[:-1] * lag_bins[1:]
    )

    mu = 2.301353
    alpha = 23.225818
    beta = 59.767895

    branching_ratio = alpha / beta
    mean_rate = mu / (1 - branching_ratio)
    response_rate = beta - alpha

    theoretical_rate = (
        mean_rate
        + (
            alpha
            / response_rate
            * (
                np.exp(-response_rate * lag_bins[:-1])
                - np.exp(-response_rate * lag_bins[1:])
            )
            / np.diff(lag_bins)
        )
    )


    rate_ratio = real_rate / theoretical_rate

    print("First 10 empirical / Hawkes ratios:")
    print(rate_ratio[:10])

    print("Last 5 empirical / Hawkes ratios:")
    print(rate_ratio[-5:])

    print("First 10 theoretical rates:")
    print(theoretical_rate[:10])
    print("Branching ratio:", branching_ratio)
    print("Mean rate:", mean_rate)
    print("Response timescale:", 1 / (beta - alpha))

    simulated_events = simulate_hawkes(
        mu,
        alpha,
        beta,
        10_000.0,
        seed=42
    )

    simulated_event_times = np.array(simulated_events)

    print("\nSimulated Hawkes process:")
    print("Number of simulated events:")
    print(len(simulated_event_times))

    print("First 10 simulated events:")
    print(simulated_event_times[:10])

    print("Last simulated event:")
    print(simulated_event_times[-1])

    simulated_counts = calculate_aftershock_counts(
        simulated_event_times,
        lag_bins
    )

    simulated_exposure = calculate_aftershock_exposure(
        simulated_event_times,
        lag_bins
    )

    simulated_rate = calculate_aftershock_rate(
        simulated_counts,
        simulated_exposure
    )

    print("\nFirst 10 simulated aftershock rates:")
    print(simulated_rate[:10])

    print("\nLast 5 simulated aftershock rates:")
    print(simulated_rate[-5:])

    simulated_ratio = simulated_rate / theoretical_rate

    simulated_log_error = (
        np.log(simulated_rate)
        - np.log(theoretical_rate)
    )

    print("\nFirst 10 simulated / Hawkes ratios:")
    print(simulated_ratio[:10])

    print("Last 5 simulated / Hawkes ratios:")
    print(simulated_ratio[-5:])

    print("\nMean absolute simulated log error:")
    print(np.mean(np.abs(simulated_log_error)))

    log_error = np.log(real_rate) - np.log(theoretical_rate)
    print("Mean absolute log error:")
    print(np.mean(np.abs(log_error)))

    empirical_excess = real_rate - mean_rate
    print("First 10 empirical excess rates:")
    print(empirical_excess[:10])

    rate_ratio = real_rate / theoretical_rate

    log_error = np.log(real_rate) - np.log(theoretical_rate)

    empirical_excess = real_rate - mean_rate

    plt.figure(figsize=(8, 5))

    plt.plot(
        lag_centers,
        real_rate,
        marker="o",
        markersize=4,
        label="Empirical"
    )

    plt.plot(
        lag_centers,
        theoretical_rate,
        label="Hawkes"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Lag (seconds)")
    plt.ylabel("Conditional event rate (events/second)")
    plt.title("Empirical vs Hawkes Aftershock Rate")

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.savefig("figures/aftershock_empirical_vs_hawkes.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        lag_centers,
        real_rate,
        marker="o",
        markersize=4
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Lag (seconds)")
    plt.ylabel("Aftershock rate (events/second)")
    plt.title("Empirical Aftershock Rate")

    plt.grid(True, which="both", alpha=0.3)

    plt.savefig("figures/aftershock_empirical.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        lag_centers,
        simulated_rate,
        marker="o",
        markersize=4,
        label="Simulated Hawkes"
    )

    plt.plot(
        lag_centers,
        theoretical_rate,
        label="Theoretical Hawkes"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Lag (seconds)")
    plt.ylabel("Conditional event rate (events/second)")
    plt.title("Simulated Hawkes vs Theoretical Aftershock Rate")

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.savefig("figures/aftershock_simulated_vs_theoretical.png", dpi=150)
    plt.show()

    print("\nNumber of real count values:")
    print(len(real_counts))

    print("Number of exposure values:")
    print(len(real_exposure))

    print("Number of rate values:")
    print(len(real_rate))

    print("\nAny zero exposure?")
    print(np.any(real_exposure == 0))

    print("Any non-finite rates?")
    print(np.any(~np.isfinite(real_rate)))

    print("\nFirst 10 real counts:")
    print(real_counts[:10])

    print("\nFirst 10 real exposure values:")
    print(real_exposure[:10])

    print("\nFirst 10 real aftershock rates:")
    print(real_rate[:10])

    print("\nLast 5 real aftershock rates:")
    print(real_rate[-5:])

    print("\nFirst 10 lag centers:")
    print(lag_centers[:10])

    print("\n First 10 rate ratios:")
    print(rate_ratio[:10])

    print("\n Last 5 rate ratios:")
    print(rate_ratio[-5:])

    print("\n First 10 log errors:")
    print(log_error[:10])

    print("\n First 10 empirical excess rates:")
    print(empirical_excess[:10])