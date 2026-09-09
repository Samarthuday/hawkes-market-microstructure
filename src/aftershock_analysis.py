import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import prepare_hawkes_event_times
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

    print(lag_bins)
    print("Number of edges:", len(lag_bins))
    print("Number of bins:", len(lag_bins) - 1)

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

    print("Toy aftershock counts:")
    print(toy_counts)

    toy_exposure = calculate_aftershock_exposure(
    toy_event_times,
    toy_lag_bins
)

    print("Toy aftershock exposure:")
    print(toy_exposure)

    counts = np.array([2, 1])
    exposure = np.array([0.006, 0.007])

    toy_rate = calculate_aftershock_rate(
        toy_counts,
        toy_exposure
    )

    print("Toy aftershock rate:")
    print(toy_rate)