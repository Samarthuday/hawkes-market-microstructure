import numpy as np
import pandas as pd


def prepare_event_times(data):

    timestamps = (
        data["timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    return timestamps

def calculate_inter_arrivals(timestamps):

    inter_arrivals = timestamps.diff().dt.total_seconds()

    inter_arrivals = inter_arrivals.dropna()

    return inter_arrivals

def estimate_poisson_intensity(inter_arrivals):

    n = len(inter_arrivals)

    total_time = inter_arrivals.sum()

    intensity = n / total_time

    return intensity

def calculate_mean_inter_arrival(inter_arrivals):

    return inter_arrivals.mean()

def exponential_pdf(x, intensity):
    return intensity * np.exp(-intensity * x)

def plot_exponential_pdf(inter_arrivals, intensity):
    import matplotlib.pyplot as plt

    x = np.linspace(0, inter_arrivals.quantile(0.99), 500)
    y = exponential_pdf(x, intensity)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Exponential")
    plt.hist(inter_arrivals, bins=100, density=True, alpha=0.6, color="blue", label="Inter-arrival times")
    plt.xlabel("Inter-arrival time (seconds)")
    plt.ylabel("Probability Density Function (PDF)")
    plt.title("Empirical vs Theoretical Exponential Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_coefficient_of_variation(inter_arrivals):

    mean = inter_arrivals.mean()
    std = inter_arrivals.std()

    cv = std / mean

    return cv

def exponential_cdf(x, intensity):
    return 1 - np.exp(-intensity * x)

def empirical_cdf(inter_arrivals):
    sorted_data = np.sort(inter_arrivals)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def plot_cdf_comparison(inter_arrivals, intensity):
    x, empirical = empirical_cdf(inter_arrivals)

    theoretical_cdf = exponential_cdf(x, intensity)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(x, empirical, label="Empirical")
    plt.plot(x, theoretical_cdf, label="Theoretical")
    plt.xlabel("Inter-arrival time (seconds)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.title("Empirical vs Theoretical Exponential Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_fano_factor(event_times, window_size):
    total_time = (event_times.max() - event_times.min()).total_seconds()
    num_windows = int(total_time // window_size) # num windows=⌊T/Δ​⌋
    window_indices = np.floor((event_times - event_times.min()).dt.total_seconds() / window_size).astype(int) # .dt.total_seconds() converts the time differences into ordinary numbers.
    window_counts = np.bincount(window_indices, minlength=num_windows) # np.bincount() essentially asks: How many times does each number appear?
    mean = np.mean(window_counts)
    variance = np.var(window_counts)
    fano_factor = variance / mean if mean != 0 else float('nan')
    return fano_factor

def create_event_count_series(event_times, window_size):
    elapsed_time = (event_times.max() - event_times.min()).total_seconds()
    num_windows = int(np.ceil(elapsed_time / window_size))
    window_edges = np.arange(0, num_windows + 1) * window_size
    window_indices = np.floor((event_times - event_times.min()).dt.total_seconds() / window_size).astype(int)
    window_counts = np.bincount(window_indices, minlength=num_windows)
    event_count_series = pd.Series(window_counts, index=pd.IntervalIndex.from_arrays(window_edges[:-1], window_edges[1:], closed='left'))
    return event_count_series

def calculate_autocorrelation(event_count_series, lag):
    mean = event_count_series.mean()
    variance = np.var(event_count_series)
    n = len(event_count_series)
    if n <= lag:
        raise ValueError("Lag is too large for the length of the series.")
    autocovariance = np.sum((event_count_series[:-lag] - mean) * (event_count_series[lag:] - mean)) / (n - lag)
    autocorrelation = autocovariance / variance if variance != 0 else float('nan')
    return autocorrelation

# A simpler way to calculate autocorrelation using pandas built-in function:
# def calculate_autocorrelation(event_count_series, lag):

#     return event_count_series.autocorr(lag=lag)

def calculate_autocorrelations(event_count_series, max_lag):
    autocorrelations = {}
    for lag in range(1, max_lag + 1):
        autocorrelations[lag] = calculate_autocorrelation(event_count_series, lag)
    return autocorrelations

def calculate_window_intensity(event_count_series, window_size):
    return event_count_series / window_size

def plot_window_intensity(event_count_series, window_size):
    import matplotlib.pyplot as plt

    window_intensity = calculate_window_intensity(event_count_series, window_size)
    plt.figure(figsize=(10, 6))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity (events/second)")
    plt.title(f"Event Intensity Over Time (Window Size: {window_size} seconds)")
    plt.plot(window_intensity.index.mid, window_intensity.values, marker='o', linestyle='-')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    from data_loader import load_trade_data
    from trade_processing import process_trade_data

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    timestamps = prepare_event_times(
        processed_data
    )

    inter_arrivals = calculate_inter_arrivals(
        timestamps
    )
    intensity = estimate_poisson_intensity(
    inter_arrivals
    )
    mean_inter_arrival = calculate_mean_inter_arrival(
    inter_arrivals
    )

    print("\nMean inter-arrival time:")
    print(f"{mean_inter_arrival:.6f} seconds")

    print("\nInverse of Poisson intensity:")
    print(f"{1 / intensity:.6f} seconds")

    print("\nPoisson intensity:")
    print(f"{intensity:.6f} events/second")

    print("Number of unique event times:", len(timestamps))
    print("Number of inter-arrival times:", len(inter_arrivals))

    print("\nFirst 10 inter-arrival times:")
    print(inter_arrivals.head(10))
    plot_exponential_pdf(inter_arrivals, intensity)
    event_count_series = create_event_count_series(timestamps, window_size=10)
    autocorrelations = calculate_autocorrelations(
        event_count_series,
        max_lag=20
    )
    print("\nAutocorrelations:")
    for lag, value in autocorrelations.items():
        print(f"Lag {lag:>2}: {value:.6f}")
    print("\nEvent count series:")
    print(event_count_series)
    plot_window_intensity(event_count_series, window_size=10)
    print("\nFano Factors:")
    for window_size in [1, 2, *range(5, 101, 5)]:
        fano_factor = calculate_fano_factor(timestamps, window_size=window_size)
        print(f"Window size {window_size:>3}: {fano_factor:.6f}")
    plot_cdf_comparison(inter_arrivals, intensity)