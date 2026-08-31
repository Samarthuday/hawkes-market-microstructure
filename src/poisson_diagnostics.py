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
    plot_cdf_comparison(inter_arrivals, intensity)