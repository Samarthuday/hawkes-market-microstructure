import matplotlib.pyplot as plt
import pandas as pd


def inter_arrival_stats(data):
    """
    Calculate descriptive statistics for trade inter-arrival times.

    Inter-arrival time is the time between two consecutive trades:

        Δt_i = t_i - t_{i-1}

    The function reports statistics for:
    1. All inter-arrival times, including zero values.
    2. Positive inter-arrival times only.

    Zero inter-arrival times can occur when multiple trades have
    the same timestamp.
    """

    # Remove the first NaN value created by the time-difference calculation.
    inter_arrivals = data["inter_arrival_seconds"].dropna()

    # Calculate descriptive statistics for all inter-arrival times.
    stats = {
        "count": len(inter_arrivals),
        "mean": inter_arrivals.mean(),
        "median": inter_arrivals.median(),
        "std": inter_arrivals.std(),
        "min": inter_arrivals.min(),
        "max": inter_arrivals.max(),

        # Quantiles describe the distribution of inter-arrival times.
        "q25": inter_arrivals.quantile(0.25),
        "q75": inter_arrivals.quantile(0.75),
        "q95": inter_arrivals.quantile(0.95),
        "q99": inter_arrivals.quantile(0.99),

        # Fraction of observations where consecutive trades
        # have exactly the same timestamp.
        "zero_fraction": (inter_arrivals == 0).mean(),
    }

    # Keep only strictly positive inter-arrival times.
    # These represent actual elapsed time between distinct timestamps.
    positive_inter_arrivals = inter_arrivals[
        inter_arrivals > 0
    ]

    stats["positive_count"] = len(positive_inter_arrivals)
    stats["positive_mean"] = positive_inter_arrivals.mean()
    stats["positive_median"] = positive_inter_arrivals.median()
    stats["positive_std"] = positive_inter_arrivals.std()

    # Coefficient of variation:
    #
    #     CV = σ / μ
    #
    # For an exponential distribution, the theoretical CV is 1.
    # Therefore, CV provides a simple diagnostic for how closely
    # the inter-arrival times resemble an exponential distribution.
    positive_mean = positive_inter_arrivals.mean()

    stats["positive_cv"] = (
        positive_inter_arrivals.std() / positive_mean
        if positive_mean != 0
        else float("nan")
    )

    return stats


def plot_inter_arrival_distribution(data):
    """
    Plot the distribution of all trade inter-arrival times.
    """

    # Remove the NaN value corresponding to the first observation.
    inter_arrivals = data["inter_arrival_seconds"].dropna()

    plt.figure(figsize=(10, 6))

    # density=True normalizes the histogram so that its area
    # approximately integrates to 1, allowing it to be interpreted
    # as an empirical probability density.
    plt.hist(
        inter_arrivals,
        bins=100,
        density=True
    )

    plt.xlabel("Inter-arrival time (seconds)")
    plt.ylabel("Density")
    plt.title("Distribution of Trade Inter-arrival Times")

    plt.tight_layout()
    plt.show()


def plot_positive_inter_arrivals(data):
    """
    Plot the distribution of strictly positive inter-arrival times.

    Zero inter-arrival times are excluded so that the plot focuses
    on the actual elapsed time between distinct timestamps.
    """

    inter_arrivals = data["inter_arrival_seconds"].dropna()

    # Remove observations where consecutive trades have
    # exactly the same timestamp.
    positive_inter_arrivals = inter_arrivals[
        inter_arrivals > 0
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        positive_inter_arrivals,
        bins=100,
        density=True
    )

    plt.xlabel("Positive inter-arrival time (seconds)")
    plt.ylabel("Density")
    plt.title("Distribution of Positive Trade Inter-arrival Times")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from data_loader import load_trade_data
    from trade_processing import process_trade_data

    # Load the first 1,000,000 trades.
    # Using nrows is useful during development because the full
    # dataset can be very large.
    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    # Convert timestamps and calculate inter-arrival times.
    processed_data = process_trade_data(data)

    # Calculate descriptive statistics.
    stats = inter_arrival_stats(
        processed_data
    )

    # Print the statistics.
    for name, value in stats.items():

        if isinstance(value, float):
            print(f"{name}: {value:.6f}")

        else:
            print(f"{name}: {value:,}")

    # Plot the distribution including zero inter-arrival times.
    plot_inter_arrival_distribution(
        processed_data
    )

    # Plot only positive inter-arrival times.
    plot_positive_inter_arrivals(
        processed_data
    )