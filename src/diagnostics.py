import matplotlib.pyplot as plt
import pandas as pd


def inter_arrival_stats(data):
    inter_arrivals = data["inter_arrival_seconds"].dropna()
    stats = {
        "count": len(inter_arrivals),
        "mean": inter_arrivals.mean(),
        "median": inter_arrivals.median(),
        "std": inter_arrivals.std(),
        "min": inter_arrivals.min(),
        "max": inter_arrivals.max(),
        "q25": inter_arrivals.quantile(0.25),
        "q75": inter_arrivals.quantile(0.75),
        "q95": inter_arrivals.quantile(0.95),
        "q99": inter_arrivals.quantile(0.99),
        "zero_fraction": (inter_arrivals == 0).mean(),
    }
    positive_inter_arrivals = inter_arrivals[inter_arrivals > 0]
    stats["positive_count"] = len(positive_inter_arrivals)
    stats["positive_mean"] = positive_inter_arrivals.mean()
    stats["positive_median"] = positive_inter_arrivals.median()
    stats["positive_std"] = positive_inter_arrivals.std()
    stats["positive_cv"] = positive_inter_arrivals.std() / positive_inter_arrivals.mean() if positive_inter_arrivals.mean() != 0 else float('nan')
    return stats

def plot_inter_arrival_distribution(data):

    inter_arrivals = (
        data["inter_arrival_seconds"]
        .dropna()
    )

    plt.figure(figsize=(10, 6))

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
    inter_arrivals = data["inter_arrival_seconds"].dropna()
    positive_inter_arrivals = inter_arrivals[inter_arrivals > 0]
    plt.figure(figsize=(10, 6))
    plt.hist(
        positive_inter_arrivals,
        bins=100,
        density=True
    )
    plt.xlabel("Positive Inter-arrival time (seconds)")
    plt.ylabel("Density")
    plt.title("Distribution of Positive Trade Inter-arrival Times")
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

    stats = inter_arrival_stats(
        processed_data
    )

    for name, value in stats.items():

        if isinstance(value, float):

            print(f"{name}: {value:.6f}")

        else:

            print(f"{name}: {value:,}")

    plot_inter_arrival_distribution(
        processed_data
    )

    plot_positive_inter_arrivals(
        processed_data
    )