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