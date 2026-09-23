import time

import numpy as np

from data_loader import load_trade_data
from hawkes import (
    calculate_hawkes_intensity_series,
    calculate_hawkes_intensity_series_fast,
    prepare_hawkes_event_times,
)
from trade_processing import process_trade_data

# Fitted parameters from the full-dataset MLE fit (see hawkes.py),
# used only to give both implementations a realistic, non-trivial
# amount of excitation to sum over.
MU, ALPHA, BETA = 2.301358, 23.226112, 59.768948


def benchmark_at_size(event_times, n):
    """
    Time both the direct O(n^2) intensity calculation and the
    recursive O(n) calculation on the first n events of event_times.
    """

    subset = event_times[:n]

    start = time.time()
    calculate_hawkes_intensity_series(subset, MU, ALPHA, BETA)
    naive_seconds = time.time() - start

    start = time.time()
    calculate_hawkes_intensity_series_fast(subset, MU, ALPHA, BETA)
    fast_seconds = time.time() - start

    return naive_seconds, fast_seconds


if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )
    processed_data = process_trade_data(data)
    event_times = prepare_hawkes_event_times(processed_data)

    print("=" * 70)
    print("O(n^2) DIRECT vs O(n) RECURSIVE HAWKES INTENSITY BENCHMARK")
    print("=" * 70)
    print(f"\n{'n events':>10} {'naive O(n^2) (s)':>18} {'recursive O(n) (s)':>20} {'speedup':>10}")

    # The naive O(n^2) implementation is only benchmarked up to a size
    # where it still finishes in a reasonable amount of time; beyond
    # that only the recursive implementation is timed, which is
    # exactly the point being demonstrated.
    naive_sizes = [1_000, 2_000, 5_000, 10_000, 20_000]
    fast_only_sizes = [50_000, 100_000, len(event_times)]

    for n in naive_sizes:
        naive_seconds, fast_seconds = benchmark_at_size(event_times, n)
        speedup = naive_seconds / fast_seconds
        print(f"{n:>10,} {naive_seconds:>18.4f} {fast_seconds:>20.4f} {speedup:>9.1f}x")

    for n in fast_only_sizes:
        subset = event_times[:n]
        start = time.time()
        calculate_hawkes_intensity_series_fast(subset, MU, ALPHA, BETA)
        fast_seconds = time.time() - start
        print(f"{n:>10,} {'infeasible':>18} {fast_seconds:>20.4f} {'--':>10}")

    print(
        "\nThe O(n^2) implementation's runtime grows quadratically -- "
        "doubling the naive_sizes above roughly quadruples its "
        "runtime -- while the O(n) recursive implementation scales "
        "linearly, which is what makes calibrating this model on the "
        f"full {len(event_times):,}-event dataset (let alone a full "
        "month of trades) practical at all."
    )
