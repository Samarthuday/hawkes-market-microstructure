import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    calculate_log_likelihood,
    estimate_hawkes_parameters,
    is_hawkes_stable,
    prepare_hawkes_event_times,
)
from model_comparison import (
    calculate_aic,
    calculate_bic,
    calculate_poisson_log_likelihood,
    estimate_poisson_intensity,
)
from nonstationarity import split_event_times_into_windows
from trade_processing import process_trade_data

# Number of contiguous, non-overlapping time blocks used for the
# robustness check. This is deliberately coarser than the K=10/K=20
# rolling windows used in nonstationarity.py: nonstationarity.py asks
# "is there a trend across many small windows?", while this module asks
# "does the headline finding (Hawkes >> Poisson, process stable)
# replicate across a few large, independent chunks of the data?"
NUMBER_OF_BLOCKS = 5

POISSON_PARAMETERS = 1
HAWKES_PARAMETERS = 3


def fit_block(block_index, window_start, window_end, shifted_events):
    """
    Fit both the Poisson baseline and the exponential-kernel Hawkes
    model to a single time block, and compare them with AIC/BIC.

    Each block is treated as an independent observation period
    starting at time 0 (the shift is already applied by
    split_event_times_into_windows), exactly as if it were its own
    standalone dataset.
    """

    n_events = len(shifted_events)

    poisson_intensity = estimate_poisson_intensity(shifted_events)

    poisson_log_likelihood = calculate_poisson_log_likelihood(
        shifted_events,
        poisson_intensity
    )

    hawkes_fit = estimate_hawkes_parameters(shifted_events)

    mu, alpha, beta = hawkes_fit.x

    hawkes_log_likelihood = calculate_log_likelihood(
        shifted_events,
        mu,
        alpha,
        beta
    )

    poisson_aic = calculate_aic(poisson_log_likelihood, POISSON_PARAMETERS)
    hawkes_aic = calculate_aic(hawkes_log_likelihood, HAWKES_PARAMETERS)

    poisson_bic = calculate_bic(
        poisson_log_likelihood,
        POISSON_PARAMETERS,
        n_events
    )
    hawkes_bic = calculate_bic(
        hawkes_log_likelihood,
        HAWKES_PARAMETERS,
        n_events
    )

    return {
        "block_index": block_index,
        "window_start": window_start,
        "window_end": window_end,
        "span_hours": (window_end - window_start) / 3600.0,
        "n_events": n_events,
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": calculate_branching_ratio(alpha, beta),
        "stable": is_hawkes_stable(alpha, beta),
        "poisson_intensity": poisson_intensity,
        "poisson_aic": poisson_aic,
        "hawkes_aic": hawkes_aic,
        "poisson_bic": poisson_bic,
        "hawkes_bic": hawkes_bic,
        "hawkes_wins_aic": hawkes_aic < poisson_aic,
        "hawkes_wins_bic": hawkes_bic < poisson_bic,
    }


def run_time_window_robustness(event_times, number_of_blocks=NUMBER_OF_BLOCKS):
    """
    Partition the observation period into `number_of_blocks` large,
    contiguous, non-overlapping blocks and independently fit both the
    Poisson baseline and the exponential-kernel Hawkes model in each.

    Reuses split_event_times_into_windows from nonstationarity.py,
    which already implements "split [0, T] into K equal-width windows
    and shift each window's events to start at 0" -- exactly what is
    needed here, just with far fewer, larger windows.
    """

    windows = split_event_times_into_windows(
        event_times,
        number_of_blocks
    )

    rows = [
        fit_block(block_index, window_start, window_end, shifted_events)
        for block_index, window_start, window_end, shifted_events in windows
    ]

    return pd.DataFrame(rows)


def print_results_table(results_df):

    display_df = results_df[[
        "block_index",
        "span_hours",
        "n_events",
        "mu",
        "alpha",
        "beta",
        "branching_ratio",
        "stable",
        "poisson_aic",
        "hawkes_aic",
        "hawkes_wins_aic",
    ]].copy()

    display_df["span_hours"] = display_df["span_hours"].map(
        lambda x: f"{x:.2f}"
    )

    for column in ["mu", "alpha", "beta", "branching_ratio"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.6f}"
        )

    for column in ["poisson_aic", "hawkes_aic"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.2f}"
        )

    print(display_df.to_string(index=False))


def print_interpretation(results_df):

    all_stable = results_df["stable"].all()
    all_hawkes_wins_aic = results_df["hawkes_wins_aic"].all()
    all_hawkes_wins_bic = results_df["hawkes_wins_bic"].all()

    branching_ratios = results_df["branching_ratio"].to_numpy()
    branching_ratio_relative_spread = (
        (branching_ratios.max() - branching_ratios.min())
        / branching_ratios.mean()
    )

    print()
    print(
        f"Branching ratio range across blocks: "
        f"[{branching_ratios.min():.4f}, {branching_ratios.max():.4f}] "
        f"(relative spread = {branching_ratio_relative_spread:.2%})"
    )
    print(f"All blocks stable (n < 1): {all_stable}")
    print(f"Hawkes beats Poisson by AIC in every block: {all_hawkes_wins_aic}")
    print(f"Hawkes beats Poisson by BIC in every block: {all_hawkes_wins_bic}")

    if all_stable and all_hawkes_wins_aic and all_hawkes_wins_bic:
        print(
            "\nINTERPRETATION: the headline finding replicates across "
            "every independent time block -- the Hawkes model beats "
            "the Poisson baseline by both AIC and BIC, and every "
            "block-level fit is stable, regardless of which chunk of "
            "the observation period is used."
        )
    else:
        print(
            "\nINTERPRETATION: the headline finding does NOT replicate "
            "uniformly across every block -- at least one block "
            "disagrees on model ranking or stability, so the pooled "
            "full-sample conclusion should be read with that caveat."
        )


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

    print("=" * 60)
    print("TIME-WINDOW ROBUSTNESS CHECK")
    print("=" * 60)

    print("\nTotal number of events:", len(event_times))
    print(
        f"Observation period: {event_times[-1]:.2f} seconds "
        f"({event_times[-1] / 3600:.2f} hours)"
    )
    print(
        f"\nSplitting into {NUMBER_OF_BLOCKS} blocks of "
        f"~{event_times[-1] / NUMBER_OF_BLOCKS / 3600:.2f} hours each."
    )

    results_df = run_time_window_robustness(
        event_times,
        number_of_blocks=NUMBER_OF_BLOCKS
    )

    print()
    print_results_table(results_df)

    print_interpretation(results_df)
