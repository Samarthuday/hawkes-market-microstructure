import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    calculate_log_likelihood,
    estimate_hawkes_parameters,
    is_hawkes_stable,
)
from trade_processing import event_times_to_seconds, process_trade_data

# Two trades within this many seconds of one another (after jittering)
# are considered coincident for reporting purposes. It is also the
# jitter step used to break ties in the "all trades" event definition.
JITTER_SECONDS = 1e-9

# Bin width used for the "burst-aggregated" event definition.
BURST_BIN_SECONDS = 0.001


def extract_all_trade_elapsed_seconds(processed_data):
    """
    Convert every trade's timestamp (not just unique ones) into
    elapsed seconds from the first event.

    A stable sort is used so that trades sharing a timestamp keep
    their original relative order (the order in which Binance
    recorded them), which matters when we later jitter ties apart.
    """

    all_timestamps = (
        processed_data[["timestamp"]]
        .dropna()
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    elapsed_seconds = (
        all_timestamps["timestamp"] - all_timestamps["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy()

    return elapsed_seconds


def jitter_tied_timestamps(elapsed_seconds, jitter=JITTER_SECONDS):
    """
    Break ties among identical event times by adding tiny,
    strictly-increasing offsets within each group of tied trades.

    The exponential-kernel Hawkes likelihood in hawkes.py relies on
    strictly increasing event times (its recursive excitation
    calculation uses Δt_i = t_i - t_{i-1}, and the intensity function
    uses a strict "<" comparison). If multiple trades share a
    timestamp, we cannot treat each one as a distinct event without
    first separating them in time.

    Within a tied group of size m, occurring at position i, i+1, ...,
    i+m-1 (in original trade order), the j-th trade in the group
    (j = 0, ..., m-1) is nudged forward by:

        t'_{i+j} = t_{i+j} + j * jitter

    This preserves the original relative order of tied trades while
    making every timestamp strictly increasing.
    """

    elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)

    # Identify the start of each new group of identical timestamps.
    # Since the array is already sorted ascending, a "new group"
    # begins wherever the value differs from the previous one.
    is_new_group = np.empty(len(elapsed_seconds), dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = np.diff(elapsed_seconds) != 0

    group_id = np.cumsum(is_new_group)

    # Position of each trade within its own tied group:
    # 0 for the first trade in the group, 1 for the second, etc.
    within_group_index = (
        pd.Series(group_id)
        .groupby(group_id)
        .cumcount()
        .to_numpy()
    )

    jittered = elapsed_seconds + within_group_index * jitter

    return jittered


def compute_fraction_of_trades_sharing_timestamp(elapsed_seconds):
    """
    Compute the fraction of trades whose timestamp is shared with at
    least one other trade.

    This quantifies how far event definition (b) ("all trades as
    distinct events") departs from the baseline definition (a)
    ("unique deduplicated timestamps"): the larger this fraction, the
    more the two definitions disagree about what counts as an event.
    """

    is_new_group = np.empty(len(elapsed_seconds), dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = np.diff(elapsed_seconds) != 0

    group_id = np.cumsum(is_new_group)

    group_sizes = pd.Series(group_id).groupby(group_id).transform("size")

    fraction_shared = (group_sizes.to_numpy() > 1).mean()

    return fraction_shared


def build_burst_aggregated_event_times(
    elapsed_seconds,
    bin_seconds=BURST_BIN_SECONDS
):
    """
    Aggregate trades occurring within the same bin_seconds-wide time
    bin into a single event, using the first (earliest) timestamp
    observed in the bin as the event time.

    This is the coarsening counterpart to jittering: instead of
    treating every trade as its own event, many nearly-simultaneous
    trades collapse into one event.
    """

    elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)

    bin_index = np.floor(elapsed_seconds / bin_seconds).astype(np.int64)

    # elapsed_seconds is already sorted ascending, so the first value
    # observed within each bin is also the minimum for that bin.
    burst_event_times = (
        pd.Series(elapsed_seconds)
        .groupby(bin_index)
        .first()
        .to_numpy()
    )

    return burst_event_times


def fit_hawkes_and_summarize(event_times, label):
    """
    Fit the exponential-kernel Hawkes model to a given event-time
    array and summarize the fit: parameter estimates, branching
    ratio, stability, log-likelihood, and event count.
    """

    fit_result = estimate_hawkes_parameters(event_times)

    mu, alpha, beta = fit_result.x

    log_likelihood = calculate_log_likelihood(
        event_times,
        mu,
        alpha,
        beta
    )

    return {
        "definition": label,
        "n_events": len(event_times),
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": calculate_branching_ratio(alpha, beta),
        "stable": is_hawkes_stable(alpha, beta),
        "log_likelihood": log_likelihood,
    }


def print_comparison_table(rows):

    df = pd.DataFrame(rows)

    display_df = df.copy()

    for column in ["mu", "alpha", "beta", "branching_ratio"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.6f}"
        )

    display_df["log_likelihood"] = display_df["log_likelihood"].map(
        lambda x: f"{x:.2f}"
    )

    print(display_df.to_string(index=False))

    return df


def print_interpretation(df):

    alpha_values = df["alpha"].to_numpy()
    branching_ratios = df["branching_ratio"].to_numpy()
    all_stable = df["stable"].all()
    all_alpha_positive = np.all(alpha_values > 0.01)

    branching_ratio_range = (
        branching_ratios.max() - branching_ratios.min()
    )
    branching_ratio_relative_spread = (
        branching_ratio_range / branching_ratios.mean()
    )

    print()

    print(
        f"Branching ratio range across definitions: "
        f"[{branching_ratios.min():.4f}, {branching_ratios.max():.4f}] "
        f"(relative spread = {branching_ratio_relative_spread:.2%})"
    )

    if all_alpha_positive and all_stable and branching_ratio_relative_spread < 0.5:
        print(
            "\nINTERPRETATION: the core qualitative conclusions hold "
            "across all three event definitions -- alpha is clearly "
            "positive (strong self-excitation), every fit is stable "
            "(n < 1), and the branching ratio stays in a similar "
            "ballpark. The specific choice of event definition changes "
            "the fitted numbers somewhat but does not change the story."
        )
    elif all_alpha_positive and all_stable:
        print(
            "\nINTERPRETATION: self-excitation (alpha > 0) and "
            "stability (n < 1) hold under all three event definitions, "
            "but the branching ratio magnitude shifts by a non-trivial "
            "amount depending on how an 'event' is defined -- the "
            "qualitative story survives, but the quantitative headline "
            "number is sensitive to this choice."
        )
    else:
        print(
            "\nINTERPRETATION: the qualitative conclusions do NOT hold "
            "uniformly across event definitions -- at least one "
            "definition produces alpha ~ 0 or an unstable/near-unstable "
            "fit, so the event-definition choice materially changes "
            "the story."
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

    print("=" * 60)
    print("EVENT DEFINITION ROBUSTNESS CHECK")
    print("=" * 60)

    # ==========================================================
    # (a) BASELINE: UNIQUE DEDUPLICATED TIMESTAMPS
    # ==========================================================

    from trade_processing import extract_event_times

    baseline_timestamps = extract_event_times(processed_data)
    baseline_event_times = event_times_to_seconds(
        baseline_timestamps
    ).to_numpy()

    # ==========================================================
    # (b) ALL TRADES AS DISTINCT EVENTS (JITTERED)
    # ==========================================================

    all_trade_elapsed_seconds = extract_all_trade_elapsed_seconds(
        processed_data
    )

    fraction_shared = compute_fraction_of_trades_sharing_timestamp(
        all_trade_elapsed_seconds
    )

    print(
        f"\nFraction of trades sharing a timestamp with "
        f"another trade: {fraction_shared:.4%}"
    )
    print(
        f"(Baseline definition collapses "
        f"{len(all_trade_elapsed_seconds)} trades down to "
        f"{len(baseline_event_times)} unique-timestamp events.)"
    )

    all_trades_event_times = jitter_tied_timestamps(
        all_trade_elapsed_seconds
    )

    # ==========================================================
    # (c) BURST-AGGREGATED EVENTS (1 MS BINS)
    # ==========================================================

    burst_event_times = build_burst_aggregated_event_times(
        all_trade_elapsed_seconds,
        bin_seconds=BURST_BIN_SECONDS
    )

    print(
        f"Burst aggregation (bin = {BURST_BIN_SECONDS * 1000:.0f} ms) "
        f"collapses {len(all_trade_elapsed_seconds)} trades down to "
        f"{len(burst_event_times)} events."
    )

    # ==========================================================
    # FIT HAWKES UNDER EACH DEFINITION
    # ==========================================================

    print("\nFitting Hawkes MLE under each event definition "
          "(this takes a little while for the 1,000,000-trade "
          "'all trades' definition)...")

    rows = [
        fit_hawkes_and_summarize(
            baseline_event_times,
            "(a) unique timestamps [baseline]"
        ),
        fit_hawkes_and_summarize(
            all_trades_event_times,
            "(b) all trades, jittered"
        ),
        fit_hawkes_and_summarize(
            burst_event_times,
            "(c) 1ms burst-aggregated"
        ),
    ]

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60 + "\n")

    comparison_df = print_comparison_table(rows)

    print_interpretation(comparison_df)
