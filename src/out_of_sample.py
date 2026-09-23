import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    calculate_hawkes_intensity_series_fast,
    estimate_hawkes_parameters,
    is_hawkes_stable,
    prepare_hawkes_event_times,
)
from model_comparison import estimate_poisson_intensity
from nonstationarity import split_event_times_into_windows
from trade_processing import process_trade_data


def calculate_log_intensity_sum_in_window(
    event_times,
    mu,
    alpha,
    beta,
    window_start,
    window_end
):
    """
    Calculate Sum_i log(lambda(t_i)) restricted to events falling in
    [window_start, window_end), where lambda(t_i) at every event is
    computed from the *entire* event_times array (including events
    before window_start).

    This is the "excitation carries over the train/test boundary"
    piece of an out-of-sample evaluation: an event just after
    window_start is not treated as if the process had no history --
    it correctly inherits decayed excitation from events that
    occurred before the window (e.g. in the training period).
    """

    intensities = calculate_hawkes_intensity_series_fast(
        event_times, mu, alpha, beta
    )

    in_window = (event_times >= window_start) & (event_times < window_end)

    return np.sum(np.log(intensities[in_window]))


def calculate_integrated_intensity_in_window(
    event_times,
    mu,
    alpha,
    beta,
    window_start,
    window_end
):
    """
    Calculate integral_{window_start}^{window_end} lambda(t) dt,
    where lambda(t) includes excitation from every event before t
    (including events before window_start).

    For any event t_j < window_end (whether it falls before the
    window, i.e. "history", or inside it), its excitation
    contributes to the integral only over the portion of
    [window_start, window_end] that comes after t_j:

        integral_{max(window_start, t_j)}^{window_end}
            alpha * exp(-beta * (s - t_j)) ds
        = (alpha / beta) * [
              exp(-beta * max(0, window_start - t_j))
              - exp(-beta * (window_end - t_j))
          ]

    Summing this (closed-form, vectorized) contribution across every
    event with t_j < window_end, plus the baseline mu * window_length
    term, gives the integrated intensity for the window with correct
    history carry-over -- no truncation or approximation needed here
    since the exponential kernel's integral is analytic.
    """

    baseline_integral = mu * (window_end - window_start)

    relevant = event_times[event_times < window_end]

    excitation_integral = (alpha / beta) * np.sum(
        np.exp(-beta * np.maximum(0.0, window_start - relevant))
        - np.exp(-beta * (window_end - relevant))
    )

    return baseline_integral + excitation_integral


def calculate_held_out_log_likelihood_hawkes(
    event_times,
    mu,
    alpha,
    beta,
    window_start,
    window_end
):
    """
    Held-out exponential-kernel Hawkes log-likelihood for
    [window_start, window_end), using the full event series for
    excitation history but only scoring events inside the window.
    """

    return (
        calculate_log_intensity_sum_in_window(
            event_times, mu, alpha, beta, window_start, window_end
        )
        - calculate_integrated_intensity_in_window(
            event_times, mu, alpha, beta, window_start, window_end
        )
    )


def calculate_held_out_log_likelihood_poisson(
    event_times,
    poisson_rate,
    window_start,
    window_end
):
    """
    Held-out homogeneous Poisson log-likelihood for
    [window_start, window_end) at a fixed rate estimated from the
    training period.
    """

    n_test_events = np.sum(
        (event_times >= window_start) & (event_times < window_end)
    )

    return (
        n_test_events * np.log(poisson_rate)
        - poisson_rate * (window_end - window_start)
    )


def run_chronological_split(event_times, train_fraction=0.7):
    """
    Fit both models on the first train_fraction of the observation
    period (by elapsed time, not event count) and evaluate held-out
    log-likelihood on the remaining period.
    """

    T = event_times[-1]
    split_time = train_fraction * T

    train_events = event_times[event_times < split_time]

    poisson_rate = estimate_poisson_intensity(train_events)

    hawkes_fit = estimate_hawkes_parameters(train_events)
    mu, alpha, beta = hawkes_fit.x

    hawkes_test_ll = calculate_held_out_log_likelihood_hawkes(
        event_times, mu, alpha, beta, split_time, T
    )
    poisson_test_ll = calculate_held_out_log_likelihood_poisson(
        event_times, poisson_rate, split_time, T
    )

    n_test_events = np.sum(
        (event_times >= split_time) & (event_times < T)
    )

    return {
        "train_fraction": train_fraction,
        "split_time": split_time,
        "n_train_events": len(train_events),
        "n_test_events": int(n_test_events),
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": calculate_branching_ratio(alpha, beta),
        "stable": is_hawkes_stable(alpha, beta),
        "poisson_rate": poisson_rate,
        "hawkes_test_log_likelihood": hawkes_test_ll,
        "poisson_test_log_likelihood": poisson_test_ll,
        "hawkes_test_log_likelihood_per_event": (
            hawkes_test_ll / n_test_events
        ),
        "poisson_test_log_likelihood_per_event": (
            poisson_test_ll / n_test_events
        ),
    }


def run_rolling_block_folds(event_times, number_of_blocks=5):
    """
    Rolling out-of-sample check: fit on block i, evaluate held-out
    log-likelihood on block i+1, for every consecutive pair of blocks
    (reusing the same block partition as time_window_robustness.py).

    Unlike run_chronological_split (one large train/test split), this
    tests whether a model fit on one ~3.26-hour period of the data
    generalizes to predict the *next* period, repeated across the
    whole sample.
    """

    windows = split_event_times_into_windows(event_times, number_of_blocks)

    # windows are (block_index, block_start, block_end, shifted_events);
    # for this module we need the *unshifted* block boundaries against
    # the single shared event_times array, so re-derive them here.
    block_boundaries = [
        (window_start, window_end)
        for _, window_start, window_end, _ in windows
    ]

    rows = []

    for fold_index in range(number_of_blocks - 1):

        train_start, train_end = block_boundaries[fold_index]
        test_start, test_end = block_boundaries[fold_index + 1]

        train_events = event_times[
            (event_times >= train_start) & (event_times < train_end)
        ] - train_start

        poisson_rate = estimate_poisson_intensity(train_events)

        hawkes_fit = estimate_hawkes_parameters(train_events)
        mu, alpha, beta = hawkes_fit.x

        # Evaluate on the test block using only that block's own
        # history (a fold-local event_times array shifted so the
        # train block still starts at 0), so the "history carry-over"
        # is from the immediately preceding block, not the entire
        # rest of the dataset.
        fold_events = event_times[
            (event_times >= train_start) & (event_times < test_end)
        ] - train_start
        fold_test_start = test_start - train_start
        fold_test_end = test_end - train_start

        hawkes_test_ll = calculate_held_out_log_likelihood_hawkes(
            fold_events, mu, alpha, beta, fold_test_start, fold_test_end
        )
        poisson_test_ll = calculate_held_out_log_likelihood_poisson(
            fold_events, poisson_rate, fold_test_start, fold_test_end
        )

        n_test_events = np.sum(
            (event_times >= test_start) & (event_times < test_end)
        )

        rows.append({
            "fold": fold_index,
            "train_block": fold_index,
            "test_block": fold_index + 1,
            "n_train_events": len(train_events),
            "n_test_events": int(n_test_events),
            "branching_ratio": calculate_branching_ratio(alpha, beta),
            "stable": is_hawkes_stable(alpha, beta),
            "hawkes_test_ll_per_event": hawkes_test_ll / n_test_events,
            "poisson_test_ll_per_event": poisson_test_ll / n_test_events,
            "hawkes_wins": hawkes_test_ll > poisson_test_ll,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )
    processed_data = process_trade_data(data)
    event_times = prepare_hawkes_event_times(processed_data)

    print("=" * 60)
    print("OUT-OF-SAMPLE EVALUATION: 70/30 CHRONOLOGICAL SPLIT")
    print("=" * 60)

    split_result = run_chronological_split(event_times, train_fraction=0.7)

    for key, value in split_result.items():
        print(f"{key:>40}: {value}")

    print(
        f"\nHeld-out log-likelihood per test event -- "
        f"Hawkes: {split_result['hawkes_test_log_likelihood_per_event']:.6f}, "
        f"Poisson: {split_result['poisson_test_log_likelihood_per_event']:.6f}"
    )
    print(
        "Hawkes wins out-of-sample:",
        split_result["hawkes_test_log_likelihood"]
        > split_result["poisson_test_log_likelihood"]
    )

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE EVALUATION: ROLLING BLOCK FOLDS")
    print("=" * 60)

    fold_results = run_rolling_block_folds(event_times, number_of_blocks=5)

    print()
    print(fold_results.to_string(index=False))

    print(
        "\nHawkes wins out-of-sample in every fold:",
        fold_results["hawkes_wins"].all()
    )
