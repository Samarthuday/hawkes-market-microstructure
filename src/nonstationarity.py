import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_trade_data
from hawkes import (
    calculate_branching_ratio,
    estimate_hawkes_parameters,
    is_hawkes_stable,
    prepare_hawkes_event_times,
)
from trade_processing import process_trade_data


# Minimum number of events required inside a sub-window before we
# attempt an independent MLE fit. Windows with fewer events than this
# are skipped, since exponential-kernel Hawkes MLE is unreliable with
# very small samples.
MINIMUM_EVENTS_PER_WINDOW = 20


def split_event_times_into_windows(event_times, number_of_windows):
    """
    Split the observation period [0, T] into K equal-width,
    non-overlapping sub-windows.

    For K windows, the window edges are:

        e_0 = 0, e_1, e_2, ..., e_K = T

    with e_k = k * T / K.

    Events falling inside window k (edges e_k <= t < e_{k+1}, with the
    final window's right edge inclusive) are selected and then shifted
    so the window itself starts at time 0:

        t'_i = t_i - e_k

    This shift turns each sub-window into a standalone point process
    on [0, e_{k+1} - e_k], which can be fit independently with the
    same MLE machinery used on the full series.
    """

    T = event_times[-1]

    window_edges = np.linspace(0, T, number_of_windows + 1)

    windows = []

    for window_index in range(number_of_windows):

        window_start = window_edges[window_index]
        window_end = window_edges[window_index + 1]

        if window_index < number_of_windows - 1:
            in_window = (
                (event_times >= window_start)
                & (event_times < window_end)
            )
        else:
            # The final window includes its right edge so that the
            # very last event of the series is not dropped.
            in_window = (
                (event_times >= window_start)
                & (event_times <= window_end)
            )

        window_events = event_times[in_window]

        # Shift so the sub-window starts at time 0.
        shifted_events = window_events - window_start

        windows.append(
            (window_index, window_start, window_end, shifted_events)
        )

    return windows


def fit_rolling_hawkes_parameters(event_times, number_of_windows=10):
    """
    Fit an independent exponential-kernel Hawkes model in each of K
    equal-width sub-windows of the observation period.

    This is a coarse test of stationarity: if the true process were
    stationary, mu, alpha, beta (and therefore the branching ratio
    n = alpha / beta) estimated separately in each sub-window should
    fluctuate around a common value with no systematic trend over
    time, subject only to estimation noise from the smaller samples.

    Returns a pandas DataFrame with one row per window.
    """

    windows = split_event_times_into_windows(
        event_times,
        number_of_windows
    )

    results = []

    for window_index, window_start, window_end, shifted_events in windows:

        n_events = len(shifted_events)

        row = {
            "window_index": window_index,
            "window_start": window_start,
            "window_end": window_end,
            "window_midpoint": (window_start + window_end) / 2.0,
            "n_events": n_events,
            "mu": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "branching_ratio": np.nan,
            "stable": None,
            "skipped": False,
            "skip_reason": None,
        }

        if n_events < MINIMUM_EVENTS_PER_WINDOW:
            row["skipped"] = True
            row["skip_reason"] = (
                f"only {n_events} events "
                f"(< {MINIMUM_EVENTS_PER_WINDOW} required)"
            )
            results.append(row)
            continue

        try:
            fit_result = estimate_hawkes_parameters(shifted_events)

            mu, alpha, beta = fit_result.x

            row["mu"] = mu
            row["alpha"] = alpha
            row["beta"] = beta
            row["branching_ratio"] = calculate_branching_ratio(
                alpha,
                beta
            )
            row["stable"] = is_hawkes_stable(alpha, beta)

        except Exception as error:
            row["skipped"] = True
            row["skip_reason"] = f"MLE fit failed: {error}"

        results.append(row)

    return pd.DataFrame(results)


def compute_trend_correlations(results_df):
    """
    Correlate window midpoint time with each fitted parameter.

    For a stationary process, the correlation between elapsed time
    and mu, alpha, beta, or the branching ratio n = alpha / beta
    should be close to zero. A correlation with |r| that is large
    (as a rule of thumb, |r| > 0.5) is evidence of a systematic drift
    across the sample rather than pure sampling noise.
    """

    fitted = results_df.dropna(subset=["mu", "alpha", "beta"])

    correlations = {}

    for column in ["mu", "alpha", "beta", "branching_ratio"]:
        correlations[column] = np.corrcoef(
            fitted["window_midpoint"],
            fitted[column]
        )[0, 1]

    return correlations


def compute_coefficients_of_variation(results_df):
    """
    Calculate the coefficient of variation (CV = std / mean) of each
    fitted parameter across sub-windows.

    A small CV means the parameter is roughly constant across the
    sample. A large CV means the parameter varies substantially from
    window to window, which is consistent with either genuine
    nonstationarity or noisy small-sample MLE estimates (or both).
    """

    fitted = results_df.dropna(subset=["mu", "alpha", "beta"])

    cvs = {}

    for column in ["mu", "alpha", "beta", "branching_ratio"]:
        mean = fitted[column].mean()
        std = fitted[column].std()
        cvs[column] = std / mean if mean != 0 else float("nan")

    return cvs


def plot_rolling_parameters(results_df, output_path):
    """
    Plot mu, alpha, beta, and the branching ratio against the
    midpoint time of each sub-window, one subplot per parameter.
    """

    fitted = results_df.dropna(subset=["mu", "alpha", "beta"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("mu", "$\\mu$ (baseline intensity)", axes[0, 0]),
        ("alpha", "$\\alpha$ (excitation strength)", axes[0, 1]),
        ("beta", "$\\beta$ (decay rate)", axes[1, 0]),
        ("branching_ratio", "Branching ratio $n = \\alpha/\\beta$", axes[1, 1]),
    ]

    for column, label, ax in panels:

        ax.plot(
            fitted["window_midpoint"],
            fitted[column],
            marker="o",
            linestyle="-"
        )

        ax.set_xlabel("Window midpoint (seconds elapsed)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Rolling Hawkes Parameter Estimates "
        f"({len(fitted)} of {len(results_df)} windows fit)"
    )

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)


def print_results_table(results_df):

    display_df = results_df[[
        "window_index",
        "window_start",
        "window_end",
        "n_events",
        "mu",
        "alpha",
        "beta",
        "branching_ratio",
        "stable",
    ]].copy()

    for column in ["window_start", "window_end"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.2f}"
        )

    for column in ["mu", "alpha", "beta", "branching_ratio"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.6f}" if pd.notna(x) else "skipped"
        )

    print(display_df.to_string(index=False))


def print_interpretation(results_df, correlations, coefficients_of_variation):

    n_fit = results_df["mu"].notna().sum()
    n_total = len(results_df)

    print(
        f"\nFitted {n_fit} of {n_total} sub-windows "
        f"(remaining windows had too few events or failed to converge)."
    )

    print("\nCorrelation between window midpoint time and parameter:")
    for column, correlation in correlations.items():
        print(f"  {column:>16}: r = {correlation:+.4f}")

    print("\nCoefficient of variation (std / mean) across windows:")
    for column, cv in coefficients_of_variation.items():
        print(f"  {column:>16}: CV = {cv:.4f}")

    large_correlation = any(
        abs(r) > 0.5 for r in correlations.values()
    )

    large_dispersion = any(
        cv > 0.5 for cv in coefficients_of_variation.values()
    )

    print()

    if large_correlation:
        print(
            "INTERPRETATION: at least one parameter shows a strong "
            "correlation with elapsed time (|r| > 0.5), suggesting "
            "systematic drift rather than pure sampling noise -- the "
            "process is NOT obviously stationary over this sample."
        )
    elif large_dispersion:
        print(
            "INTERPRETATION: no parameter is strongly correlated with "
            "elapsed time, but window-to-window dispersion is "
            "substantial (CV > 0.5 for at least one parameter). This "
            "is more consistent with noisy small-sample estimation "
            "than with a systematic trend, but it does mean pooling "
            "all data into a single fit hides real window-to-window "
            "variability."
        )
    else:
        print(
            "INTERPRETATION: parameters are both weakly correlated "
            "with elapsed time and show limited window-to-window "
            "dispersion. The exponential Hawkes fit appears "
            "reasonably stable/stationary across this sample."
        )


def run_nonstationarity_analysis(event_times, number_of_windows, figure_path):

    print("\n" + "=" * 60)
    print(f"ROLLING HAWKES PARAMETER ESTIMATES (K = {number_of_windows})")
    print("=" * 60)

    results_df = fit_rolling_hawkes_parameters(
        event_times,
        number_of_windows=number_of_windows
    )

    print()
    print_results_table(results_df)

    correlations = compute_trend_correlations(results_df)
    coefficients_of_variation = compute_coefficients_of_variation(results_df)

    print_interpretation(
        results_df,
        correlations,
        coefficients_of_variation
    )

    plot_rolling_parameters(results_df, figure_path)

    print(f"\nFigure saved to: {figure_path}")

    return results_df


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
    print("NONSTATIONARITY CHECK: ROLLING HAWKES PARAMETERS")
    print("=" * 60)

    print("\nTotal number of events:", len(event_times))
    print(f"Observation period: {event_times[-1]:.2f} seconds "
          f"({event_times[-1] / 3600:.2f} hours)")

    # ==========================================================
    # K = 10 (PRIMARY CHECK)
    # ==========================================================

    results_k10 = run_nonstationarity_analysis(
        event_times,
        number_of_windows=10,
        figure_path="figures/nonstationarity_rolling_params.png"
    )

    # ==========================================================
    # K = 20 (ROBUSTNESS-OF-THE-ROBUSTNESS-CHECK)
    # ==========================================================

    results_k20 = run_nonstationarity_analysis(
        event_times,
        number_of_windows=20,
        figure_path="figures/nonstationarity_rolling_params_k20.png"
    )
