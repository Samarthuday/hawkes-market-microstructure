import numpy as np
import pandas as pd


def prepare_event_times(data):
    """
    Extract unique event timestamps from the trade data.

    The timestamps are:
        1. cleaned of missing values,
        2. deduplicated,
        3. sorted chronologically,
        4. re-indexed from zero.

    Returns
    -------
    pandas.Series
        Sorted series of unique event timestamps.
    """

    timestamps = (
        data["timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    return timestamps


def calculate_inter_arrivals(timestamps):
    """
    Calculate inter-arrival times between consecutive events.

    For event times t_1, t_2, ..., t_n:

        Δt_i = t_i - t_{i-1}

    The result is expressed in seconds.
    """

    # .diff() calculates the difference between consecutive timestamps.
    # .dt.total_seconds() converts the resulting Timedelta values to seconds.
    inter_arrivals = timestamps.diff().dt.total_seconds()

    # The first observation has no previous event, so its difference is NaN.
    inter_arrivals = inter_arrivals.dropna()

    return inter_arrivals


def estimate_poisson_intensity(inter_arrivals):
    """
    Estimate the intensity of a homogeneous Poisson process.

    The maximum-likelihood estimator for the Poisson intensity is:

        λ_hat = N / T

    where:
        N = number of observed inter-arrival intervals
        T = total observation time

    Since:

        T = Σ Δt_i

    we can also write:

        λ_hat = 1 / mean(Δt)

    Units:
        events / second
    """

    n = len(inter_arrivals)

    # Total observed time:
    #
    # T = Σ Δt_i
    total_time = inter_arrivals.sum()

    # Poisson intensity:
    #
    # λ_hat = N / T
    intensity = n / total_time

    return intensity


def calculate_mean_inter_arrival(inter_arrivals):
    """
    Calculate the mean inter-arrival time.

        mean(Δt) = (1/N) Σ Δt_i

    For an exponential distribution:

        E[Δt] = 1 / λ
    """

    return inter_arrivals.mean()


def exponential_pdf(x, intensity):
    """
    Calculate the theoretical PDF of an exponential distribution.

    For a Poisson process, inter-arrival times follow:

        Δt ~ Exponential(λ)

    with probability density function:

        f(t) = λ exp(-λt),   t >= 0
    """

    return intensity * np.exp(-intensity * x)


def plot_exponential_pdf(inter_arrivals, intensity):
    """
    Compare the empirical distribution of inter-arrival times
    with the theoretical exponential PDF implied by the
    estimated Poisson intensity.
    """

    import matplotlib.pyplot as plt

    # We only plot up to the 99th percentile so that a small
    # number of extremely large observations do not dominate
    # the visualization.
    x = np.linspace(
        0,
        inter_arrivals.quantile(0.99),
        500
    )

    # Theoretical exponential density.
    y = exponential_pdf(x, intensity)

    plt.figure(figsize=(10, 6))

    # Theoretical exponential PDF.
    plt.plot(
        x,
        y,
        label="Exponential"
    )

    # Empirical density of observed inter-arrival times.
    plt.hist(
        inter_arrivals,
        bins=100,
        density=True,
        alpha=0.6,
        color="blue",
        label="Inter-arrival times"
    )

    plt.xlabel("Inter-arrival time (seconds)")
    plt.ylabel("Probability Density")
    plt.title("Empirical vs Theoretical Exponential Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_coefficient_of_variation(inter_arrivals):
    """
    Calculate the coefficient of variation (CV).

    The coefficient of variation is:

        CV = σ / μ

    where:
        σ = standard deviation
        μ = mean

    For an exponential distribution:

        CV = 1

    Therefore, a CV substantially different from 1 provides
    evidence against exponentially distributed inter-arrival times.
    """

    mean = inter_arrivals.mean()
    std = inter_arrivals.std()

    cv = std / mean

    return cv


def exponential_cdf(x, intensity):
    """
    Calculate the theoretical CDF of an exponential distribution.

    For an exponential random variable:

        F(t) = P(Δt <= t)

    and:

        F(t) = 1 - exp(-λt)
    """

    return 1 - np.exp(-intensity * x)


def empirical_cdf(inter_arrivals):
    """
    Calculate the empirical cumulative distribution function (ECDF).

    After sorting the observations:

        x_(1) <= x_(2) <= ... <= x_(n)

    the empirical CDF at x_(i) is:

        F_n(x_(i)) = i / n
    """

    # Sort the observed inter-arrival times.
    sorted_data = np.sort(inter_arrivals)

    # Construct the empirical cumulative probabilities:
    #
    # 1/n, 2/n, ..., n/n
    cdf = (
        np.arange(1, len(sorted_data) + 1)
        / len(sorted_data)
    )

    return sorted_data, cdf


def plot_cdf_comparison(inter_arrivals, intensity):
    """
    Compare the empirical CDF of the observed inter-arrival
    times with the theoretical exponential CDF.
    """

    x, empirical = empirical_cdf(inter_arrivals)

    # Evaluate the theoretical exponential CDF at the same x-values.
    theoretical_cdf = exponential_cdf(
        x,
        intensity
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        empirical,
        label="Empirical"
    )

    plt.plot(
        x,
        theoretical_cdf,
        label="Theoretical"
    )

    plt.xlabel("Inter-arrival time (seconds)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.title("Empirical vs Theoretical Exponential Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_fano_factor(event_times, window_size):
    """
    Calculate the Fano factor for a given time-window size.

    First, divide the observation period into windows of length Δ.
    Let N_Δ be the number of events occurring in each window.

    The Fano factor is:

        F = Var(N_Δ) / E[N_Δ]

    For an ideal Poisson process:

        F = 1

    Interpretation:
        F ≈ 1  -> Poisson-like dispersion
        F > 1  -> overdispersion / clustering
        F < 1  -> underdispersion / more regular arrivals
    """

    # Total observation time:
    #
    # T = t_max - t_min
    total_time = (
        event_times.max() - event_times.min()
    ).total_seconds()

    # Number of complete windows:
    #
    # K = floor(T / Δ)
    num_windows = int(
        total_time // window_size
    )

    # Convert each timestamp into elapsed seconds from
    # the first observed event.
    elapsed_seconds = (
        event_times - event_times.min()
    ).dt.total_seconds()

    # Determine which window each event belongs to.
    #
    # window_index = floor(elapsed_time / Δ)
    #
    # For Δ = 10:
    #
    # [0, 10)   -> 0
    # [10, 20)  -> 1
    # [20, 30)  -> 2
    window_indices = np.floor(
        elapsed_seconds / window_size
    ).astype(int)

    # Count how many events occur in each window.
    window_counts = np.bincount(
        window_indices,
        minlength=num_windows
    )

    # Calculate the mean and variance of the window event counts.
    mean = np.mean(window_counts)
    variance = np.var(window_counts)

    # Fano factor:
    #
    # F = Var(N_Δ) / E[N_Δ]
    fano_factor = (
        variance / mean
        if mean != 0
        else float("nan")
    )

    return fano_factor


def create_event_count_series(event_times, window_size):
    """
    Divide the observation period into fixed time windows and
    create a time series containing the number of events in each window.

    Each observation in the resulting series is:

        N_k = number of events in window k

    where each window has length Δ = window_size.
    """

    # Total elapsed time in seconds.
    elapsed_time = (
        event_times.max() - event_times.min()
    ).total_seconds()

    # Number of windows required to cover the entire observation period.
    #
    # K = ceil(T / Δ)
    num_windows = int(
        np.ceil(elapsed_time / window_size)
    )

    # Create the boundaries of the windows.
    #
    # Example:
    # window_size = 10
    #
    # edges = [0, 10, 20, 30, ...]
    window_edges = (
        np.arange(0, num_windows + 1)
        * window_size
    )

    # Convert each event timestamp into elapsed seconds
    # from the first event.
    elapsed_seconds = (
        event_times - event_times.min()
    ).dt.total_seconds()

    # Determine the window containing each event.
    window_indices = np.floor(
        elapsed_seconds / window_size
    ).astype(int)

    # Count events in each window.
    window_counts = np.bincount(
        window_indices,
        minlength=num_windows
    )

    # Create intervals:
    #
    # [0, 10), [10, 20), [20, 30), ...
    #
    # closed="left" means that the left boundary is included
    # while the right boundary is excluded.
    event_count_series = pd.Series(
        window_counts,
        index=pd.IntervalIndex.from_arrays(
            window_edges[:-1],
            window_edges[1:],
            closed="left"
        )
    )

    return event_count_series


def calculate_autocorrelation(event_count_series, lag):
    """
    Calculate the autocorrelation of the event-count series.

    Autocorrelation measures the dependence between event counts
    separated by a given lag.

    For lag k:

        ρ(k) = γ(k) / γ(0)

    where:

        γ(k) = autocovariance at lag k
        γ(0) = variance

    For independent Poisson increments, counts in non-overlapping
    windows should have approximately zero autocorrelation.
    """

    mean = event_count_series.mean()

    # Variance of the event-count series.
    variance = np.var(event_count_series)

    n = len(event_count_series)

    if n <= lag:
        raise ValueError(
            "Lag is too large for the length of the series."
        )

    # Autocovariance at lag k:
    #
    # γ(k) =
    # 1/(n-k) Σ [X_t - μ][X_(t+k) - μ]
    autocovariance = np.sum(
        (event_count_series[:-lag] - mean)
        * (event_count_series[lag:] - mean)
    ) / (n - lag)

    # Autocorrelation:
    #
    # ρ(k) = γ(k) / γ(0)
    autocorrelation = (
        autocovariance / variance
        if variance != 0
        else float("nan")
    )

    return autocorrelation


# A simpler implementation using pandas' built-in autocorrelation:
#
# def calculate_autocorrelation(event_count_series, lag):
#     return event_count_series.autocorr(lag=lag)


def calculate_autocorrelations(event_count_series, max_lag):
    """
    Calculate autocorrelation for multiple lag values.

    For example, max_lag=20 calculates:

        ρ(1), ρ(2), ..., ρ(20)
    """

    autocorrelations = {}

    for lag in range(1, max_lag + 1):

        autocorrelations[lag] = calculate_autocorrelation(
            event_count_series,
            lag
        )

    return autocorrelations


def calculate_window_intensity(event_count_series, window_size):
    """
    Calculate the observed event intensity in each time window.

    For a window containing N events and having length Δ:

        λ_window = N / Δ

    Units:
        events / second
    """

    return event_count_series / window_size


def plot_window_intensity(event_count_series, window_size):
    """
    Plot the estimated event intensity over time.

    Each point represents the observed event intensity
    within one time window.
    """

    import matplotlib.pyplot as plt

    window_intensity = calculate_window_intensity(
        event_count_series,
        window_size
    )

    plt.figure(figsize=(10, 6))

    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity (events/second)")

    plt.title(
        f"Event Intensity Over Time "
        f"(Window Size: {window_size} seconds)"
    )

    # IntervalIndex.mid gives the midpoint of each time window,
    # which is used as the x-coordinate.
    plt.plot(
        window_intensity.index.mid,
        window_intensity.values,
        marker="o",
        linestyle="-"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from data_loader import load_trade_data
    from trade_processing import process_trade_data

    # Load the first 1,000,000 trades.
    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    # Convert timestamps and calculate inter-arrival variables.
    processed_data = process_trade_data(data)

    # Extract unique chronological event times.
    timestamps = prepare_event_times(
        processed_data
    )

    # Calculate inter-arrival times.
    inter_arrivals = calculate_inter_arrivals(
        timestamps
    )

    # Estimate Poisson intensity:
    #
    # λ_hat = N / T
    intensity = estimate_poisson_intensity(
        inter_arrivals
    )

    # Calculate average inter-arrival time.
    mean_inter_arrival = calculate_mean_inter_arrival(
        inter_arrivals
    )

    print("\nMean inter-arrival time:")
    print(f"{mean_inter_arrival:.6f} seconds")

    print("\nInverse of Poisson intensity:")
    print(f"{1 / intensity:.6f} seconds")

    print("\nPoisson intensity:")
    print(f"{intensity:.6f} events/second")

    print(
        "Number of unique event times:",
        len(timestamps)
    )

    print(
        "Number of inter-arrival times:",
        len(inter_arrivals)
    )

    print("\nFirst 10 inter-arrival times:")
    print(inter_arrivals.head(10))

    # Compare empirical inter-arrival distribution
    # against the theoretical exponential distribution.
    plot_exponential_pdf(
        inter_arrivals,
        intensity
    )

    # Create event counts using 10-second windows.
    event_count_series = create_event_count_series(
        timestamps,
        window_size=10
    )

    # Calculate autocorrelation for lags 1 through 20.
    autocorrelations = calculate_autocorrelations(
        event_count_series,
        max_lag=20
    )

    print("\nAutocorrelations:")

    for lag, value in autocorrelations.items():
        print(
            f"Lag {lag:>2}: {value:.6f}"
        )

    print("\nEvent count series:")
    print(event_count_series)

    # Plot the estimated intensity over time.
    plot_window_intensity(
        event_count_series,
        window_size=10
    )

    # Calculate the Fano factor for several window sizes.
    print("\nFano Factors:")

    for window_size in [
        1,
        2,
        *range(5, 101, 5)
    ]:

        fano_factor = calculate_fano_factor(
            timestamps,
            window_size=window_size
        )

        print(
            f"Window size {window_size:>3}: "
            f"{fano_factor:.6f}"
        )

    # Compare empirical and theoretical CDFs.
    plot_cdf_comparison(
        inter_arrivals,
        intensity
    )