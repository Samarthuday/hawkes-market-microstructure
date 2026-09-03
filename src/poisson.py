from data_loader import load_trade_data
from trade_processing import extract_event_times, process_trade_data


def estimate_poisson_intensity(data):
    """
    Estimate the constant event intensity of a homogeneous Poisson process.

    For a Poisson process, the maximum-likelihood estimate of the intensity
    is:

        λ_hat = N / T

    where:
        N = number of observed inter-arrivals
        T = total observation time

    Since N events produce N - 1 inter-arrival intervals, we use:

        λ_hat = (N - 1) / (t_N - t_1)

    Units:
        events / second
    """

    # Total number of unique event timestamps.
    n_events = len(data)

    # Start and end of the observation period.
    start_time = data["timestamp"].iloc[0]
    end_time = data["timestamp"].iloc[-1]

    # Total observation time:
    #
    # T = t_N - t_1
    #
    # Convert the pandas Timedelta into seconds.
    observation_time = (
        end_time - start_time
    ).total_seconds()

    # There are N - 1 inter-arrival intervals between N events.
    #
    # λ_hat = (N - 1) / T
    intensity = (
        n_events - 1
    ) / observation_time

    return intensity, observation_time


if __name__ == "__main__":

    # Load a subset of the trade data.
    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    # Convert timestamps and calculate basic time-related fields.
    processed_data = process_trade_data(data)

    # Extract unique, sorted event timestamps.
    event_times = extract_event_times(
        processed_data
    )

    # Estimate the baseline Poisson intensity.
    intensity, observation_time = estimate_poisson_intensity(
        event_times
    )

    print(f"Number of events: {len(event_times):,}")
    print(f"Observation time: {observation_time:.6f} seconds")
    print(f"Poisson intensity: {intensity:.2f} events/second")