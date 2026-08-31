from data_loader import load_trade_data
from trade_processing import extract_event_times, process_trade_data


def estimate_poisson_intensity(data):

    n_events = len(data)

    start_time = data["timestamp"].iloc[0]

    end_time = data["timestamp"].iloc[-1]

    observation_time = (
        end_time - start_time
    ).total_seconds()

    intensity = (
        n_events - 1
    ) / observation_time

    return intensity, observation_time


if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    event_times = extract_event_times(
        processed_data
    )

    intensity, observation_time = estimate_poisson_intensity(
        event_times
    )

    print(f"Number of events: {len(event_times):,}")
    print(f"Observation time: {observation_time:.6f} seconds")
    print(f"Poisson intensity: {intensity:.2f} events/second")