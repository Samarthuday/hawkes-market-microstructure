def estimate_poisson_intensity(data):
    n_events = len(data)
    start_time = data["timestamp"].iloc[0]
    end_time = data["timestamp"].iloc[-1]
    observation_time = (
        end_time - start_time
    ).total_seconds()  # Total observation time in seconds
    intensity = (n_events - 1) / observation_time  # Estimate intensity (events per second) (mu bar = (N-1)/T)
    return intensity, observation_time

if __name__ == "__main__":

    from data_loader import load_trade_data
    from trade_processing import process_trade_data

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    intensity, observation_time = estimate_poisson_intensity(
        processed_data
    )

    print(f"Number of events: {len(processed_data):,}")
    print(f"Observation time: {observation_time:.6f} seconds")
    print(f"Poisson intensity: {intensity:.2f} events/second")