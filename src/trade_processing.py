import pandas as pd


def process_trade_data(data):
    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp_us"],
        unit="us",
        utc=True
    )

    data["inter_arrival_us"] = data["timestamp_us"].diff()

    data["inter_arrival_seconds"] = (
        data["inter_arrival_us"] / 1_000_000
    )

    return data


def extract_event_times(data):
    event_times = (
        data[["timestamp"]]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    return event_times


def event_times_to_seconds(event_times):
    event_difference = (
        event_times["timestamp"] - event_times["timestamp"].iloc[0]
    )

    return event_difference.dt.total_seconds()