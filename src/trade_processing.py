import pandas as pd


def process_trade_data(data):
    """
    Process raw trade data by converting timestamps and
    calculating inter-arrival times between consecutive trades.

    Inter-arrival time is defined as:

        Δt_i = t_i - t_{i-1}

    where t_i is the timestamp of event i.
    """

    # Create a copy so that the original DataFrame is not modified.
    data = data.copy()

    # Convert Unix timestamps from microseconds to UTC datetime.
    #
    # timestamp_us:
    #     Unix timestamp measured in microseconds.
    #
    # unit="us":
    #     Tells pandas that the input is in microseconds.
    #
    # utc=True:
    #     Stores the resulting timestamps in UTC.
    data["timestamp"] = pd.to_datetime(
        data["timestamp_us"],
        unit="us",
        utc=True
    )

    # Calculate the time between consecutive trades in microseconds.
    #
    # Δt_i = timestamp_us[i] - timestamp_us[i-1]
    #
    # The first observation has no previous event, so its value is NaN.
    data["inter_arrival_us"] = data["timestamp_us"].diff()

    # Convert inter-arrival times from microseconds to seconds.
    #
    # 1 second = 1,000,000 microseconds
    #
    # Therefore:
    #
    # Δt_seconds = Δt_microseconds / 10^6
    data["inter_arrival_seconds"] = (
        data["inter_arrival_us"] / 1_000_000
    )

    return data


def extract_event_times(data):
    """
    Extract the unique event timestamps from processed trade data.

    The timestamps are:
        1. Removed if missing
        2. Deduplicated
        3. Sorted chronologically
        4. Given a clean integer index

    The resulting timestamps represent the event times used
    in the Poisson and Hawkes process analysis.
    """

    event_times = (
        data[["timestamp"]]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    return event_times


def event_times_to_seconds(event_times):
    """
    Convert absolute event timestamps into elapsed time from
    the first observed event.

    For event i:

        τ_i = t_i - t_1

    where:
        t_i = timestamp of event i
        t_1 = timestamp of the first event

    The result is expressed in seconds.

    This representation is useful for point-process calculations,
    where we work with elapsed time rather than absolute timestamps.
    """

    # Calculate elapsed time relative to the first event.
    event_difference = (
        event_times["timestamp"] - event_times["timestamp"].iloc[0]
    )

    # Convert pandas Timedelta values into seconds.
    return event_difference.dt.total_seconds()