import pandas as pd


def process_trade_data(data):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["time"], unit="us", utc=True)
    data["inter_arrival_us"] = data["time"].diff()
    data["inter_arrival_seconds"] = data["inter_arrival_us"] / 1_000_000
    return data

if __name__ == "__main__":

    from data_loader import load_trade_data

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=1_000_000
    )

    processed_data = process_trade_data(data)

    print(processed_data.head())
    print(processed_data.tail())
    print(processed_data.shape)

    print(
    processed_data[
        ["id", "timestamp", "inter_arrival_us", "inter_arrival_seconds",]
    ]
)