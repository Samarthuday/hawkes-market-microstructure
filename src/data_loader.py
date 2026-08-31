import pandas as pd


def load_trade_data(file_path, nrows=None):
    columns = [
        "trade_id",
        "price",
        "quantity",
        "quote_quantity",
        "timestamp_us",
        "is_buyer_maker",
        "is_best_match",
    ]
    df = pd.read_csv(file_path, names=columns, nrows=nrows)
    return df

if __name__ == "__main__":

    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=10
    )

    print(data)
    print(data.dtypes)