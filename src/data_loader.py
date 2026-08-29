import pandas as pd



def load_trade_data(file_path, nrows=None):
    columns = [
        "id",
        "price",
        "qty",
        "quote_qty",
        "time",
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