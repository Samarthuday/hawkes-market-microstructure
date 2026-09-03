import pandas as pd


def load_trade_data(file_path, nrows=None):
    """
    Load Binance trade data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing trade-level data.

    nrows : int, optional
        Number of rows to read. If None, the entire file is loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the raw trade data.
    """

    # Binance trade data is stored without column headers.
    # We therefore explicitly assign names to each column.
    columns = [
        "trade_id",          # Unique identifier for each trade
        "price",             # Execution price of the trade
        "quantity",          # Amount of BTC traded
        "quote_quantity",    # Value of the trade in USDT
        "timestamp_us",      # Trade timestamp in microseconds
        "is_buyer_maker",    # Whether the buyer was the maker
        "is_best_match",     # Whether the trade was the best match
    ]

    # Read the CSV file into a pandas DataFrame.
    # nrows can be used to load only a subset of the data,
    # which is useful during development and testing.
    df = pd.read_csv(
        file_path,
        names=columns,
        nrows=nrows
    )

    return df


if __name__ == "__main__":

    # Load the first 10 trades as a small test sample.
    data = load_trade_data(
        "data/BTCUSDT-trades-2025-01.csv",
        nrows=10
    )

    # Display the loaded data.
    print(data)

    # Display the data type of each column.
    # This helps verify that pandas has interpreted the raw data correctly.
    print(data.dtypes)