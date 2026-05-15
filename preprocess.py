import pandas as pd
import os

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df.set_index("timestamp", inplace=True)

    full_range = pd.date_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(full_range)

    df["views"] = df["views"].ffill()

    # Features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df["rolling_mean_7"] = df["views"].rolling(7).mean()
    df["rolling_mean_30"] = df["views"].rolling(30).mean()

    df["rolling_std_7"] = df["views"].rolling(7).std()

    df["lag_1"] = df["views"].shift(1)
    df["lag_7"] = df["views"].shift(7)

    df = df.dropna()

    df = df.reset_index()
    df.rename(columns={"index": "timestamp"}, inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_pageviews.csv", index=False)

    return df