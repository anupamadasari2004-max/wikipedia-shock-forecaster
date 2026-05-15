import pandas as pd

df = pd.read_csv("data/raw/Artificial_intelligence_pageviews.csv")

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nDtypes:\n", df.dtypes)
print("\nNull values:\n", df.isnull().sum())
print("\nUnique values:\n", df.nunique())

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nMin date:", df["timestamp"].min())
print("Max date:", df["timestamp"].max())

print("\nZero views count:", (df["views"] == 0).sum())
print("Negative views count:", (df["views"] < 0).sum())

print("\nDuplicate rows:", df.duplicated().sum())