import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv("data/processed/clean_pageviews.csv")

features = [
    "lag_1", "lag_7",
    "rolling_mean_7", "rolling_mean_30",
    "day_of_week", "month"
]

X = df[features]
y = df["views"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/forecast_model.pkl")

print("Model trained successfully!")