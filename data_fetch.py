import requests
import pandas as pd
import os

def fetch_wikipedia_data(article, start, end):

    article = article.replace(" ", "_")

    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{article}/daily/{start}/{end}"

    headers = {
        "User-Agent": "WikiShockApp (student project)"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return None

    data = response.json()
    items = data.get("items", [])

    if not items:
        return None

    df = pd.DataFrame(items)
    df = df[["timestamp", "views"]]

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d00")
    df = df.sort_values("timestamp")

    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{article}_pageviews.csv"
    df.to_csv(file_path, index=False)

    return df