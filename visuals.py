import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data/processed/clean_pageviews.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_basic_trend(df):
    plt.figure()
    plt.plot(df["timestamp"], df["views"])
    plt.title("Daily Pageviews Trend")
    plt.xlabel("Date")
    plt.ylabel("Views")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_rolling_trend(df):
    plt.figure()
    plt.plot(df["timestamp"], df["views"], label="Actual")
    plt.plot(df["timestamp"], df["rolling_mean_7"], label="7-day Avg")
    plt.plot(df["timestamp"], df["rolling_mean_30"], label="30-day Avg")
    
    plt.legend()
    plt.title("Trend with Rolling Averages")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_seismograph(df):
    plt.figure()
    
    # baseline (30-day avg)
    plt.plot(df["timestamp"], df["rolling_mean_30"], linestyle="--", label="Baseline")
    
    # actual signal
    plt.plot(df["timestamp"], df["views"], label="Actual")
    
    # highlight spikes
    threshold = df["rolling_mean_30"] + 2 * df["rolling_std_7"]
    spikes = df[df["views"] > threshold]
    
    plt.scatter(spikes["timestamp"], spikes["views"])
    
    plt.title("Attention Seismograph (Shock Detection)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    
    plot_basic_trend(df)
    plot_rolling_trend(df)
    plot_seismograph(df)