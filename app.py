import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime

from data_fetch import fetch_wikipedia_data
from preprocess import preprocess_data

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Wiki Shock Forecaster", layout="wide")

st.title("📊 Wikipedia Attention Shock Forecaster")

# ---------------- POPULAR ARTICLES ----------------
POPULAR_ARTICLES = [
    "Artificial_intelligence",
    "Deep_learning",
    "Machine_learning",
    "ChatGPT",
    "Elon_Musk",
    "India",
    "Python_(programming_language)",
    "Cristiano_Ronaldo",
    "World_War_II",
    "Narendra_Modi"
]

# ---------------- INPUTS ----------------
st.sidebar.header("🔧 User Inputs")

selected_article = st.sidebar.selectbox("Choose popular article", POPULAR_ARTICLES)

custom_article = st.sidebar.text_input("Or enter custom article", "")

article = custom_article if custom_article.strip() != "" else selected_article
article = article.replace(" ", "_")

start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))

sensitivity = st.sidebar.slider("Shock Sensitivity", 1.0, 4.0, 2.0, 0.1)

start = start_date.strftime("%Y%m%d")
end = end_date.strftime("%Y%m%d")

st.sidebar.info("Use '_' instead of spaces or select from dropdown.")

# ---------------- DISPLAY INPUTS ----------------
st.subheader("📌 Selected Inputs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Article", article)
col2.metric("Start Date", str(start_date))
col3.metric("End Date", str(end_date))
col4.metric("Sensitivity", sensitivity)

st.info(f"Analyzing '{article}' from {start_date} to {end_date}")

st.write("---")

# ---------------- MAIN ----------------
if st.button("Fetch & Analyze Data"):

    df_raw = fetch_wikipedia_data(article, start, end)

    if df_raw is None:
        st.error("❌ Invalid article or date range.")
        st.stop()

    df = preprocess_data(f"data/raw/{article}_pageviews.csv")

    model = joblib.load("models/forecast_model.pkl")

    features = [
        "lag_1", "lag_7",
        "rolling_mean_7", "rolling_mean_30",
        "day_of_week", "month"
    ]

    df["predicted"] = model.predict(df[features])

    # Residuals
    df["residual"] = df["views"] - df["predicted"]

    threshold = df["residual"].mean() + sensitivity * df["residual"].std()
    df["shock"] = df["residual"] > threshold

    # Severity
    df["severity"] = df["residual"] / df["residual"].std()

    # Health
    latest = df.iloc[-1]["residual"]
    if latest > threshold:
        health = "🚨 Shock"
    elif latest > threshold * 0.5:
        health = "⚠️ Elevated"
    else:
        health = "✅ Normal"

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trend",
        "🤖 Forecast",
        "⚡ Shock Detection",
        "📊 Insights"
    ])

    # TREND
    with tab1:
        fig, ax = plt.subplots()
        ax.plot(df["timestamp"], df["views"])
        st.pyplot(fig)

    # FORECAST
    with tab2:
        fig, ax = plt.subplots()
        ax.plot(df["timestamp"], df["views"], label="Actual")
        ax.plot(df["timestamp"], df["predicted"], label="Predicted")
        ax.legend()
        st.pyplot(fig)

    # SHOCK
    with tab3:
        shocks = df[df["shock"] == True]

        st.metric("Total Shocks", len(shocks))
        st.metric("Attention Status", health)

        fig, ax = plt.subplots()
        ax.plot(df["timestamp"], df["views"])
        ax.scatter(shocks["timestamp"], shocks["views"])
        st.pyplot(fig)

        st.dataframe(shocks[["timestamp", "views", "severity"]])

    # INSIGHTS
    with tab4:
        st.write("### Top 5 Shocks")
        top = df.sort_values("severity", ascending=False).head(5)
        st.dataframe(top[["timestamp", "views", "severity"]])

        st.write("### Monthly Trend")
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
        st.bar_chart(df.groupby("month")["views"].mean())

        st.success("Detected spikes likely due to major real-world events.")