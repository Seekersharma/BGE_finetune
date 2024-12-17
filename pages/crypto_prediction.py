# Imports
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import datetime as dt
from crypto_utils import save_predictions_to_csv, compare_predictions_and_actuals  # Utility imports

# Configure the page
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="💰",
)

# Function to fetch cryptocurrencies
def fetch_cryptos():
    return {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Binance Coin": "BNB-USD",
        "Ripple": "XRP-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "Dogecoin": "DOGE-USD",
        "Polkadot": "DOT-USD",
        "Litecoin": "LTC-USD",
        "Shiba Inu": "SHIB-USD",
        "Stellar": "XLM-USD",
        "Hedera": "HBAR-USD",
        "XDC Network": "XDC-USD",
        "XRP": "XRP-USD"
    }

# Fetch historical data from Yahoo Finance
def fetch_crypto_history(crypto_ticker, period, interval):
    try:
        crypto_data = yf.Ticker(crypto_ticker)
        data = crypto_data.history(period=period, interval=interval)
        if data.empty:
            st.error(f"No data fetched for {crypto_ticker}. Please try another period or interval.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch live cryptocurrency prices from Coinbase
def fetch_live_price(symbol):
    url = f"https://api.coinbase.com/v2/prices/{symbol}/spot"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json()["data"]["amount"])
    else:
        st.error(f"Error fetching live price for {symbol}: {response.status_code}")
        return None

# Generate ARIMA forecasts
def generate_arima_forecasts(data, steps):
    try:
        if data is None or data.empty:
            st.error("Insufficient data for ARIMA predictions.")
            return None, None, None, None, None

        train_size = int(len(data) * 0.9)
        train_data = data["Close"][:train_size]
        test_data = data["Close"][train_size:]

        model = ARIMA(train_data, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps)
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)

        return train_data, test_data, forecast, forecast_index, model_fit
    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return None, None, None, None, None

# Generate recommendations table
def generate_recommendation_table(data):
    recommendations = []
    for i in range(1, len(data)):
        signal = "BUY" if data["Close"].iloc[i] > data["Close"].iloc[i - 1] else "SELL"
        recommendations.append({
            "Date": data.index[i].strftime("%Y-%m-%d"),
            "Close Price": round(data["Close"].iloc[i], 2),
            "Signal": signal
        })
    return pd.DataFrame(recommendations)

# Day of the Week Analysis
def day_of_week_analysis(data):
    data["DayOfWeek"] = data.index.day_name()
    return data.groupby("DayOfWeek")["Close"].mean()

# Compare forecast to actuals
def compare_forecast_to_actual(forecast, forecast_index, crypto_ticker):
    try:
        actual_data = yf.download(crypto_ticker, start=forecast_index[0], end=forecast_index[-1] + pd.Timedelta(days=1))
        actual_close = actual_data["Close"].reindex(forecast_index, method="nearest")

        comparison_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecasted Price": forecast.values,
            "Actual Price": actual_close.values
        }).dropna()

        comparison_df["Difference"] = (comparison_df["Actual Price"] - comparison_df["Forecasted Price"]).round(2)
        return comparison_df
    except Exception as e:
        st.error(f"Error comparing forecast to actual: {e}")
        return None

# Sidebar configuration
st.sidebar.markdown("## User Input Features")
crypto_dict = fetch_cryptos()
selected_crypto = st.sidebar.selectbox("Select a cryptocurrency", list(crypto_dict.keys()))
crypto_ticker = crypto_dict[selected_crypto]

period = st.sidebar.selectbox("Select a period", ["1mo", "3mo", "6mo", "1y", "5y"])
interval = st.sidebar.selectbox("Select an interval", ["1d", "1wk"])
steps = st.sidebar.slider("Select Lookahead Days", min_value=1, max_value=30, value=7)

# Main section
st.title("Cryptocurrency Price Prediction")
st.write(f"Analyzing {selected_crypto} ({crypto_ticker})")

# Fetch historical data
crypto_data = fetch_crypto_history(crypto_ticker, period, interval)
if crypto_data is not None:
    # Historical Data Visualization
    st.subheader("Historical Data - Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=crypto_data.index,
        open=crypto_data["Open"],
        high=crypto_data["High"],
        low=crypto_data["Low"],
        close=crypto_data["Close"]
    )])
    st.plotly_chart(fig)

    # Live Price & Signal
    st.subheader("Live Price & Trading Signal")
    live_price = fetch_live_price(crypto_ticker)
    if live_price:
        last_close = crypto_data["Close"].iloc[-1]
        signal = "BUY" if live_price > last_close else "SELL" if live_price < last_close else "HOLD"
        st.metric(label="Current Live Price", value=f"${live_price:.2f}")
        st.write(f"**Trading Signal:** {signal}")

    # ARIMA Forecasting
    st.subheader("ARIMA Model Predictions")
    train_data, test_data, forecast, forecast_index, model_fit = generate_arima_forecasts(crypto_data, steps)
    if forecast is not None:
        forecast_df = pd.DataFrame({"Date": forecast_index, "Forecasted Price": forecast.values})
        st.dataframe(forecast_df)

        save_predictions_to_csv(forecast, forecast_index, crypto_ticker)
        st.success("Predictions saved for future comparison.")

    # Recommendations Table
    st.subheader("Recommendation Table")
    st.dataframe(generate_recommendation_table(crypto_data))

    # Day of the Week Analysis
    st.subheader("Day of the Week Analysis")
    st.bar_chart(day_of_week_analysis(crypto_data))

    # Compare Predictions to Actual
    st.subheader("Forecast vs. Actual Comparison")
    comparison_df = compare_forecast_to_actual(forecast, forecast_index, crypto_ticker)
    if comparison_df is not None:
        st.dataframe(comparison_df)
        st.line_chart(comparison_df.set_index("Date")[["Forecasted Price", "Actual Price"]])

    # Compare Stored Predictions to Actual
    st.subheader("Stored Forecast Comparison")
    if st.button("Compare Predictions with Actual Prices"):
        comparison_df = compare_predictions_and_actuals("predictions.csv")
        if not comparison_df.empty:
            st.dataframe(comparison_df)
        else:
            st.info("No actual data available for comparison.")
