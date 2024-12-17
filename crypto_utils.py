import yfinance as yf
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt


# Fetch historical crypto data
def fetch_crypto_history(ticker, period, interval):
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return None


# Fetch live price from Coinbase
def fetch_live_price(symbol):
    url = f"https://api.coinbase.com/v2/prices/{symbol}/spot"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json()["data"]["amount"])
    return None


# Step 1: Save Predictions to Storage
def save_predictions_to_csv(predictions, forecast_index, crypto_ticker):
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Predicted Price": predictions.values,
        "Crypto": crypto_ticker
    })
    forecast_df.to_csv("predictions.csv", mode="a", index=False, header=not pd.io.common.file_exists("predictions.csv"))


# Step 2: Fetch Actual Prices Daily
def fetch_actual_prices(dates, crypto_ticker):
    actual_data = yf.download(crypto_ticker, start=dates.min(), end=dates.max() + pd.Timedelta(days=1))
    return actual_data["Close"]


# Step 3: Automate Comparison
def compare_predictions_and_actuals(prediction_file):
    # Load stored predictions
    predictions_df = pd.read_csv(prediction_file)

    # Fetch actual prices for forecasted dates
    dates = pd.to_datetime(predictions_df["Date"])
    crypto_ticker = predictions_df["Crypto"].iloc[0]
    actual_prices = fetch_actual_prices(dates, crypto_ticker)

    # Merge predictions with actuals
    predictions_df["Actual Price"] = predictions_df["Date"].map(actual_prices.to_dict())
    predictions_df["Difference"] = predictions_df["Actual Price"] - predictions_df["Predicted Price"]

    # Calculate accuracy metrics
    predictions_df = predictions_df.dropna()
    if not predictions_df.empty:
        rmse = np.sqrt(mean_squared_error(predictions_df["Actual Price"], predictions_df["Predicted Price"]))
        mae = mean_absolute_error(predictions_df["Actual Price"], predictions_df["Predicted Price"])
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    else:
        print("No actual data available for comparison.")
    return predictions_df
