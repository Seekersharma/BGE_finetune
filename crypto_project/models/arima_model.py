import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch historical data from Binance API
def fetch_historical_data(symbol="BTCUSDT", interval="1d", start_year=2017):
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(pd.Timestamp.now().timestamp() * 1000)  # Current time in milliseconds
    start_time = int(pd.Timestamp(f"{start_year}-01-01").timestamp() * 1000)  # Start time in ms

    all_data = []
    limit = 1000  # Maximum rows per request

    while start_time < end_time:
        # Fetch data from Binance
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit,
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)

            # Update the start_time to the last retrieved timestamp
            start_time = data[-1][0] + 1
        else:
            print(f"Error fetching data: {response.status_code}, {response.text}")
            break

    # Convert data to a DataFrame
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(all_data, columns=columns)

    # Convert data types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = pd.to_numeric(df["close"])

    return df[["timestamp", "close"]]

# Clean data function
def clean_data(df):
    df = df.dropna()  # Remove rows with missing values
    df = df[df['close'] > 0]  # Filter invalid prices
    return df

# Train ARIMA model
def train_arima_model(train_data):
    model = ARIMA(train_data['close'], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

# Fetch live price function
def fetch_live_price():
    url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    response = requests.get(url)
    if response.status_code == 200:
        live_data = response.json()
        return float(live_data['data']['amount'])
    else:
        raise Exception(f"Error fetching live price: {response.status_code}")

# Generate Buy/Sell signal
def generate_signal(predicted_price, current_price, threshold=0.02):
    if predicted_price > current_price * (1 + threshold):
        return "Buy Signal"
    elif predicted_price < current_price * (1 - threshold):
        return "Sell Signal"
    else:
        return "Hold Signal"

# Main execution
if __name__ == "__main__":
    try:
        # Fetch historical data
        df = fetch_historical_data(symbol="BTCUSDT", interval="1d", start_year=2017)

        # Clean data
        df = clean_data(df)

        # Split into train-test
        train_data = df[:-30]  # Last 30 days as test
        test_data = df[-30:]

        # Train ARIMA model
        arima_model_fit = train_arima_model(train_data)

        # Predict the next day's close price
        next_day_prediction = arima_model_fit.forecast(steps=1)[0]

        # Fetch live price
        current_price = fetch_live_price()

        # Generate Buy/Sell signal
        signal = generate_signal(next_day_prediction, current_price)

        # Display results
        print(f"Predicted Price: {next_day_prediction}")
        print(f"Current Live Price: {current_price}")
        print(f"Signal: {signal}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
