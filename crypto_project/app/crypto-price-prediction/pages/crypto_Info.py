# Import streamlit
import streamlit as st

# Import libraries
import datetime as dt
import pandas as pd
import ccxt

# Configure the page
st.set_page_config(
    page_title="Crypto Info",
    page_icon="💰",
)


# Function to fetch cryptocurrency symbols from an exchange
def fetch_cryptos():
    exchange = ccxt.binance()  # Use Binance as an example
    markets = exchange.load_markets()
    crypto_dict = {symbol: markets[symbol]['info']['baseAsset'] for symbol in markets}
    return crypto_dict


# Function to fetch cryptocurrency data
def fetch_crypto_info(symbol):
    exchange = ccxt.binance()  # Use Binance as an example
    ticker = exchange.fetch_ticker(symbol)
    return {
        "symbol": ticker['symbol'],
        "currentPrice": ticker['last'],
        "dayHigh": ticker['high'],
        "dayLow": ticker['low'],
        "24hVolume": ticker['quoteVolume'],
        "percentageChange": ticker['percentage'],
    }


# Function to fetch cryptocurrency history
def fetch_crypto_history(symbol, timeframe='1d', limit=90):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# Sidebar for cryptocurrency selection
st.sidebar.markdown("## **User Input Features**")

# Fetch cryptocurrency list
crypto_dict = fetch_cryptos()

# Add a dropdown for selecting the cryptocurrency
st.sidebar.markdown("### **Select Cryptocurrency**")
crypto = st.sidebar.selectbox("Choose a cryptocurrency", list(crypto_dict.keys()))

# Add a disabled input for the cryptocurrency symbol
st.sidebar.markdown("### **Cryptocurrency Symbol**")
st.sidebar.text_input(label="Selected Symbol", placeholder=crypto, disabled=True)

# Fetch cryptocurrency data
try:
    crypto_data_info = fetch_crypto_info(crypto)
except Exception as e:
    st.error(f"Error: Unable to fetch the cryptocurrency data. {e}")
    st.stop()

# Title and Subtitle
st.markdown("# **Crypto Info Plus**")
st.markdown("##### **Enhancing Your Cryptocurrency Market Insights**")

# Display Basic Information
st.markdown("## **Basic Information**")
st.dataframe(pd.DataFrame([crypto_data_info]).T, hide_index=True, width=500)

# Display Market Data
st.markdown("## **Market Data**")
col1, col2 = st.columns(2)

col1.metric("Current Price", crypto_data_info["currentPrice"])
col2.metric("24h High", crypto_data_info["dayHigh"])
col1.metric("24h Low", crypto_data_info["dayLow"])
col2.metric("24h Volume", crypto_data_info["24hVolume"])
col1.metric("24h Change (%)", f"{crypto_data_info['percentageChange']}%")

# Fetch and display cryptocurrency history
st.markdown("## **Historical Data**")
try:
    history_df = fetch_crypto_history(crypto)
    st.line_chart(history_df.set_index('timestamp')['close'], use_container_width=True)
except Exception as e:
    st.error(f"Error: Unable to fetch historical data. {e}")
