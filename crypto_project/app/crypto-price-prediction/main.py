import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Crypto Price Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme and animations
st.markdown(
    """
    <style>
    /* Background and text color for dark theme */
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
        animation: fadeIn 1.5s ease-in-out;
    }
    .header-title {
        font-size: 2.5rem;
        color: #00d4ff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        animation: slideIn 2s ease-in-out;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #b3b3b3;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Animation for fade-in effect */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Animation for slide-in effect */
    @keyframes slideIn {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .crypto-icon {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    img {
        width: 150px;
        animation: rotate 3s linear infinite;
    }
    /* Animation for rotating logo */
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page content
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="header-title">📈 Crypto Price Prediction and Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting Crypto Prices with Machine Learning</div>', unsafe_allow_html=True)

# Load and display the Bitcoin logo with animation
bitcoin_image = Image.open("bitcoin_logo.png")  # Ensure this file is in your working directory
st.markdown('<div class="crypto-icon">', unsafe_allow_html=True)
st.image(bitcoin_image, width=150)
st.markdown('</div>', unsafe_allow_html=True)

# Add a brief description
st.write("Welcome to the **Crypto Price Prediction App**! 🚀 This app leverages cutting-edge machine learning models to predict cryptocurrency prices and provide actionable recommendations.")

# Closing div
st.markdown('</div>', unsafe_allow_html=True)
