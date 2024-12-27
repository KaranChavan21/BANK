import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def fetch_bank_nifty_price():
    """
    Fetches the current Bank Nifty index price using yfinance.
    Note: Yahoo Finance may not have a direct ticker for Bank Nifty.
    Often, Bank Nifty futures or ETFs can be used as proxies.
    For this example, we'll use the ticker "^NSEBANK" which represents Bank Nifty.
    """
    ticker = "^NSEBANK"
    try:
        bank_nifty = yf.Ticker(ticker)
        data = bank_nifty.history(period="1d")
        current_price = data['Close'].iloc[-1]
        return current_price
    except Exception as e:
        st.warning(f"Error fetching Bank Nifty price: {e}")
        return None

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call option.
    """
    #K = K - 50
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European put option.
    """
    #K = K + 50
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price

# --------------------------------------------------------------------------------
# 2. STREAMLIT APP
# --------------------------------------------------------------------------------
def main():
    st.title("📈 Bank Nifty Option Price Calculator")

    # Initialize session state variables
    if 'run_loop' not in st.session_state:
        st.session_state.run_loop = False
    if 'time_series' not in st.session_state:
        st.session_state.time_series = []
    if 'latest_call' not in st.session_state:
        st.session_state.latest_call = 0.0
    if 'latest_put' not in st.session_state:
        st.session_state.latest_put = 0.0

    # --------------------------------------------------------------------------------
    # A. INPUTS SECTION
    # --------------------------------------------------------------------------------
    st.sidebar.header("🔧 Input Parameters")

    # Strike price
    strike_price = st.sidebar.number_input(
        "🔢 Strike Price (K):",
        min_value=10000,
        max_value=100000,
        value=51400,
        step=100
    )

    # Expiry in days
    expiry_days = st.sidebar.number_input(
        "📅 Days to Expiry:",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )

    # Risk-Free Rate and Volatility
    r = st.sidebar.number_input(
        "💹 Risk-Free Interest Rate (annual %) (e.g., 7 for 7%):",
        min_value=0.0,
        max_value=20.0,
        value=7.0,
        step=0.1
    ) / 100.0

    sigma = st.sidebar.number_input(
        "📊 Volatility of Bank Nifty (annual %) (e.g., 15 for 15%):",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1
    ) / 100.0

    # --------------------------------------------------------------------------------
    # B. START / STOP BUTTONS
    # --------------------------------------------------------------------------------
    st.sidebar.header("▶️ Control Panel")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_button = st.button("▶️ Start", key='start')
    with col2:
        stop_button = st.button("⏹️ Stop", key='stop')

    # Handle button clicks
    if start_button:
        st.session_state.run_loop = True
    if stop_button:
        st.session_state.run_loop = False

    # Display the status
    if st.session_state.run_loop:
        st.sidebar.success("🔄 Continuous update is **ON**")
    else:
        st.sidebar.warning("⏸️ Continuous update is **OFF**")

    # --------------------------------------------------------------------------------
    # C. AUTOREFRESH
    # --------------------------------------------------------------------------------
    if st.session_state.run_loop:
        # Auto-refresh every 5 seconds
        st_autorefresh(interval=4000, limit=None, key="autorefresh")
    
    # --------------------------------------------------------------------------------
    # D. FETCH AND CALCULATE OPTION PRICES
    # --------------------------------------------------------------------------------
    # Calculate time to expiration in years
    T = expiry_days / 365.0

    # Fetch Bank Nifty Price
    S = fetch_bank_nifty_price()
    if S is None:
        S = 0.0

    # Calculate Call and Put prices
    call_price = black_scholes_call(S, strike_price, T, r, sigma)
    put_price  = black_scholes_put(S, strike_price, T, r, sigma)

    # Update latest prices
    st.session_state.latest_call = call_price
    st.session_state.latest_put  = put_price

    # Append to time series
    now_str = datetime.now().strftime("%H:%M:%S")
    st.session_state.time_series.append({
        "time": now_str,
        "call": call_price,
        "put": put_price,
    })

    # Limit to last 50 records
    st.session_state.time_series = st.session_state.time_series[-50:]

    # --------------------------------------------------------------------------------
    # E. DISPLAY LATEST RESULTS
    # --------------------------------------------------------------------------------
    st.subheader("📊 Latest Option Prices")

    cols = st.columns(2)
    with cols[0]:
        st.metric(
            label="💼 Call Option Price",
            value=f"₹{call_price:.2f}"
        )
    with cols[1]:
        st.metric(
            label="📉 Put Option Price",
            value=f"₹{put_price:.2f}"
        )

    st.write(f"**Current Bank Nifty Price (S):** {S:.2f}")
    st.write(f"**Strike Price (K):** {strike_price}")
    st.write(f"**Time to Expiration (T):** {T*365:.0f} days ({T:.4f} years)")
    st.write(f"**Risk-Free Rate (r):** {r*100:.2f}%")
    st.write(f"**Volatility (σ):** {sigma*100:.2f}%")

    # --------------------------------------------------------------------------------
    # F. DISPLAY OPTION PRICE MOVEMENT
    # --------------------------------------------------------------------------------
    st.subheader("📈 Option Price Movement Over Time")

    if st.session_state.time_series:
        # Create DataFrame for charting
        df = pd.DataFrame(st.session_state.time_series)
        df.set_index("time", inplace=True)

        # Plot Call and Put prices
        st.line_chart(df[["call", "put"]])
    else:
        st.write("No data to display yet. Click 'Start' to begin.")

if __name__ == "__main__":
    main()
