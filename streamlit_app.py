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
    Tries multiple tickers and returns the first successful fetch.
    If all fail, returns None.
    """
    tickers = ["^NSEBANK", "BANKNIFTY.NS"]
    for ticker in tickers:
        try:
            bank_nifty = yf.Ticker(ticker)
            data = bank_nifty.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                return current_price
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {e}")
            continue
    return None

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call option.
    Returns the call price, d1, and d2.
    """
    
    if T <= 0 or S <= 0 or K <= 0:
        return 0.0, None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European put option.
    Returns the put price, d1, and d2.
    """
   
    if T <= 0 or S <= 0 or K <= 0:
        return 0.0, None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price, d1, d2

# --------------------------------------------------------------------------------
# 2. STREAMLIT APP
# --------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Bank Nifty Option Price Calculator", layout="wide")
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
        value=51400,  # Example default value
        step=100
    )

    # Expiry date input
    expiry_date = st.sidebar.date_input(
        "📅 Expiry Date:",
        value=datetime.today() + timedelta(days=30),
        min_value=datetime.today()
    )

    # Calculate days to expiry
    today = datetime.today()
    delta = expiry_date - today.date()
    expiry_days = delta.days

    if expiry_days <= 0:
        st.sidebar.error("Expiry date must be in the future.")
        expiry_days = 30  # Default to 30 days

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
        count = st_autorefresh(interval=5000, limit=None, key="autorefresh")

    # --------------------------------------------------------------------------------
    # D. FETCH AND CALCULATE OPTION PRICES
    # --------------------------------------------------------------------------------
    # Calculate time to expiration in years
    T = expiry_days / 365.0

    # Fetch Bank Nifty Price
    with st.spinner('Fetching Bank Nifty price...'):
        S = fetch_bank_nifty_price()
    if S is None:
        st.sidebar.warning("Unable to fetch Bank Nifty price. Please enter manually.")
        S = st.sidebar.number_input(
            "💰 Current Bank Nifty Price (S):",
            min_value=10000.0,
            max_value=100000.0,
            value=51271.05,  # Example default value
            step=100.0
        )

    # Calculate Call and Put prices
    call_price, d1_call, d2_call = black_scholes_call(S, strike_price, T, r, sigma)
    put_price, d1_put, d2_put = black_scholes_put(S, strike_price, T, r, sigma)

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

    # --------------------------------------------------------------------------------
    # G. DEBUGGING INFORMATION
    # --------------------------------------------------------------------------------
    st.subheader("🔍 Debugging Information")

    st.markdown("**Parameters Used:**")
    st.write(f"- **S (Underlying Price):** {S:.2f}")
    st.write(f"- **K (Strike Price):** {strike_price}")
    st.write(f"- **T (Time to Expiration):** {T:.4f} years")
    st.write(f"- **r (Risk-Free Rate):** {r*100:.2f}%")
    st.write(f"- **σ (Volatility):** {sigma*100:.2f}%")

    st.markdown("**Call Option Components:**")
    st.write(f"- **d1:** {d1_call:.4f}")
    st.write(f"- **d2:** {d2_call:.4f}")
    st.write(f"- **N(d1):** {norm.cdf(d1_call):.4f}")
    st.write(f"- **N(d2):** {norm.cdf(d2_call):.4f}")
    st.write(f"- **Call Price:** ₹{call_price:.2f}")

    st.markdown("**Put Option Components:**")
    st.write(f"- **d1:** {d1_put:.4f}")
    st.write(f"- **d2:** {d2_put:.4f}")
    st.write(f"- **N(-d1):** {norm.cdf(-d1_put):.4f}")
    st.write(f"- **N(-d2):** {norm.cdf(-d2_put):.4f}")
    st.write(f"- **Put Price:** ₹{put_price:.2f}")

    # --------------------------------------------------------------------------------
    # H. PUT PRICE VS STRIKE PRICE PLOT
    # --------------------------------------------------------------------------------
    st.subheader("📉 Put Price vs. Strike Price")

    # Define a range of strike prices around the current K
    K_min = strike_price - 500  # Adjust as needed
    K_max = strike_price + 500
    K_step = 100
    K_range = np.arange(K_min, K_max + K_step, K_step)

    # Calculate Put prices for each K in the range
    put_prices_range = []
    for K in K_range:
        P, _, _ = black_scholes_put(S, K, T, r, sigma)
        put_prices_range.append(P)

    # Create a DataFrame for plotting
    df_put = pd.DataFrame({
        'Strike Price': K_range,
        'Put Price': put_prices_range
    })

    # Plot using Streamlit's built-in charting
    st.line_chart(df_put.set_index('Strike Price'))

    # --------------------------------------------------------------------------------
    # I. OPTION PRICE COMPONENTS (Optional)
    # --------------------------------------------------------------------------------
    """
    Uncomment the following section if you want to display option price components for multiple K's.
    This can help in debugging and understanding how Put prices change with K.

    for K in K_range:
        P, d1, d2 = black_scholes_put(S, K, T, r, sigma)
        st.write(f"**K = {K}:** P = ₹{P:.2f}, d1 = {d1:.4f}, d2 = {d2:.4f}")
    """

if __name__ == "__main__":
    main()
