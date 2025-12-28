import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market Regime Control Center", layout="wide", page_icon="🎛️")

# --- 1. SIDEBAR: USER PREFERENCES ---
st.sidebar.header("🎛️ Signal Thresholds")

with st.sidebar.expander("1. Trend Definitions", expanded=True):
    # Default: 200 is the "Line in the Sand" for long-term Bull/Bear
    sma_window = st.number_input("Bull/Bear Line (SMA)", min_value=50, max_value=365, value=200, 
                                 help="Price above this moving average = Bull Market.")
    # Default: 25 is standard. 20 is aggressive. 30 is conservative.
    adx_threshold = st.slider("Trend Strength Min (ADX)", 10, 50, 25, 
                              help="Minimum ADX required to call a trend 'Sustained'. Below this = Choppy.")

with st.sidebar.expander("2. Risk & Volatility", expanded=True):
    # Default: 20 is the historic average. >20 usually implies stress.
    vix_high = st.slider("High Volatility Threshold (VIX)", 15, 40, 20, 
                         help="VIX levels above this are considered 'Volatile/Fearful'.")
    # Default: 70 is standard Overbought. 80 is extreme.
    rsi_overbought = st.slider("RSI Overbought", 60, 90, 70, 
                               help="RSI above this suggests the market is overheated (Melt-up risk).")
    # Default: 30 is standard Oversold.
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30, 
                             help="RSI below this suggests the market is panicked (Bounce likely).")

with st.sidebar.expander("3. Lookback Windows"):
    # Default: 20 days (1 trading month) is standard for relative rotation
    rotation_days = st.slider("Rotation Lookback (Days)", 5, 90, 20, 
                              help="Number of days to compare relative performance (e.g. Gold vs Stocks).")

# --- 2. DATA ENGINE (Cached) ---
# We cache this function so changing sliders doesn't re-download data every time
@st.cache_data(ttl=900) 
def get_data():
    tickers = {
        'Benchmarks': ['SPY', 'QQQ', 'IWM', 'GLD', 'BTC-USD', 'TLT'],
        'Sectors': ['XLK', 'XLE', 'XLF', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU'],
        'Indicators': ['^VIX', '^VIX3M', 'HYG', 'IEF']
    }
    all_symbols = [item for sublist in tickers.values() for item in sublist]
    # Fetch ample history to calculate max SMA (365) + buffers
    data = yf.download(all_symbols, period="2y", progress=False)
    return data, tickers

# --- 3. LOGIC ENGINE (Dynamic) ---
def process_signals(data, tickers):
    # Handle yfinance MultiIndex
    try:
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
    except KeyError:
        closes = data

    latest = closes.iloc[-1]
    
    # A. MARKET REGIME (SPY)
    spy_price = latest['SPY']
    spy_sma = closes['SPY'].rolling(window=sma_window).mean().iloc[-1]
    
    # ADX Calculation
    def calc_adx(high, low, close, lookback=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(lookback).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/lookback).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/lookback).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.rolling(lookback).mean().iloc[-1]

    adx_val = calc_adx(highs['SPY'], lows['SPY'], closes['SPY'])
    
    # RSI Calculation
    delta = closes['SPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
    
    # B. DETERMINING REGIME STRING
    if spy_price > spy_sma:
        trend = "BULL"
        if rsi_val > rsi_overbought: sub_trend = "OVERHEATED"
        elif adx_val > adx_threshold: sub_trend = "STRONG/SUSTAINED"
        else: sub_trend = "WEAK/CHOPPY"
    else:
        trend = "BEAR"
        if rsi_val < rsi_oversold: sub_trend = "OVERSOLD (Bounce?)"
        elif adx_val > adx_threshold: sub_trend = "CRASHING"
        else: sub_trend = "DRIFTING"

    # C. RELATIVE PERFORMANCE
    rel_perf = closes.pct_change(rotation_days).iloc[-1] * 100

    return {
        'trend': trend,
        'sub_trend': sub_trend,
        'spy_price': spy_price,
        'spy_sma': spy_sma,
        'adx': adx_val,
        'rsi': rsi_val,
        'vix': latest['^VIX'],
        'vix3m': latest['^VIX3M'],
        'rel_perf': rel_perf
    }

# --- 4. DASHBOARD RENDER ---
try:
    with st.spinner("Fetching Market Data..."):
        raw_data, ticker_map = get_data()
        
    sig = process_signals(raw_data, ticker_map)

    # HEADLINE
    st.title(f"Market Status: {sig['trend']} ({sig['sub_trend']})")
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: Yahoo Finance")
    
    # METRIC ROW
    c1, c2, c3, c4 = st.columns(4)
    
    # 1. Price vs SMA
    is_bull = sig['spy_price'] > sig['spy_sma']
    c1.metric("SPY vs Trend", f"${sig['spy_price']:.2f}", 
              f"{'Above' if is_bull else 'Below'} SMA{sma_window} (${sig['spy_sma']:.2f})",
              delta_color="normal" if is_bull else "inverse")
    
    # 2. VIX (Fear)
    vix_state = "High Volatility" if sig['vix'] > vix_high else "Steady"
    c2.metric("Volatility (VIX)", f"{sig['vix']:.2f}", 
              f"{vix_state} (>{vix_high})",
              delta_color="inverse") # Red if high (up)
    
    # 3. RSI (Overbought/Sold)
    rsi_state = "Neutral"
    if sig['rsi'] > rsi_overbought: rsi_state = "Overbought"
    if sig['rsi'] < rsi_oversold: rsi_state = "Oversold"
    c3.metric("Momentum (RSI)", f"{sig['rsi']:.1f}", rsi_state)

    # 4. Term Structure
    contango = sig['vix'] < sig['vix3m']
    c4.metric("VIX Term Structure", f"{sig['vix']/sig['vix3m']:.2f}", 
              "Healthy (Contango)" if contango else "DANGER (Backwardation)",
              delta_color="normal" if contango else "inverse")

    st.markdown("---")

    # CHARTS
    tab1, tab2 = st.tabs(["📊 Asset Class Rotation", "🏭 Sector Rotation"])
    
    with tab1:
        # Asset Class Bar Chart
        assets = ticker_map['Benchmarks']
        asset_perf = sig['rel_perf'][assets].sort_values(ascending=True)
        
        # Color logic based on user lookback
        fig = px.bar(asset_perf, x=asset_perf.values, y=asset_perf.index, orientation='h',
                     title=f"Winning Assets (Last {rotation_days} Days)",
                     labels={'x': 'Return %', 'y': 'Asset'},
                     text_auto='.2f',
                     color=asset_perf.values, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Sector Bar Chart
        sectors = ticker_map['Sectors']
        sec_perf = sig['rel_perf'][sectors].sort_values(ascending=True)
        
        fig2 = px.bar(sec_perf, x=sec_perf.values, y=sec_perf.index, orientation='h',
                     title=f"Sector Leaders & Laggards (Last {rotation_days} Days)",
                     labels={'x': 'Return %', 'y': 'Sector'},
                     text_auto='.2f',
                     color=sec_perf.values, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig2, use_container_width=True)

    # RAW DATA CHECK
    with st.expander("Show Raw Signal Data"):
        st.write("Current calculated metrics based on your settings:")
        st.json(sig)

except Exception as e:
    st.error(f"System Error: {e}")
