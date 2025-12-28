import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Regime & Cycle Explorer", layout="wide", page_icon="🧭")

# --- 1. SIDEBAR: USER PREFERENCES ---
st.sidebar.header("🎛️ Control Panel")

with st.sidebar.expander("1. Market Regime Settings", expanded=True):
    sma_window = st.number_input("Bull/Bear Line (SMA)", min_value=50, max_value=365, value=200, 
                                 help="Price above this moving average = Bull Market.")
    adx_threshold = st.slider("Trend Strength Min (ADX)", 10, 50, 25)
    vix_high = st.slider("High Volatility (VIX)", 15, 40, 20)
    rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)

with st.sidebar.expander("2. Rotation & Cycles", expanded=True):
    # RRG Logic settings
    st.write("**Cycle Sensitivity**")
    cycle_window = st.slider("Trend Lookback", 10, 100, 20, 
                             help="Days used to determine the core trend (X-Axis).")
    momentum_window = st.slider("Momentum Lookback", 3, 30, 10, 
                                help="Days used to determine acceleration/velocity (Y-Axis).")

# --- 2. DATA ENGINE (Cached) ---
@st.cache_data(ttl=900) 
def get_data():
    tickers = {
        'Benchmarks': ['SPY', 'QQQ', 'IWM', 'GLD', 'BTC-USD', 'TLT'],
        'Sectors': ['XLK', 'XLE', 'XLF', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU'],
        'Indicators': ['^VIX', '^VIX3M', 'HYG', 'IEF']
    }
    all_symbols = [item for sublist in tickers.values() for item in sublist]
    
    # Fetch ample history
    data = yf.download(all_symbols, period="2y", progress=False)
    return data, tickers

# --- 3. LOGIC ENGINES ---

def calculate_rrg(df, sectors, benchmark='SPY'):
    """
    Calculates Relative Rotation Graph components:
    1. RS-Ratio (Trend): X-Axis
    2. RS-Momentum (Rate of Change): Y-Axis
    """
    results = []
    
    # Prepare Benchmark Close
    bench_close = df['Close'][benchmark].ffill()
    
    for sector in sectors:
        sec_close = df['Close'][sector].ffill()
        
        # 1. Relative Strength (RS) = Sector / Benchmark
        rs = sec_close / bench_close
        
        # 2. RS-Ratio (Trend)
        # We normalize around 100. 
        # If RS > Moving Average of RS, trend is Up.
        rs_mean = rs.rolling(window=cycle_window).mean()
        rs_ratio = 100 + ((rs - rs_mean) / rs_mean) * 100
        
        # 3. RS-Momentum (Velocity)
        # Is the Ratio moving up or down? (ROC of the Ratio)
        rs_mom = 100 + rs_ratio.diff(momentum_window)
        
        # Get latest values
        curr_ratio = rs_ratio.iloc[-1]
        curr_mom = rs_mom.iloc[-1]
        
        # Classify Quadrant
        if curr_ratio > 100 and curr_mom > 100:
            status = "LEADING (Strong)"
            color = "green"
        elif curr_ratio > 100 and curr_mom < 100:
            status = "WEAKENING (Plateaued)"
            color = "yellow"
        elif curr_ratio < 100 and curr_mom < 100:
            status = "LAGGING (Weak)"
            color = "red"
        else:
            status = "IMPROVING (Up & Coming)"
            color = "blue"
            
        results.append({
            'Sector': sector,
            'RS_Ratio': curr_ratio,
            'RS_Momentum': curr_mom,
            'Status': status,
            'Color': color
        })
        
    return pd.DataFrame(results)

def process_market_regime(data):
    # Extract & FFill
    try:
        closes = data['Close'].ffill()
        highs = data['High'].ffill()
        lows = data['Low'].ffill()
    except KeyError:
        closes = data.ffill()
        highs = closes
        lows = closes

    latest = closes.iloc[-1]
    
    # 1. SPY TREND
    spy_price = latest['SPY']
    spy_sma = closes['SPY'].rolling(window=sma_window).mean().iloc[-1]
    
    # 2. ADX (Trend Strength)
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
        sum_di = plus_di + minus_di
        dx = 100 * (abs(plus_di - minus_di) / sum_di)
        return dx.rolling(lookback).mean().iloc[-1]

    adx_val = calc_adx(highs['SPY'], lows['SPY'], closes['SPY'])
    
    # 3. RSI (Momentum)
    delta = closes['SPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 4. Regime Text
    if pd.isna(spy_sma): 
        trend = "LOADING..."
        sub_trend = "Insufficient Data"
    elif spy_price > spy_sma:
        trend = "BULL"
        if rsi_val > rsi_overbought: sub_trend = "OVERHEATED"
        elif adx_val > adx_threshold: sub_trend = "STRONG/SUSTAINED"
        else: sub_trend = "WEAK/CHOPPY"
    else:
        trend = "BEAR"
        if rsi_val < rsi_oversold: sub_trend = "OVERSOLD (Bounce?)"
        elif adx_val > adx_threshold: sub_trend = "CRASHING"
        else: sub_trend = "DRIFTING"

    return {
        'trend': trend, 'sub_trend': sub_trend,
        'spy_price': spy_price, 'spy_sma': spy_sma,
        'adx': adx_val, 'rsi': rsi_val,
        'vix': latest['^VIX'], 'vix3m': latest['^VIX3M'],
        'hyg': latest['HYG'], 'ief': latest['IEF'] # For credit
    }

# --- 4. DASHBOARD RENDER ---
try:
    with st.spinner("Analyzing Market Data..."):
        raw_data, ticker_map = get_data()
        
    m = process_market_regime(raw_data)
    rrg_df = calculate_rrg(raw_data, ticker_map['Sectors'], 'SPY')

    # HEADLINE
    st.title(f"Market Radar: {m['trend']} ({m['sub_trend']})")
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Reference: Yahoo Finance")
    
    # METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    
    # Price vs SMA
    is_bull = m['spy_price'] > m['spy_sma']
    c1.metric("SPY Trend", f"${m['spy_price']:.2f}", 
              f"{'Above' if is_bull else 'Below'} SMA{sma_window}",
              delta_color="normal" if is_bull else "inverse")
    
    # VIX
    vix_state = "High Vol" if m['vix'] > vix_high else "Steady"
    c2.metric("Fear (VIX)", f"{m['vix']:.2f}", vix_state, delta_color="inverse")
    
    # RSI
    rsi_state = "Neutral"
    if m['rsi'] > rsi_overbought: rsi_state = "Overbought"
    if m['rsi'] < rsi_oversold: rsi_state = "Oversold"
    c3.metric("Momentum (RSI)", f"{m['rsi']:.1f}", rsi_state)

    # Term Structure
    if pd.isna(m['vix3m']) or m['vix3m'] == 0:
        c4.metric("Term Structure", "N/A", "No Data", delta_color="off")
    else:
        contango = m['vix'] < m['vix3m']
        c4.metric("VIX Curve", f"{m['vix']/m['vix3m']:.2f}", 
                  "Contango (Healthy)" if contango else "Backwardation (Panic)",
                  delta_color="normal" if contango else "inverse")

    st.markdown("---")

    # TABS FOR DETAIL
    tab_rrg, tab_list, tab_raw = st.tabs(["🔄 Sector Cycles (RRG)", "📋 Leaderboard", "🛠️ Raw Data"])
    
    with tab_rrg:
        st.subheader("Sector Life Cycle: The 4 Quadrants")
        st.info("Top Right = Leading | Bottom Right = Plateaued | Bottom Left = Lagging | Top Left = Up & Coming")
        
        # SCATTER PLOT (RRG APPROXIMATION)
        fig = px.scatter(rrg_df, x="RS_Ratio", y="RS_Momentum", 
                         color="Status", text="Sector",
                         color_discrete_map={
                             "LEADING (Strong)": "green",
                             "WEAKENING (Plateaued)": "orange",
                             "LAGGING (Weak)": "red",
                             "IMPROVING (Up & Coming)": "blue"
                         },
                         title="Relative Rotation: Sector vs. SPY",
                         hover_data=["RS_Ratio", "RS_Momentum"])
        
        # Add Quadrant Lines
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        fig.add_vline(x=100, line_dash="dash", line_color="gray")
        fig.update_traces(textposition='top center', marker_size=12)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_list:
        st.subheader("Sector Status Breakdown")
        
        # Sort so user sees Leaders first, then Up & Comers
        sort_order = ["LEADING (Strong)", "IMPROVING (Up & Coming)", "WEAKENING (Plateaued)", "LAGGING (Weak)"]
        rrg_df['Status'] = pd.Categorical(rrg_df['Status'], categories=sort_order, ordered=True)
        rrg_df = rrg_df.sort_values('Status')
        
        # Display as a clean table with color highlighting
        def color_status(val):
            if "LEADING" in val: return 'background-color: #d4edda; color: green' # Light Green
            if "WEAKENING" in val: return 'background-color: #fff3cd; color: orange' # Light Yellow
            if "LAGGING" in val: return 'background-color: #f8d7da; color: red' # Light Red
            if "IMPROVING" in val: return 'background-color: #d1ecf1; color: blue' # Light Blue
            return ''

        st.dataframe(rrg_df[['Sector', 'Status', 'RS_Ratio', 'RS_Momentum']].style.map(color_status, subset=['Status']), use_container_width=True)

    with tab_raw:
        st.json(m)

except Exception as e:
    st.error(f"Application Error: {e}")
