import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Command Center", layout="wide", page_icon="📡")

# --- 1. SIDEBAR: STRATEGY CONTROLS ---
st.sidebar.header("🎛️ Strategy Inputs")

with st.sidebar.expander("1. Technical Thresholds", expanded=True):
    sma_fast = st.number_input("Fast SMA (Trend)", value=50, help="Medium-term trend line.")
    sma_slow = st.number_input("Slow SMA (Regime)", value=200, help="Long-term 'Line in the Sand'.")
    adx_threshold = st.slider("Trend Strength (ADX)", 10, 50, 25)

with st.sidebar.expander("2. Fundamental Thresholds", expanded=True):
    # Historical S&P 500 average P/E is roughly 15-20.
    pe_cheap = st.slider("P/E Undervalued", 10, 20, 15, help="Below this = Cheap/Value Zone.")
    pe_expensive = st.slider("P/E Overvalued", 20, 40, 25, help="Above this = Expensive/Bubble Zone.")

with st.sidebar.expander("3. Cycle Sensitivity"):
    cycle_window = st.slider("RRG Trend Lookback", 10, 100, 20)
    momentum_window = st.slider("RRG Momentum Lookback", 3, 30, 10)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600) # Cache for 1 hour since fundamentals don't change often
def get_data():
    tickers = {
        'Benchmarks': ['SPY', 'QQQ', 'IWM', 'GLD', 'BTC-USD', 'TLT'],
        'Sectors': ['XLK', 'XLE', 'XLF', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU'],
        'Indicators': ['^VIX', '^VIX3M']
    }
    all_symbols = [item for sublist in tickers.values() for item in sublist]
    
    # 1. Fetch Price History
    history = yf.download(all_symbols, period="2y", progress=False)
    
    # 2. Fetch Fundamentals (P/E Ratios)
    # Note: Fetching TTM PE for ETFs can be tricky via free API. 
    # We use a known proxy method: retrieving 'trailingPE' from yf.Ticker info.
    # This is slower, so we cache it.
    fundamentals = {}
    for sym in tickers['Benchmarks'] + tickers['Sectors']:
        try:
            # We skip currencies/commodities for P/E
            if sym in ['GLD', 'BTC-USD', 'TLT', '^VIX', '^VIX3M']:
                fundamentals[sym] = np.nan
            else:
                info = yf.Ticker(sym).info
                fundamentals[sym] = info.get('trailingPE', np.nan)
        except:
            fundamentals[sym] = np.nan
            
    return history, fundamentals, tickers

# --- 3. LOGIC ENGINE ---
def analyze_market(history, fundamentals):
    # Fix Data
    try:
        closes = history['Close'].ffill()
        highs = history['High'].ffill()
        lows = history['Low'].ffill()
    except KeyError:
        closes = history.ffill()
        highs = closes
        lows = closes

    latest = closes.iloc[-1]
    
    # --- A. TECHNICAL DEEP DIVE ---
    spy = closes['SPY']
    val_fast = spy.rolling(sma_fast).mean().iloc[-1]
    val_slow = spy.rolling(sma_slow).mean().iloc[-1]
    curr_price = spy.iloc[-1]
    
    # 1. Crossover Logic
    # Golden Cross: Fast > Slow. Death Cross: Fast < Slow.
    alignment = "BULLISH STACK" if val_fast > val_slow else "BEARISH STACK"
    
    # 2. Location Logic
    if curr_price > val_fast and curr_price > val_slow:
        loc = "Full Bull"
    elif curr_price < val_fast and curr_price < val_slow:
        loc = "Full Bear"
    else:
        loc = "Conflict/Correction"
        
    # 3. ADX
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
        sum_di = sum_di.replace(0, 0.0001)
        dx = 100 * (abs(plus_di - minus_di) / sum_di)
        return dx.rolling(lookback).mean().iloc[-1]

    adx = calc_adx(highs['SPY'], lows['SPY'], closes['SPY'])
    
    # --- B. FUNDAMENTAL DEEP DIVE ---
    spy_pe = fundamentals.get('SPY', np.nan)
    
    if pd.isna(spy_pe):
        valuation = "Unknown"
    elif spy_pe < pe_cheap:
        valuation = "Undervalued (Value)"
    elif spy_pe > pe_expensive:
        valuation = "Overvalued (Expensive)"
    else:
        valuation = "Fair Value"
        
    # --- C. FUSION SIGNAL ---
    # Combine Tech + Fund
    if "Bull" in loc and "Undervalued" in valuation:
        fusion = "💎 GENERATIONAL BUY (Cheap + Rising)"
        fusion_color = "green"
    elif "Bull" in loc and "Overvalued" in valuation:
        fusion = "🔥 MELT UP (Expensive + Rising)"
        fusion_color = "orange"
    elif "Bear" in loc and "Overvalued" in valuation:
        fusion = "💣 BUBBLE POP (Expensive + Falling)"
        fusion_color = "red"
    elif "Bear" in loc and "Undervalued" in valuation:
        fusion = "🛒 VALUE TRAP (Cheap + Falling)"
        fusion_color = "blue"
    else:
        fusion = "MIXED / HOLD"
        fusion_color = "gray"

    return {
        'price': curr_price,
        'sma_fast': val_fast,
        'sma_slow': val_slow,
        'alignment': alignment,
        'location': loc,
        'adx': adx,
        'pe': spy_pe,
        'valuation': valuation,
        'fusion': fusion,
        'fusion_color': fusion_color,
        'vix': latest['^VIX'],
        'vix3m': latest['^VIX3M']
    }

def calculate_rrg(df, fundamentals, sectors, benchmark='SPY'):
    # Prepare RRG Data
    results = []
    bench_close = df['Close'][benchmark].ffill()
    
    for sector in sectors:
        sec_close = df['Close'][sector].ffill()
        rs = sec_close / bench_close
        rs_mean = rs.rolling(window=cycle_window).mean()
        rs_ratio = 100 + ((rs - rs_mean) / rs_mean) * 100
        rs_mom = 100 + rs_ratio.diff(momentum_window)
        
        # Get PE
        pe = fundamentals.get(sector, 0)
        
        results.append({
            'Sector': sector,
            'RS_Ratio': rs_ratio.iloc[-1],
            'RS_Momentum': rs_mom.iloc[-1],
            'PE': pe if not pd.isna(pe) else 0, # Size bubble by PE
            'Status': "LEADING" if rs_ratio.iloc[-1] > 100 and rs_mom.iloc[-1] > 100 else "LAGGING"
        })
        
    # Add Benchmark
    results.append({'Sector': 'SPY', 'RS_Ratio': 100, 'RS_Momentum': 100, 'PE': fundamentals.get('SPY', 20), 'Status': 'BENCH'})
    return pd.DataFrame(results)

# --- 4. EXECUTION ---
try:
    with st.spinner("Crunching Numbers (Price + Fundamentals)..."):
        history, fundamentals, tickers = get_data()
    
    sig = analyze_market(history, fundamentals)
    rrg_df = calculate_rrg(history, fundamentals, tickers['Sectors'])

    # --- TITLE SECTION ---
    st.title("📡 Market Command Center")
    st.markdown(f"### Overall Signal: :{sig['fusion_color']}[{sig['fusion']}]")
    
    # --- ROW 1: THE METRICS BOARD ---
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        # Technicals
        delta = sig['price'] - sig['sma_slow']
        st.metric("Price vs Slow SMA", 
                  f"${sig['price']:.2f}",
                  f"{delta:.2f} ({(delta/sig['sma_slow'])*100:.1f}%)",
                  help=f"Price: ${sig['price']:.2f}\nSMA{int(sma_slow)}: ${sig['sma_slow']:.2f}")
        st.caption(f"Trend Stack: {sig['alignment']}")

    with c2:
        # Technical Nuance
        spread = sig['sma_fast'] - sig['sma_slow']
        st.metric(f"SMA Spread ({int(sma_fast)} vs {int(sma_slow)})", 
                  f"${spread:.2f}",
                  "Widening" if spread > 0 else "Converging",
                  help=f"SMA{int(sma_fast)}: ${sig['sma_fast']:.2f}\nSMA{int(sma_slow)}: ${sig['sma_slow']:.2f}")
        st.caption(f"ADX Strength: {sig['adx']:.1f}")

    with c3:
        # Fundamentals
        st.metric("S&P 500 P/E Ratio", 
                  f"{sig['pe']:.2f}" if not pd.isna(sig['pe']) else "N/A",
                  sig['valuation'],
                  delta_color="off",
                  help=f"Cheap < {pe_cheap} | Expensive > {pe_expensive}")
        st.caption("Trailing 12-Month Earnings")

    with c4:
        # Risk / Fear
        vix_ratio = sig['vix'] / sig['vix3m'] if sig['vix3m'] else 1
        st.metric("Volatility Regime", 
                  f"{sig['vix']:.2f}", 
                  "Backwardation (Fear)" if vix_ratio > 1 else "Contango (Calm)",
                  delta_color="inverse")
        st.caption(f"Term Ratio: {vix_ratio:.2f}")

    st.markdown("---")

    # --- ROW 2: FUSION INSIGHTS (Tech + Fund) ---
    t1, t2 = st.tabs(["🧩 Sector Valuation Matrix", "🔄 Strategic Rotation"])
    
    with t1:
        st.subheader("Where is the Value?")
        st.info("X-Axis: Relative Strength (Momentum) | Y-Axis: P/E Ratio (Value). ideal = Bottom Right (Strong + Cheap).")
        
        # Scatter: Momentum (X) vs Value (Y)
        # Note: We invert Y axis usually because Lower PE is better
        fig_val = px.scatter(rrg_df, x="RS_Ratio", y="PE", 
                             text="Sector", size="PE", color="RS_Ratio",
                             color_continuous_scale="RdYlGn",
                             title="Momentum vs. Valuation")
        
        # Add Lines
        fig_val.add_hline(y=pe_expensive, line_dash="dash", line_color="red", annotation_text="Expensive")
        fig_val.add_hline(y=pe_cheap, line_dash="dash", line_color="green", annotation_text="Cheap")
        fig_val.add_vline(x=100, line_dash="solid", line_color="gray")
        
        st.plotly_chart(fig_val, use_container_width=True)
        
    with t2:
        st.subheader("Sector Rotation Cycle")
        st.info("Size of bubble = P/E Ratio (Bigger = More Expensive).")
        
        fig_rrg = px.scatter(rrg_df, x="RS_Ratio", y="RS_Momentum", 
                             text="Sector", size="PE", color="Status",
                             color_discrete_map={'LEADING': 'green', 'LAGGING': 'red', 'BENCH': 'black'},
                             title="Relative Rotation Graph (Sized by P/E)")
        
        fig_rrg.add_hline(y=100, line_color="gray")
        fig_rrg.add_vline(x=100, line_color="gray")
        fig_rrg.update_layout(height=600)
        
        st.plotly_chart(fig_rrg, use_container_width=True)

except Exception as e:
    st.error(f"Data Error: {e}")
    st.warning("Note: Fundamental data (P/E) relies on Yahoo Finance headers which can sometimes be blocked or slow.")
