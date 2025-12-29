import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Command Center", layout="wide", page_icon="📡")

# --- 1. SIDEBAR: CONTROLS ---
st.sidebar.header("🎛️ Control Panel")

with st.sidebar.expander("1. Trend Thresholds", expanded=True):
    sma_slow = st.number_input("Bull/Bear Line (SMA)", 50, 365, 200)
    sma_fast = st.number_input("Conflict Zone (SMA)", 20, 100, 50)

with st.sidebar.expander("2. Valuation Thresholds", expanded=True):
    pe_cheap = st.slider("Undervalued P/E", 10, 20, 15)
    pe_expensive = st.slider("Overvalued P/E", 20, 40, 25)

with st.sidebar.expander("3. Sector Rotation", expanded=True):
    cycle_window = st.slider("Trend Lookback (Days)", 10, 100, 20)
    momentum_window = st.slider("Momentum Lookback (Days)", 3, 30, 10)

# --- 2. DATA ENGINE (ROBUST) ---
@st.cache_data(ttl=3600)
def get_market_data():
    # Define Universe
    tickers = {
        'Index': ['SPY', '^VIX'],
        'Sectors': ['XLK', 'XLE', 'XLF', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU']
    }
    all_syms = tickers['Index'] + tickers['Sectors']
    
    # 1. Download Data
    # auto_adjust=True helps with splits/dividends
    data = yf.download(all_syms, period="2y", group_by='ticker', auto_adjust=True, progress=False)
    
    if data.empty:
        return pd.DataFrame(), {}, tickers

    # 2. FLATTEN & CLEAN
    df_close = pd.DataFrame()
    
    # Iterate through symbols and safely extract 'Close'
    for sym in all_syms:
        try:
            # Handle MultiIndex (Ticker -> Close) vs Single Index
            if isinstance(data.columns, pd.MultiIndex):
                if sym in data.columns.levels[0]:
                    df_close[sym] = data[sym]['Close']
            else:
                # Fallback for single ticker download scenarios
                if 'Close' in data.columns:
                    df_close[sym] = data['Close']
        except Exception:
            continue # Skip bad tickers, don't crash
            
    # CRITICAL FIX 1: Forward Fill (Fixes Weekends/Holidays)
    df_close = df_close.ffill()
    
    # CRITICAL FIX 2: Do NOT dropna(). 
    # Just drop rows where SPY specifically is missing, as SPY is the engine.
    if 'SPY' in df_close.columns:
        df_close = df_close.dropna(subset=['SPY'])
    
    # 3. Fetch Fundamentals (P/E)
    fundamentals = {}
    try:
        spy_ticker = yf.Ticker("SPY")
        # specific check to avoid yfinance errors
        funds = spy_ticker.info
        fundamentals['SPY'] = funds.get('trailingPE', 25.0)
    except:
        fundamentals['SPY'] = 25.0
        
    return df_close, fundamentals, tickers

# --- 3. LOGIC ENGINES ---

def get_regime_narrative(price, sma200, sma50, pe, pe_cheap, pe_exp):
    # Trend Coordinate (Y): 0=Bear, 1=Conflict, 2=Bull
    if price < sma200 and price < sma50: t_score = 0
    elif price > sma200 and price > sma50: t_score = 2
    else: t_score = 1
    
    # Value Coordinate (X): 0=Cheap, 1=Fair, 2=Expensive
    if pe < pe_cheap: v_score = 0
    elif pe > pe_exp: v_score = 2
    else: v_score = 1
    
    matrix = [
        ["Value Trap", "Correction", "Bubble Pop"],    # Bear
        ["Accumulation", "Rotation", "Distribution"],  # Conflict
        ["Gen. Buy", "Goldilocks", "Melt-Up"]          # Bull
    ]
    return t_score, v_score, matrix[t_score][v_score]

def calculate_rrg(df_close, sectors, benchmark='SPY'):
    results = []
    
    if benchmark not in df_close.columns:
        return pd.DataFrame() # Fail gracefully if SPY missing

    bench = df_close[benchmark]
    
    for sec in sectors:
        if sec not in df_close.columns:
            continue
            
        # Relative Strength
        rs = df_close[sec] / bench
        
        # RS-Ratio (Trend) - Normalize around 100
        rs_mean = rs.rolling(window=cycle_window).mean()
        rs_ratio = 100 + ((rs - rs_mean) / rs_mean) * 100
        
        # RS-Momentum (Rate of Change of Trend)
        rs_mom = 100 + rs_ratio.diff(momentum_window)
        
        # Current State
        try:
            curr_r = rs_ratio.iloc[-1]
            curr_m = rs_mom.iloc[-1]
            
            # Handle NaNs in calculation (e.g. recent IPOs)
            if pd.isna(curr_r) or pd.isna(curr_m):
                continue
                
            if curr_r > 100 and curr_m > 100: status = "LEADING"
            elif curr_r > 100 and curr_m < 100: status = "WEAKENING"
            elif curr_r < 100 and curr_m < 100: status = "LAGGING"
            else: status = "IMPROVING"
                
            results.append({
                'Sector': sec,
                'RS_Ratio': curr_r,
                'RS_Momentum': curr_m,
                'Status': status
            })
        except:
            continue
    
    return pd.DataFrame(results)

def plot_compass(t_score, v_score):
    labels = [
        ["Value Trap", "Correction", "Bubble Pop"],
        ["Accumulation", "Rotation", "Distribution"],
        ["Gen. Buy", "Goldilocks", "Melt-Up"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        x=['Cheap', 'Fair', 'Expensive'],
        y=['Bear', 'Conflict', 'Bull'],
        colorscale='RdYlGn', opacity=0.6, showscale=False
    ))
    
    # Add Text Labels
    for y in range(3):
        for x in range(3):
            fig.add_annotation(x=x, y=y, text=f"<b>{labels[y][x]}</b>", showarrow=False)
            
    # Add "YOU" Marker
    fig.add_trace(go.Scatter(
        x=[v_score], y=[t_score], mode='markers+text',
        marker=dict(size=30, color='white', line=dict(width=3, color='black')),
        text=["📍 YOU"], textposition="top center"
    ))
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    return fig

# --- 4. EXECUTION ---
try:
    with st.spinner("Analyzing Market Structure..."):
        df, funds, tickers = get_market_data()
        
    # --- SAFETY CHECK ---
    if df.empty or 'SPY' not in df.columns:
        st.error("⚠️ Data Download Failed. Unable to retrieve SPY data from Yahoo Finance.")
        st.info("Check your internet connection or try again in 5 minutes (API Rate Limit).")
        st.stop()

    # --- MACRO ANALYSIS (SPY) ---
    spy = df['SPY']
    current_price = spy.iloc[-1]
    
    # Calculate SMAs (Check for sufficiency data)
    if len(spy) < sma_slow:
        st.warning(f"Not enough data history to calculate {sma_slow}-day SMA. Showing available data.")
        sma200 = spy.mean() # Fallback
        sma50 = spy.mean()
    else:
        sma200 = spy.rolling(sma_slow).mean().iloc[-1]
        sma50 = spy.rolling(sma_fast).mean().iloc[-1]
        
    current_pe = funds.get('SPY', 25.0)
    
    t_score, v_score, narrative = get_regime_narrative(
        current_price, sma200, sma50, current_pe, pe_cheap, pe_expensive
    )
    
    # --- SECTOR ANALYSIS (RRG) ---
    rrg_df = calculate_rrg(df, tickers['Sectors'])

    # --- DASHBOARD RENDER ---
    st.title(f"📡 Market Status: {narrative}")
    
    # TOP ROW: METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Trend", "BULL" if t_score==2 else "BEAR" if t_score==0 else "CONFLICT", 
              f"Price ${current_price:.0f}")
    m2.metric("Valuation (P/E)", f"{current_pe:.1f}x", 
              "Expensive" if v_score==2 else "Cheap" if v_score==0 else "Fair",
              delta_color="inverse")
    
    # VIX Handling
    if '^VIX' in df.columns:
        vix = df['^VIX'].iloc[-1]
        m3.metric("Volatility (VIX)", f"{vix:.2f}", "High Risk" if vix > 20 else "Stable", delta_color="inverse")
    else:
        m3.metric("Volatility (VIX)", "N/A", "Data Missing", delta_color="off")
    
    # Forecast / Distance
    dist_bear = (current_price - sma200) / current_price
    m4.metric("Safety Cushion", f"{dist_bear:.1%}", "Distance to Bear Market")

    st.markdown("---")

    # MIDDLE ROW: COMPASS & SECTORS
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("🗺️ Regime Compass")
        st.caption("Where are we in the Big Cycle?")
        fig_compass = plot_compass(t_score, v_score)
        st.plotly_chart(fig_compass, use_container_width=True)
        
        st.info(f"**Insight:** The market is currently **{narrative}**. Ensure your position sizing matches this environment.")

    with c_right:
        st.subheader("🔄 Sector Rotation (RRG)")
        st.caption("Which engines are firing? (Top Right = Leaders)")
        
        if not rrg_df.empty:
            # Static Quadrant Background
            fig_rrg = px.scatter(rrg_df, x="RS_Ratio", y="RS_Momentum", 
                                 color="Status", text="Sector",
                                 color_discrete_map={
                                     "LEADING": "green", "WEAKENING": "orange",
                                     "LAGGING": "red", "IMPROVING": "blue"
                                 },
                                 hover_data=["RS_Ratio", "RS_Momentum"])
            
            fig_rrg.add_hline(y=100, line_color="gray", line_dash="dash")
            fig_rrg.add_vline(x=100, line_color="gray", line_dash="dash")
            fig_rrg.update_traces(textposition='top center', marker_size=12)
            fig_rrg.update_layout(height=450, xaxis_title="Relative Trend", yaxis_title="Relative Momentum")
            st.plotly_chart(fig_rrg, use_container_width=True)
        else:
            st.warning("Not enough sector data to generate RRG.")

    # BOTTOM ROW: DETAILS
    with st.expander("📊 View Raw Sector Data"):
        if not rrg_df.empty:
            try:
                st.dataframe(rrg_df.sort_values("RS_Ratio", ascending=False).style.background_gradient(cmap="Greens", subset=["RS_Ratio"]), use_container_width=True)
            except Exception:
                # Fallback if matplotlib fails or other styling issue
                st.dataframe(rrg_df.sort_values("RS_Ratio", ascending=False), use_container_width=True)
        else:
            st.write("No data available.")

except Exception as e:
    st.error(f"Critical System Error: {e}")
    # Print simple trace for debugging
    import traceback
    st.text(traceback.format_exc())
