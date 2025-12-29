import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Regime Compass", layout="wide", page_icon="🧭")

# --- 1. SIDEBAR: THRESHOLDS ---
st.sidebar.header("🎛️ Calibration")

with st.sidebar.expander("1. Trend Boundaries (Y-Axis)", expanded=True):
    sma_slow = st.number_input("Bull/Bear Line (SMA)", 50, 365, 200, help="The boundary between Bull and Bear.")
    sma_fast = st.number_input("Conflict Zone (SMA)", 20, 100, 50, help="Used to define the 'Conflict/Chop' middle zone.")

with st.sidebar.expander("2. Value Boundaries (X-Axis)", expanded=True):
    pe_cheap = st.slider("Undervalued P/E", 10, 20, 15)
    pe_expensive = st.slider("Overvalued P/E", 20, 40, 25)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_data_and_fundamentals():
    tickers = ['SPY', '^VIX']
    # Fetch History
    data = yf.download(tickers, period="2y", progress=False)
    
    # Fetch SPY Fundamentals (Earnings)
    # We derive Earnings from Price / PE to reverse engineer the "E"
    spy_ticker = yf.Ticker("SPY")
    try:
        current_pe = spy_ticker.info.get('trailingPE', 24.5) # Default fallback if API fails
        price = data['Close']['SPY'].iloc[-1]
        earnings = price / current_pe
    except:
        current_pe = 25.0
        earnings = 10.0 # Fallback
        
    return data, current_pe, earnings

# --- 3. LOGIC & SCENARIO ENGINE ---
def calculate_regime_state(price, sma200, sma50, pe, pe_cheap, pe_exp):
    # 1. Determine Trend Coordinate (Y)
    # 0 = Bear, 1 = Conflict, 2 = Bull
    if price < sma200 and price < sma50:
        trend_score = 0 # Bear
        trend_name = "BEAR"
    elif price > sma200 and price > sma50:
        trend_score = 2 # Bull
        trend_name = "BULL"
    else:
        trend_score = 1 # Conflict
        trend_name = "CONFLICT"

    # 2. Determine Value Coordinate (X)
    # 0 = Undervalued, 1 = Fair, 2 = Overvalued
    if pe < pe_cheap:
        value_score = 0
        value_name = "UNDERVALUED"
    elif pe > pe_exp:
        value_score = 2
        value_name = "OVERVALUED"
    else:
        value_score = 1
        value_name = "FAIR VALUE"

    # 3. Define the 9 Narratives
    matrix_names = [
        ["Value Trap (Catching Knives)", "Standard Correction", "Bubble Pop (Crash)"],  # Bear Row
        ["Accumulation Zone", "Rotation / Chop", "Distribution Top"],                   # Conflict Row
        ["Generational Buy", "Goldilocks Growth", "Melt-Up (FOMO)"]                     # Bull Row
    ]
    
    current_narrative = matrix_names[trend_score][value_score]
    
    return trend_score, value_score, trend_name, value_name, current_narrative

def calculate_scenarios(price, earnings, sma200, sma50, pe_cheap, pe_exp):
    """
    Calculates distance to nearest boundaries
    """
    scenarios = []
    
    # A. Price Moves (Vertical Shift)
    dist_to_bear = ((sma200 - price) / price) * 100
    if price > sma200:
        scenarios.append(f"📉 **- {abs(dist_to_bear):.1f}% drop** leads to **BEAR** Trend.")
    else:
        scenarios.append(f"📈 **+ {abs(dist_to_bear):.1f}% rally** leads to **BULL** Trend.")

    # B. Valuation Moves (Horizontal Shift)
    # Target Price = Target_PE * Earnings
    target_price_fair = pe_cheap * earnings
    target_price_exp = pe_exp * earnings
    
    dist_to_fair = ((target_price_fair - price) / price) * 100
    dist_to_exp = ((target_price_exp - price) / price) * 100
    
    current_pe = price / earnings
    
    if current_pe > pe_exp:
        scenarios.append(f"📉 **- {abs(dist_to_exp):.1f}% drop** (or earnings growth) needed to reach **FAIR VALUE**.")
    elif current_pe < pe_cheap:
        scenarios.append(f"📈 **+ {abs(dist_to_fair):.1f}% rally** needed to become **FAIR VALUE**.")
        
    return scenarios

# --- 4. VISUALIZATION ENGINE ---
def plot_regime_compass(trend_score, value_score):
    # The Grid Labels
    z = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] # Color mapping
    
    # Custom Colors for the heatmap (Red -> Yellow -> Green logic mixed with Risk)
    # Row 0 (Bear): Blue (Cheap Trap), Red (Correction), Dark Red (Crash)
    # Row 1 (Chop): Cyan (Accum), Gray (Chop), Orange (Dist)
    # Row 2 (Bull): Bright Green (Gen Buy), Green (Goldilocks), Magenta (Melt Up)
    colors = [
        [0.0, "blue"], [0.1, "red"], [0.3, "darkred"],      # Bear Row
        [0.4, "cyan"], [0.5, "gray"], [0.6, "orange"],      # Conflict Row
        [0.7, "lime"], [0.8, "green"], [1.0, "magenta"]     # Bull Row
    ]
    
    labels = [
        ["Value Trap", "Correction", "Bubble Pop"],
        ["Accumulation", "Rotation", "Distribution"],
        ["Gen. Buy", "Goldilocks", "Melt-Up"]
    ]

    fig = go.Figure()

    # 1. The Heatmap Background
    fig.add_trace(go.Heatmap(
        z=[[0.2, 0.1, 0.05], [0.5, 0.4, 0.3], [0.9, 0.8, 0.6]], # Dummy Z for coloring
        x=['Undervalued', 'Fair', 'Overvalued'],
        y=['Bear', 'Conflict', 'Bull'],
        colorscale='RdYlGn', 
        showscale=False,
        opacity=0.6
    ))

    # 2. The Text Labels
    annotations = []
    for y in range(3):
        for x in range(3):
            fig.add_annotation(
                x=x, y=y,
                text=f"<b>{labels[y][x]}</b>",
                showarrow=False,
                font=dict(color="black", size=14)
            )

    # 3. The "You Are Here" Marker
    # We add random jitter to the score so the dot isn't always perfectly centered, 
    # making it look more analog if we had granular data (simulated here for UI)
    fig.add_trace(go.Scatter(
        x=[value_score], y=[trend_score],
        mode='markers+text',
        marker=dict(size=25, color='white', line=dict(width=3, color='black')),
        text=["📍 YOU"], textposition="top center",
        name="Current State"
    ))

    fig.update_layout(
        title="The 9 Narratives Matrix",
        xaxis=dict(title="Valuation", side="bottom"),
        yaxis=dict(title="Trend Strength"),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- 5. MAIN EXECUTION ---
try:
    with st.spinner("Calibrating Compass..."):
        data, current_pe, earnings = get_data_and_fundamentals()
        
        # Process Data
        try:
            closes = data['Close'].ffill()
        except KeyError:
            closes = data.ffill()

        price = closes['SPY'].iloc[-1]
        sma200 = closes['SPY'].rolling(sma_slow).mean().iloc[-1]
        sma50 = closes['SPY'].rolling(sma_fast).mean().iloc[-1]
        
        # Calculate State
        t_score, v_score, t_name, v_name, narrative = calculate_regime_state(
            price, sma200, sma50, current_pe, pe_cheap, pe_expensive
        )
        
        # Calculate Scenarios
        next_moves = calculate_scenarios(price, earnings, sma200, sma50, pe_cheap, pe_expensive)

    # --- UI LAYOUT ---
    st.title(f"📍 Market Status: {narrative}")
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Regime", f"{t_name} + {v_name}", narrative)
    c2.metric("S&P 500 Price", f"${price:.2f}", f"{(price/sma200 - 1)*100:.1f}% vs SMA{sma_slow}")
    c3.metric("Valuation (P/E)", f"{current_pe:.1f}x", f"Earnings Est: ${earnings:.2f}")
    c4.metric("Risk Level", "HIGH" if v_score==2 or t_score==0 else "MODERATE")

    st.markdown("---")

    # The Core Matrix & Forecast
    col_map, col_logic = st.columns([2, 1])

    with col_map:
        st.subheader("🗺️ The Strategic Map")
        fig = plot_regime_compass(t_score, v_score)
        st.plotly_chart(fig, use_container_width=True)

    with col_logic:
        st.subheader("🔮 Forecast & Next Moves")
        st.info(f"We are currently in the **{narrative}** zone.")
        
        st.markdown("### How we leave this zone:")
        for move in next_moves:
            st.markdown(move)
            
        st.markdown("---")
        st.markdown("### Strategic Directive:")
        if narrative == "Melt-Up (FOMO)":
            st.warning("⚠️ **Strategy:** Participate, but tighten stops. Do not add new leverage. Look for exit liquidity.")
        elif narrative == "Goldilocks Growth":
            st.success("✅ **Strategy:** Buy Aggressively. Fundamentals and Technicals agree.")
        elif narrative == "Value Trap (Catching Knives)":
            st.error("🛑 **Strategy:** Do NOT Buy. Wait for Trend to turn Positive (Price > SMA).")
        elif narrative == "Distribution Top":
            st.warning("⚠️ **Strategy:** Reduce Position Size. Smart money is selling.")
        else:
            st.info("ℹ️ **Strategy:** Stick to your system. No extreme signals present.")

except Exception as e:
    st.error(f"System Error: {e}")
    st.write("Debug Info:", e)
