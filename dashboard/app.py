import os
import sys
import pandas as pd
import streamlit as st

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_forecast
from core.data import fetch_stock_data

# Set Page Config (Centered layout matches GOV.UK 960px grid)
st.set_page_config(
    page_title="Chronos Forecasting Service - GOV.UK",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Complete GOV.UK Frontend Design System Override
st.markdown("""
<style>
    /* 1. Global Page Resets */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: "GDS Transport", Arial, sans-serif !important;
        background-color: #f3f2f1 !important; /* GDS light grey background */
        color: #0b0c0c !important; /* GDS text black */
    }
    
    /* Center and constrain main content to GOV.UK 960px grid */
    .main .block-container {
        max-width: 960px !important;
        padding: 0px 30px 50px 30px !important;
        background-color: #ffffff !important; /* Main page is white */
        min-height: 100vh;
        border-left: 1px solid #b1b4b6;
        border-right: 1px solid #b1b4b6;
        box-shadow: none !important;
    }

    /* Hide Streamlit elements */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        border-bottom: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }
    
    /* 2. Official GOV.UK Header Bar */
    .govuk-header {
        background-color: #0b0c0c;
        padding: 10px 0;
        border-bottom: 10px solid #1d70b8; /* GDS Blue */
        margin-left: -30px;
        margin-right: -30px;
        margin-top: -60px;
        padding-left: 30px;
        padding-right: 30px;
    }
    .govuk-header-logo {
        display: flex;
        align-items: center;
        text-decoration: none;
    }
    .govuk-header-logo-text {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        font-family: Arial, sans-serif !important;
    }
    .govuk-crown-logo {
        fill: #ffffff;
        margin-right: 10px;
        width: 30px;
        height: 30px;
    }
    
    /* Phase Banner */
    .govuk-phase-banner {
        padding: 8px 0;
        border-bottom: 1px solid #b1b4b6;
        margin-bottom: 25px;
        font-size: 16px;
    }
    .govuk-phase-tag {
        background-color: #1d70b8;
        color: #ffffff;
        font-weight: 700;
        font-size: 12px;
        padding: 2px 8px;
        margin-right: 10px;
        text-transform: uppercase;
    }
    
    /* Back Link component */
    .govuk-back-link {
        font-size: 16px;
        color: #0b0c0c !important;
        text-decoration: underline;
        margin-bottom: 30px;
        display: inline-block;
        cursor: pointer;
    }

    /* 3. Typography & Form Controls */
    h1.govuk-heading-xl {
        font-size: 48px !important;
        font-weight: 700 !important;
        margin-top: 20px !important;
        margin-bottom: 30px !important;
        color: #0b0c0c !important;
    }
    h2.govuk-heading-l {
        font-size: 36px !important;
        font-weight: 700 !important;
        border-bottom: 4px solid #0b0c0c !important;
        padding-bottom: 10px !important;
        margin-top: 40px !important;
        margin-bottom: 20px !important;
        color: #0b0c0c !important;
    }
    
    /* Inputs */
    .govuk-form-group {
        margin-bottom: 30px;
    }
    .govuk-label {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #0b0c0c !important;
        margin-bottom: 5px !important;
        display: block;
    }
    .govuk-hint {
        font-size: 16px !important;
        color: #505a5f !important;
        margin-bottom: 15px !important;
        display: block;
    }
    
    /* Streamlit Text Input Override */
    div[data-testid="stTextInput"] input {
        border: 2px solid #0b0c0c !important;
        border-radius: 0px !important;
        font-size: 19px !important;
        padding: 8px 12px !important;
        height: auto !important;
    }
    div[data-testid="stTextInput"] input:focus {
        outline: 3px solid #ffdd00 !important; /* Yellow focus ring */
        outline-offset: 0px !important;
        border: 2px solid #0b0c0c !important;
    }

    /* GDS Green Primary Button */
    div.stButton > button {
        background-color: #00703c !important; /* GDS Green */
        color: #ffffff !important;
        font-size: 19px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        border: none !important;
        border-radius: 0px !important;
        border-bottom: 3px solid #005a30 !important;
        box-shadow: none !important;
        width: auto !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        background-color: #005a30 !important;
    }
    div.stButton > button:focus {
        background-color: #ffdd00 !important;
        color: #0b0c0c !important;
        border: 3px solid #0b0c0c !important;
        outline: none !important;
    }

    /* 4. GDS Warning Text Callout */
    .govuk-warning-text {
        display: flex;
        align-items: flex-start;
        margin: 20px 0;
    }
    .govuk-warning-icon {
        background-color: #0b0c0c;
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        flex-shrink: 0;
    }
    .govuk-warning-text-content {
        font-size: 19px;
        font-weight: 700;
    }

    /* 5. GDS Inset Text */
    .govuk-inset-text {
        border-left: 10px solid #b1b4b6 !important;
        background-color: #ffffff !important;
        padding: 15px 20px !important;
        margin: 25px 0 !important;
        font-size: 19px !important;
        color: #0b0c0c !important;
        line-height: 1.5 !important;
    }

    /* 6. GDS Metrics Grid (Clean thin borders instead of modern cards) */
    .gds-metric-row {
        display: flex;
        border-top: 1px solid #b1b4b6;
        border-bottom: 1px solid #b1b4b6;
        padding: 20px 0;
        margin: 30px 0;
    }
    .gds-metric-cell {
        flex: 1;
        padding-right: 20px;
    }
    .gds-metric-label {
        font-size: 16px;
        color: #505a5f;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .gds-metric-value {
        font-size: 48px;
        font-weight: 700;
        color: #0b0c0c;
    }
    .gds-metric-change {
        font-size: 19px;
        font-weight: 700;
    }

    /* 7. Clean GDS Tables */
    div[data-testid="stDataFrame"] {
        border: none !important;
        border-radius: 0px !important;
    }
    .govuk-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 19px;
    }
    .govuk-table th {
        border-bottom: 2px solid #0b0c0c;
        text-align: left;
        padding: 10px 0;
        font-weight: 700;
    }
    .govuk-table td {
        border-bottom: 1px solid #b1b4b6;
        padding: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# 1. Header and Logo
st.markdown("""
<div class="govuk-header">
    <a href="#" class="govuk-header-logo">
        <svg class="govuk-crown-logo" viewBox="0 0 100 100">
            <path d="M50 15 L70 35 L60 35 L60 55 L40 55 L40 35 L30 35 Z" />
        </svg>
        <span class="govuk-header-logo-text">GOV.UK</span>
    </a>
</div>
<div class="govuk-phase-banner">
    <span class="govuk-phase-tag">Beta</span>
    This is a new service – your feedback will help us to improve it.
</div>
<a class="govuk-back-link">← Back to services</a>
""", unsafe_allow_html=True)

# 2. Main heading
st.markdown('<h1 class="govuk-heading-xl">Forecast stock and asset prices</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="govuk-inset-text">
    Use this service to calculate price predictions for publicly traded assets. The engine fits a combined mathematical model using historic daily datasets.
</div>
""", unsafe_allow_html=True)

# Form Fields in a structured layout
st.markdown('<div class="govuk-form-group">', unsafe_allow_html=True)
st.markdown('<label class="govuk-label">Ticker symbol</label>', unsafe_allow_html=True)
st.markdown('<span class="govuk-hint">Enter the market identifier (e.g. AAPL for Apple Inc. or BTC-USD for Bitcoin).</span>', unsafe_allow_html=True)
ticker = st.text_input("", value="AAPL", label_visibility="collapsed").upper().strip()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="govuk-form-group">', unsafe_allow_html=True)
st.markdown('<label class="govuk-label">Forecasting horizon (Days)</label>', unsafe_allow_html=True)
st.markdown('<span class="govuk-hint">Select the number of days into the future to project the prices.</span>', unsafe_allow_html=True)
days_to_predict = st.slider("", min_value=5, max_value=90, value=30, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

run_clicked = st.button("Calculate forecast")

# Output Section
if run_clicked or ticker:
    with st.spinner("Calculating forecast model..."):
        try:
            # Data Fetching
            hist_data = fetch_stock_data(ticker, period="1mo")
            current_price = hist_data['y'].iloc[-1]
            
            # Prediction
            forecast_df = get_forecast(ticker, days_to_predict=days_to_predict)
            final_predicted = forecast_df['hybrid_val'].iloc[-1]
            
            price_change = final_predicted - current_price
            pct_change = (price_change / current_price) * 100
            
            change_color = "#00703c" if price_change >= 0 else "#d4351c"
            
            # Render GDS Metric Grid
            st.markdown(f"""
            <div class="gds-metric-row">
                <div class="gds-metric-cell" style="border-right: 1px solid #b1b4b6;">
                    <div class="gds-metric-label">Current Price</div>
                    <div class="gds-metric-value">${current_price:.2f}</div>
                </div>
                <div class="gds-metric-cell" style="border-right: 1px solid #b1b4b6;">
                    <div class="gds-metric-label">Forecast Price ({days_to_predict} Days)</div>
                    <div class="gds-metric-value">${final_predicted:.2f}</div>
                </div>
                <div class="gds-metric-cell">
                    <div class="gds-metric-label">Projected Trend</div>
                    <div class="gds-metric-value" style="color: {change_color};">{pct_change:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # GDS Chart Section
            st.markdown('<h2 class="govuk-heading-l">Forecast chart</h2>', unsafe_allow_html=True)
            
            hist_plot = hist_data.tail(30)[['ds', 'y']].copy()
            hist_plot.columns = ['ds', 'Actual Price']
            
            forecast_plot = forecast_df[['ds', 'hybrid_val', 'prophet_val', 'xgb_val']].copy()
            forecast_plot.columns = ['ds', 'Hybrid Prediction', 'Prophet Component', 'XGBoost Component']
            
            combined_plot = pd.merge(hist_plot, forecast_plot, on='ds', how='outer').set_index('ds')
            
            # Chart using high-contrast GDS colors (Black for actuals, GDS Blue for hybrid predictions)
            st.line_chart(combined_plot, color=["#0b0c0c", "#1d70b8", "#d4351c", "#00703c"])
            
            st.markdown("<p style='font-size: 16px; color: #505a5f; margin-top: 10px; margin-bottom: 40px;'>Chart showing actual values (black) and forecasts: hybrid ensemble model (blue), Prophet component (red), and XGBoost component (green).</p>", unsafe_allow_html=True)
            
            # GDS Table Section
            st.markdown('<h2 class="govuk-heading-l">Forecast breakdown</h2>', unsafe_allow_html=True)
            
            # Display raw forecast table with GDS typography and no vertical borders
            display_df = forecast_df.copy()
            display_df.columns = ['Date', 'Prophet component', 'XGBoost component', 'Hybrid prediction']
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Build GDS HTML Table directly to enforce official look
            table_rows = []
            for _, row in display_df.iterrows():
                table_rows.append(f"""
                <tr>
                    <td>{row['Date']}</td>
                    <td>${row['Prophet component']:.2f}</td>
                    <td>${row['XGBoost component']:.2f}</td>
                    <td><strong>${row['Hybrid prediction']:.2f}</strong></td>
                </tr>
                """)
                
            st.markdown(f"""
            <table class="govuk-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Prophet Baseline</th>
                        <th>XGBoost Momentum Offset</th>
                        <th>Hybrid Forecasted Price</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
            # GDS warning text at the bottom
            st.markdown("""
            <div class="govuk-warning-text">
                <span class="govuk-warning-icon">!</span>
                <span class="govuk-warning-text-content">
                    Warning: Price projections are calculations based on historical values. Past performance is not a guarantee of future outcomes.
                </span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="govuk-warning-text" style="color: #d4351c;">
                <span class="govuk-warning-icon" style="background-color: #d4351c;">!</span>
                <span class="govuk-warning-text-content">
                    Calculation Error: {str(e)}
                </span>
            </div>
            """, unsafe_allow_html=True)
