import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- Page Configuration (Wide layout essential for dashboards) ---
st.set_page_config(
    page_title="Real Estate Asset Exchange", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Define Raw Data Loader for Analytics (Cached for performance) ---
@st.cache_data
def load_raw_data():
    try:
        # Load the raw dataset to perform EDA and generate historical graphs
        df = pd.read_csv('train.csv')
        
        # --- Time-Series Data Engineering ---
        # Ames dataset YrSold ranges from 2006-2010
        # Combine Year Sold and Month Sold into a single continuous datetime object
        df['DateSold'] = pd.to_datetime(dict(year=df.YrSold, month=df.MoSold, day=1))
        return df
    except FileNotFoundError:
        return None

# Load the raw data for EDA
raw_df = load_raw_data()

# --- Custom CSS for the Financial Aesthetic ---
st.markdown("""
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 1rem;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #00FF00;
        font-family: 'Courier New', monospace;
    }
    .metric-container {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 10px;
    }
    .live-indicator {
        color: #ff4b4b;
        font-weight: bold;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:18px;
    font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Global Component: Header Section ---
col_title, col_time = st.columns([7, 2])
with col_title:
    st.title("🏛️ Real Estate Spot Market & Analytics")
with col_time:
    st.markdown(f"<p style='text-align: right; padding-top: 25px;'><span class='live-indicator'>● ANALYTICS ACTIVE</span> {datetime.now().strftime('%Y-%m-%d UTC')}</p>", unsafe_allow_html=True)


# --- ORGANIZE APP USING TABS ---
tab_predict, tab_analytics = st.tabs(["🏦 Valuation Terminal", "📊 Market Intelligence & Historical Trends"])

# ==============================================================================
# TAB 1: VALUATION TERMINAL (Existing Prediction Logic)
# ==============================================================================
with tab_predict:
    # --- Load Model safely ---
    @st.cache_resource
    def load_model():
        try:
            with open('house_model.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    model = load_model()
    BASELINE_PRICE = 163000

    if model is None:
        st.error("SYSTEM HALT: 'house_model.pkl' not found. Please execute the training protocol.")
    else:
        # --- Current Market Snapshot ---
        st.markdown("### Regional Indices")
        idx1, idx2, idx3, idx4 = st.columns(4)
        idx1.metric("Tier 1 Quality Index", "$240,500", "+1.2% ▲")
        idx2.metric("Tier 2 Quality Index", "$165,200", "-0.4% ▼")
        idx3.metric("New Construction (2020+)*", "$310,000", "+2.5% ▲")
        idx4.metric("Avg Price/SqFt", "$108.50", "+0.9% ▲")
        st.caption("*New construction index simulated based on historical trends.")
        
        st.markdown("---")

        # --- The "Trading Desk" ---
        st.markdown("### 💻 Spot Valuation Terminal")
        st.markdown("Input asset parameters to generate a live algorithmic valuation.")
        
        # Perfect Symmetrical Alignment
        qual = st.slider("Material & Finish Quality (1-10)", min_value=1, max_value=10, value=6)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_area, col_year, col_bath, col_garage = st.columns(4)
        with col_area:
            area = st.number_input("Gross Area (SqFt)", min_value=500, max_value=10000, value=1500, step=50)
        with col_year:
            year = st.number_input("Construction Year", min_value=1850, max_value=2024, value=2010, step=1)
        with col_bath:
            bath = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=1)
        with col_garage:
            garage = st.selectbox("Garage (Vehicles)", options=[0, 1, 2, 3, 4], index=2)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        calculate_btn = st.button("REQUEST SPOT QUOTE", type="primary", use_container_width=True)

        st.markdown("---")

        # --- Prediction Output ---
        if calculate_btn:
            # Revert to linear transformation in application logic to handle log transformation in training
            user_input = np.array([[qual, area, garage, bath, year]])
            log_prediction = model.predict(user_input)
            final_price = np.exp(log_prediction)[0]
            price_delta = final_price - BASELINE_PRICE
            
            st.markdown("### 🔔 Spot Quote Executed")
            
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.metric(
                    label="LIVE SPOT VALUATION (USD)", 
                    value=f"${final_price:,.2f}", 
                    delta=f"{price_delta:+,.2f} vs Market Median"
                )
            with res_col2:
                st.success("STATUS: QUOTE VALID")
                st.caption(f"**Execution Algorithm:** Multi-variate Linear Regression")
                st.caption(f"**Timestamp:** {datetime.now().strftime('%H:%M:%S UTC')}")


# ==============================================================================
# TAB 2: MARKET INTELLIGENCE (NEW COMPREHENSIVE GRAPHS SECTION)
# ==============================================================================
with tab_analytics:
    st.markdown("## Comprehensive Market Performance Analysis")
    
    if raw_df is None:
        st.error("'train.csv' not found. Cannot load historical graphs.")
    else:
        # Define shared dark theme for all Plotly graphs to match Bullion aesthetic
        plotly_layout_settings = dict(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#262730', zeroline=False),
            font=dict(color='#FAFAFA', family="Courier New, monospace")
        )

        # --------------------------------------------------------------------------
        # GRAPH 1: Historical Price Trend (Line Graph - requested)
        # --------------------------------------------------------------------------
        st.markdown("### 1. Historical Asset Exchange Rate Index (Time Series)")
        st.markdown("Analyzing average clearing prices on a month-by-month basis between 2006 and 2010.")
        
        # Prepare Time Series Data (Group by month and calculate average price)
        time_series_data = raw_df.groupby('DateSold')['SalePrice'].mean().reset_index()
        
        fig_line = px.line(
            time_series_data, 
            x='DateSold', 
            y='SalePrice',
            markers=True,
            title='Real Estate Clearing Prices (Monthly Moving Average)'
        )
        # Apply style updates: Green lines/markers to match bullion ticker
        fig_line.update_traces(line_color='#00FF00', marker=dict(color='#00FF00', size=6))
        fig_line.update_layout(**plotly_layout_settings)
        fig_line.update_layout(yaxis_title="Average Clearing Price (USD)")
        
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------------------------------
        # GRAPH 2: Area Correlation (Scatter Plot)
        # --------------------------------------------------------------------------
        st.markdown("### 2. Physical Footprint vs Valuation (Area Correlation)")
        
        fig_scatter = px.scatter(
            raw_df, 
            x='GrLivArea', 
            y='SalePrice', 
            color='OverallQual',
            trendline="ols", # Add ordinary least squares regression line
            hover_data=['YearBuilt'],
            title='Price Correlation with Gross Living Area (Colored by Quality)'
        )
        fig_scatter.update_layout(**plotly_layout_settings)
        fig_scatter.update_layout(xaxis_title="Gross Area (SqFt)", yaxis_title="Clearing Price (USD)")
        
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --------------------------------------------------------------------------
        # OTHER GRAPH TYPES (Histogram & Box Plot)
        # --------------------------------------------------------------------------
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("### 3. Market Valuation Distribution")
            fig_hist = px.histogram(
                raw_df, 
                x='SalePrice', 
                nbins=50, 
                title='Market Volume by Price Tiers',
                color_discrete_sequence=['#00FF00']
            )
            fig_hist.update_layout(**plotly_layout_settings)
            fig_hist.update_layout(xaxis_title="Price Tier (USD)", yaxis_title="Volume of Transactions")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with row2_col2:
            st.markdown("### 4. Quality Grade Impact (Box Plot)")
            fig_box = px.box(
                raw_df, 
                x='OverallQual', 
                y='SalePrice', 
                title='Valuation Distribution by Asset Quality Grade',
                color='OverallQual'
            )
            fig_box.update_layout(**plotly_layout_settings)
            fig_box.update_layout(xaxis_title="Asset Quality Grade (1-10)", yaxis_title="Price Range (USD)")
            st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# GLOBAL COMPONENT: SIMULATED ORDER BOOK (Kept at bottom)
# ==============================================================================
st.markdown("---")
st.markdown("### 📋 Simulated Order Book (Cleared Transactions)")

recent_data = pd.DataFrame({
    "Asset ID": ["AM-2101", "AM-1903", "AM-2022", "AM-1877"],
    "Cleared Time": ["10:21:44", "10:20:12", "10:18:05", "10:15:33"],
    "Quality": ["Tier 8", "Tier 5", "Tier 7", "Tier 6"],
    "SqFt": [2200, 1100, 1850, 1400],
    "Price Cleared": ["$285,000", "$130,000", "$210,000", "$165,000"],
    "Market Action": ["BUY", "SELL", "BUY", "HOLD"]
})
# Stylized dataframe output
st.dataframe(recent_data, use_container_width=True, hide_index=True)