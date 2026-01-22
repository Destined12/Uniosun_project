import streamlit as st
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

# ===============================
# PAGE CONFIG (FULL SCREEN)
# ===============================
st.set_page_config(
    page_title="Compressor Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# REMOVE DEFAULT STREAMLIT PADDING
# ===============================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        .blink {
            animation: blink-animation 1s steps(2, start) infinite;
        }
        @keyframes blink-animation {
            to { visibility: hidden; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# USERS (hashed passwords)
# ===============================
USERS = {
    os.getenv("ADMIN_USERNAME"): hashlib.sha256(os.getenv("ADMIN_PASSWORD").encode()).hexdigest()
}

# ===============================
# DESCRIPTIONS MAPPING
# ===============================
DESCRIPTION = {
    "COMPRESSOR INLET PRESSURE": "CIP",
    "COMPRESSOR INLET AIR TEMP": "CIT",
    "COMPRESSOR DISCHARGE PRESSURE": "CDP",
    "COMPRESSOR DISCHARGE TEMPERATURE": "CDT",
    "IGV POSITION": "IGV_Position",
    "FILTER DIFFERENTIAL PRESSURE": "Filter_DP",
    "GT SPEED": "GT_Speed",
    "LOAD": "Load",
    "MAXIMUM VIBRATION": "VIB"  # optional
}

DESCRIPTION_NORMALIZED = {k.upper(): v for k, v in DESCRIPTION.items()}


# ===============================
# HELPER FUNCTIONS
# ===============================
def preprocess_df(df):
    """Convert types, fill missing, add derived metrics."""

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    # Ensure numeric
    for col in DESCRIPTION.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived metrics
    if 'CDT' in df.columns and 'CIT' in df.columns:
        df['DELTA_T'] = df['CDT'] - df['CIT']
    if 'CDP' in df.columns and 'CIP' in df.columns:
        df['PRESSURE_RATIO'] = df['CDP'] / df['CIP']

    # Simulated failure probability
    df['FAIL_PROB'] = np.random.rand(len(df))

    # System state based on last FAIL_PROB
    latest_fp = df['FAIL_PROB'].iloc[-1] if not df.empty else 0
    if latest_fp < 0.5:
        state = "Healthy"
    elif 0.5 <= latest_fp < 0.75:
        state = "Warning"
    else:
        state = "Danger"

    return df, latest_fp, state


def status_card(title, value, state):
    if state == "Healthy":
        color = "#22C55E"
        css = ""
    elif state == "Warning":
        color = "#FACC15"
        css = ""
    else:
        color = "#DC2626"
        css = "blink"
    try:
        value_display = f"{float(value):.2f}"
    except (ValueError, TypeError):
        value_display = str(value)
    st.markdown(
        f"""
        <div class="{css}" style="
            background:{color};
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
        ">
            <h4>{title}</h4>
            <h2>{value_display}</h2>
            <p>{state}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def plot(df, y_cols, title, yaxis, rolling_hours=1):
    """Plot time series with rolling average and curve border outline."""
    df_plot = df.copy()

    # Ensure Timestamp is datetime and set as index
    df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
    df_plot.set_index('Timestamp', inplace=True)

    # Apply rolling average if hours > 1
    if rolling_hours > 1:
        df_plot[y_cols] = df_plot[y_cols].rolling(f'{rolling_hours}H').mean()

    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot[col],
            mode="lines+markers",
            name=col,
            line=dict(width=3)  # curve border outline
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title=yaxis,
        height=450,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)


def load_preload_data():
    """Load the CSV data at startup."""
    try:
        df = pd.read_csv('GT1_Compressor_Wide_Format_2025.csv')  # Replace with your CSV path
        df, fp, state = preprocess_df(df)
        return df, fp, state
    except Exception as e:
        st.error(f"Failed to load preload data: {e}")
        return pd.DataFrame(), 0, "Healthy"


def append_new_upload(df_existing, uploaded_file):
    """Append new uploaded data without removing old data."""
    try:
        df_new = pd.read_csv(uploaded_file)
        df_new, _, _ = preprocess_df(df_new)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['Timestamp']).sort_values('Timestamp')
        return df_combined
    except Exception as e:
        st.error(f"Failed to process upload: {e}")
        return df_existing


# ===============================
# SESSION STATE INITIALIZATION
# ===============================
def init_session_state():
    """Initialize all session state variables safely."""
    defaults = {
        "logged_in": False,
        "df": pd.DataFrame(),
        "fp": 0,
        "state": "Healthy",
        "alerts": [],
        "last_upload": "No uploads yet",
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load preloaded CSV only if df is empty
    if st.session_state.df.empty:
        df, fp, state = load_preload_data()
        st.session_state.df = df
        st.session_state.fp = fp
        st.session_state.state = state
        st.session_state.last_upload = "Preload"
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===============================
# LOGIN PAGE
# ===============================
def login():
    st.title("Login")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            pwd_hash = hashlib.sha256(password.encode()).hexdigest()
            if USERS.get(username) == pwd_hash:
                st.session_state.logged_in = True
            else:
                st.error("Invalid username or password")


# ===============================
# DASHBOARD PAGE
# ===============================
def dashboard():
    st.title("Compressor Health Monitoring")
    st.caption("Calabar Power Plant – Gas Turbine Compressor")

    # ===== TOP ACTION BAR =====
    col1, col2, col3 = st.columns([2, 2, 1])

    # ===== FILE UPLOAD =====
    with col1:
        uploaded_file = st.file_uploader("Upload Turbine CSV", type=["csv"])
        if uploaded_file:
            st.session_state.df = append_new_upload(st.session_state.df, uploaded_file)
            st.session_state.last_upload = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Upload successful: {uploaded_file.name}")
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ===== TIME RANGE SELECTION =====
    with col2:
        hours = st.selectbox("Time Range (hours)", [1, 6, 12, 24], index=3)
        if st.button("Update Dashboard"):
            st.success(f"Dashboard updated: displaying {hours}-hour rolling mean")

    # ===== SESSION STATE METRICS =====
    with col3:
        st.metric("Last Upload", st.session_state.last_upload)
        st.metric("Last Update", st.session_state.last_update)

    st.divider()

    # ===== STATUS CARDS =====
    cols = st.columns(4)
    with cols[0]:
        # Dynamic 3D Gauge for Failure Probability
        fp_pct = st.session_state.fp * 100
        if st.session_state.state == "Healthy":
            gauge_color = "green"
        elif st.session_state.state == "Warning":
            gauge_color = "yellow"
        else:
            gauge_color = "red"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fp_pct,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
            },
            title={'text': "Failure Probability (%)"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with cols[1]:
        status_card("Risk Level", "N/A", st.session_state.state)
    with cols[2]:
        status_card("Model Status", "Running", "Healthy")
    with cols[3]:
        status_card("System State", st.session_state.state, st.session_state.state)

    st.divider()

    # ===== ALERTS =====
    if st.session_state.alerts:
        st.subheader("Recent Alerts")
        for alert in st.session_state.alerts[-5:]:
            st.error(alert)

    st.divider()

    # ===== DASHBOARD GRAPHS =====
    df = st.session_state.df
    if not df.empty:
        df_plot = df.copy()
        df_plot.set_index('Timestamp', inplace=True)

        # Apply rolling average if hours > 1
        if hours > 1:
            df_plot = df_plot.rolling(f'{hours}H').mean()

        df_plot_reset = df_plot.reset_index()
        df_plot_reset['Timestamp'] = df_plot_reset['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Plot each metric with curve border outline
        plot(df_plot_reset, ["CIT", "CDT"], "Inlet vs Discharge Temperature", "°C", rolling_hours=hours)
        plot(df_plot_reset, ["CIP", "CDP"], "Inlet vs Discharge Pressure", "psi", rolling_hours=hours)
        if "DELTA_T" in df_plot_reset.columns:
            plot(df_plot_reset, ["DELTA_T"], "Delta Temperature (CDT - CIT)", "°C", rolling_hours=hours)
        if "PRESSURE_RATIO" in df_plot_reset.columns:
            plot(df_plot_reset, ["PRESSURE_RATIO"], "Pressure Ratio (CDP / CIP)", "Ratio", rolling_hours=hours)


# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Initialize all session state variables
    init_session_state()

    if not st.session_state.logged_in:
        login()
    else:
        dashboard()


if __name__ == "__main__":
    main()
