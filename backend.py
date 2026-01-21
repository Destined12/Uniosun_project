# backend.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DESCRIPTION = {
    "COMPRESSOR INLET PRESSURE": "CIP",
    "COMPRESSOR INLET AIR TEMP": "CIT",
    "COMPRESSOR DISCHARGE PRESSURE": "CDP",
    "COMPRESSOR DISCHARGE TEMPERATURE": "CDT",
    "VIBRATION": "VIB"
}

def extract_and_process(uploaded_file, log_date):
    df_raw = pd.read_csv(uploaded_file, header=2)

    df_raw["DESCRIPTION"] = df_raw["DESCRIPTION"].astype(str).str.strip().str.upper()
    df_raw = df_raw[df_raw["DESCRIPTION"].isin(DESCRIPTION.keys())]
    df_raw["PARAM"] = df_raw["DESCRIPTION"].map(DESCRIPTION)

    hour_cols = df_raw.columns[7:31]

    records = []
    base_time = datetime.strptime(log_date, "%Y-%m-%d")

    for _, row in df_raw.iterrows():
        for i, col in enumerate(hour_cols):
            records.append({
                "Timestamp": base_time + timedelta(hours=i),
                "Parameter": row["PARAM"],
                "Value": pd.to_numeric(row[col], errors="coerce")
            })

    df_long = pd.DataFrame(records)

    df = df_long.pivot_table(
        index="Timestamp",
        columns="Parameter",
        values="Value"
    ).reset_index()

    # ===============================
    # DERIVED METRICS
    # ===============================
    df["DELTA_T"] = df["CDT"] - df["CIT"]
    df["PRESSURE_RATIO"] = df["CDP"] / df["CIP"]

    df = df.sort_values("Timestamp")

    # ===============================
    # FAILURE PROBABILITY (ML PLACEHOLDER)
    # Replace this with trained model later
    # ===============================
    risk_score = (
        0.3 * (df["DELTA_T"].mean() / 100) +
        0.4 * (df["PRESSURE_RATIO"].mean() / 3) +
        0.3 * (df["VIB"].mean() / 5)
    )

    failure_probability = round(min(max(risk_score, 0), 1), 2)

    # ===============================
    # SYSTEM STATE
    # ===============================
    if failure_probability < 0.5:
        state = "Waiting"
    elif 0.6 <= failure_probability <= 0.7:
        state = "Warning"
    else:
        state = "Danger"

    return df, failure_probability, state
