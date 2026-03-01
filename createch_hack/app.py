"""
AI-Driven Dynamic Construction Monitoring Prototype
Production-quality Streamlit application for intelligent construction system simulation.
Refactored: type hints, config dict, st.cache_data, deterministic seed control.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Construction Monitoring System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# Model and risk parameters (single source of truth for tuning)
MODEL_CONFIG: Dict[str, Any] = {
    "risk_low_threshold": 0.3,
    "risk_medium_threshold": 0.6,
    "weight_settlement": 0.5,
    "weight_load": 0.3,
    "weight_anomaly": 0.2,
    "isolation_forest_contamination": 0.1,
    "isolation_forest_n_estimators": 100,
    "risk_norm_settlement_ref": 30.0,
    "risk_norm_load_ref": 30.0,
    "risk_norm_anomaly_ref": 0.2,
}


def get_config(random_seed: int) -> Dict[str, Any]:
    """Return config dict with runtime seed applied (for model and simulation)."""
    return {**MODEL_CONFIG, "random_seed": random_seed}


# Scenario mode: modifiers applied to sensor simulation (noise, drift, temp, anomalies)
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Normal": {
        "noise_mult": 1.0,
        "settlement_noise_mult": 1.0,
        "load_noise_mult": 1.0,
        "temp_offset": 0,
        "temp_amplitude": 8,
        "load_drift_mult": 1.0,
        "settlement_drift_mult": 1.0,
        "anomaly_count_mult": 1.0,
        "load_spike": False,
    },
    "Monsoon": {
        "noise_mult": 1.3,
        "settlement_noise_mult": 2.0,
        "load_noise_mult": 1.0,
        "temp_offset": -6,
        "temp_amplitude": 5,
        "load_drift_mult": 1.0,
        "settlement_drift_mult": 1.4,
        "anomaly_count_mult": 1.7,
        "load_spike": False,
    },
    "Heatwave": {
        "noise_mult": 1.1,
        "settlement_noise_mult": 1.0,
        "load_noise_mult": 1.5,
        "temp_offset": 10,
        "temp_amplitude": 10,
        "load_drift_mult": 1.3,
        "settlement_drift_mult": 1.0,
        "anomaly_count_mult": 1.3,
        "load_spike": False,
    },
    "Night Shift": {
        "noise_mult": 0.5,
        "settlement_noise_mult": 0.5,
        "load_noise_mult": 0.5,
        "temp_offset": -2,
        "temp_amplitude": 5,
        "load_drift_mult": 1.0,
        "settlement_drift_mult": 0.85,
        "anomaly_count_mult": 0.5,
        "load_spike": True,
    },
}

SCENARIO_LABELS: Dict[str, str] = {
    "Normal": "Baseline conditions",
    "Monsoon": "Higher settlement noise, lower temperature, more anomalies",
    "Heatwave": "Higher temperature, higher load variance, moderate anomalies",
    "Night Shift": "Lower noise, fewer anomalies, occasional load spikes",
}


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4a4a6a;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2c3e50;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .recommendation-panel {
        background: #f0f4f8;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .risk-low { border-left-color: #28a745 !important; }
    .risk-medium { border-left-color: #ffc107 !important; }
    .risk-high { border-left-color: #dc3545 !important; }
    .scenario-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        background: linear-gradient(135deg, #e8eef4 0%, #d0dae6 100%);
        border: 1px solid #b0bcc8;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .scenario-desc { font-size: 0.8rem; color: #5a6575; margin-top: 0.2rem; }
    .rec-card-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .rec-card-table th {
        text-align: left;
        padding: 0.5rem 0.6rem;
        background: rgba(0,0,0,0.06);
        border-bottom: 1px solid #dee2e6;
        font-weight: 600;
        color: #1a1a2e;
    }
    .rec-card-table td {
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid #eee;
        color: #333;
        vertical-align: top;
    }
    .rec-card-table tr:last-child td { border-bottom: none; }
    .rec-next-check {
        font-weight: 600;
        color: #1a1a2e;
        padding: 0.5rem 0.6rem;
        background: rgba(44, 62, 80, 0.08);
        border-radius: 6px;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .stDataFrame div[data-testid="stDataFrame"] { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CSV UPLOAD: VALIDATION AND PARSING
# =============================================================================

REQUIRED_CSV_COLUMNS: Dict[str, str] = {
    "timestamp": "Timestamp",
    "actual_load_kn": "Actual_Load_kN",
    "actual_settlement_mm": "Actual_Settlement_mm",
    "temperature_c": "Temperature_C",
}


def _normalize_col(name: str) -> str:
    """Normalize column name for matching: lowercase, strip, replace spaces with underscores."""
    return str(name).strip().lower().replace(" ", "_")


def validate_and_parse_uploaded_csv(
    uploaded_file: Any,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validate uploaded CSV has required columns and parse to DataFrame.
    Required columns (case-insensitive): timestamp, actual_load_kN, actual_settlement_mm, temperature_C.
    Returns (df, None) on success, (None, error_message) on failure.
    """
    if uploaded_file is None:
        return None, "No file uploaded."
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        raw: pd.DataFrame = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read CSV: {e!s}"

    if raw.empty:
        return None, "The CSV file is empty."

    raw_cols_lower: Dict[str, str] = {_normalize_col(c): c for c in raw.columns}
    missing: List[str] = []
    rename_map: Dict[str, str] = {}
    for req_lower, canonical in REQUIRED_CSV_COLUMNS.items():
        if req_lower not in raw_cols_lower:
            missing.append(canonical)
        else:
            rename_map[raw_cols_lower[req_lower]] = canonical

    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}. Expected: timestamp, actual_load_kN, actual_settlement_mm, temperature_C."

    df = raw[list(rename_map.keys())].copy()
    df = df.rename(columns=rename_map)

    # Parse timestamp to datetime
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    except Exception as e:
        return None, f"Could not parse 'Timestamp' column as dates: {e!s}"

    # Coerce numeric columns
    for col in ["Actual_Load_kN", "Actual_Settlement_mm", "Temperature_C"]:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as e:
            return None, f"Column '{col}' must contain numbers: {e!s}"

    # Drop rows with NaN in required columns
    before: int = len(df)
    df = df.dropna(subset=["Timestamp", "Actual_Load_kN", "Actual_Settlement_mm", "Temperature_C"])
    if len(df) == 0:
        return None, "No valid rows after removing missing values in required columns."
    if len(df) < before:
        # Optional: could warn that some rows were dropped
        pass

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Actual_Load_kN"] = df["Actual_Load_kN"].astype(float).round(2)
    df["Actual_Settlement_mm"] = df["Actual_Settlement_mm"].astype(float).round(2)
    df["Temperature_C"] = df["Temperature_C"].astype(float).round(1)
    return df, None


# =============================================================================
# SIDEBAR: BASELINE DESIGN INPUT & SIMULATION PARAMETERS
# =============================================================================


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar and return all user inputs as a dictionary."""
    st.sidebar.header("Data & Baseline")
    st.sidebar.markdown("---")

    data_source: str = st.sidebar.radio(
        "Data source",
        options=["Use Simulated Data", "Upload CSV Data"],
        index=0,
        help="Simulate sensor data or upload your own CSV.",
    )
    use_simulated: bool = data_source == "Use Simulated Data"

    uploaded_file: Optional[Any] = None
    if not use_simulated:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="CSV with columns: timestamp, actual_load_kN, actual_settlement_mm, temperature_C",
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Baseline design (used for deviation and bands)")

    scenario: str = st.sidebar.selectbox(
        "Scenario Mode",
        options=list(SCENARIOS.keys()),
        index=0,
        help="Environmental / operational scenario affecting sensor simulation",
    )
    st.sidebar.caption(SCENARIO_LABELS.get(scenario, ""))

    st.sidebar.markdown("---")

    designed_load: float = st.sidebar.number_input(
        "Designed Load (kN)",
        min_value=100.0,
        max_value=5000.0,
        value=1200.0,
        step=50.0,
        help="Expected structural load in kilonewtons",
    )

    expected_settlement: float = st.sidebar.number_input(
        "Expected Settlement (mm)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
        help="Design settlement in millimeters",
    )

    deviation_threshold: int = st.sidebar.slider(
        "Allowed Deviation Threshold (%)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Percentage beyond which deviation is flagged",
    )

    n_points: int = st.sidebar.slider(
        "Number of Sensor Data Points",
        min_value=20,
        max_value=200,
        value=80,
        step=10,
        help="Length of simulated time series",
    )

    noise_level: float = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=0.5,
        value=0.08,
        step=0.01,
        help="Random noise in sensor readings (0 = clean)",
    )

    random_seed: int = st.sidebar.slider(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
        help="Deterministic seed for sensor simulation and anomaly model",
    )

    inject_anomalies: bool = st.sidebar.toggle(
        "Inject Anomalies (Demo Mode)",
        value=True,
        help="Inject synthetic anomalies for demonstration",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Design Variables (Demo)")
    baseline_reinforcement_pct: float = st.sidebar.slider(
        "Baseline reinforcement ratio (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Baseline reinforcement ratio for recalibration demo",
    )
    safety_factor: float = st.sidebar.slider(
        "Safety factor",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="Multiplier for suggested reinforcement change",
    )
    max_reinforcement_change_pct: float = st.sidebar.slider(
        "Max allowed reinforcement change (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Cap on suggested reinforcement change",
    )
    show_projected_band: bool = st.sidebar.checkbox(
        "Show projected band on Settlement chart",
        value=False,
        help="Show projected settlement level after recalibration (when triggered)",
    )
    show_projected_load_curve: bool = st.sidebar.checkbox(
        "Show projected load curve on Load chart",
        value=False,
        help="Show adjusted (reduced) load curve from Execution Plan (when MEDIUM/HIGH)",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Generate New Data", use_container_width=True):
        st.rerun()

    return {
        "use_simulated": use_simulated,
        "uploaded_file": uploaded_file,
        "scenario": scenario,
        "designed_load": designed_load,
        "expected_settlement": expected_settlement,
        "deviation_threshold": deviation_threshold,
        "n_points": n_points,
        "noise_level": noise_level,
        "random_seed": random_seed,
        "inject_anomalies": inject_anomalies,
        "baseline_reinforcement_pct": baseline_reinforcement_pct,
        "safety_factor": safety_factor,
        "max_reinforcement_change_pct": max_reinforcement_change_pct,
        "show_projected_band": show_projected_band,
        "show_projected_load_curve": show_projected_load_curve,
    }


# =============================================================================
# SENSOR SIMULATION (CACHED)
# =============================================================================


@st.cache_data(show_spinner=False)
def simulate_sensor_data(
    designed_load: float,
    expected_settlement: float,
    n_points: int,
    noise_level: float,
    inject_anomalies: bool,
    random_seed: int,
    scenario: str,
) -> pd.DataFrame:
    """
    Simulate time-series sensor data: Actual Load, Actual Settlement, Temperature.
    Scenario mode modifies noise, drift, temperature, and anomaly behavior.
    """
    np.random.seed(random_seed)
    s: Dict[str, Any] = SCENARIOS.get(scenario, SCENARIOS["Normal"])

    base_time: datetime = datetime(2026, 3, 1, 8, 0, 0)
    timestamps: List[datetime] = [
        base_time + timedelta(minutes=15 * i) for i in range(n_points)
    ]

    t: np.ndarray = np.linspace(0, 4 * np.pi, n_points)
    load_noise_eff: float = noise_level * s["load_noise_mult"]
    settlement_noise_eff: float = noise_level * s["settlement_noise_mult"]
    load_drift: float = 0.05 * s["load_drift_mult"]
    settlement_drift: float = 0.08 * s["settlement_drift_mult"]

    load_base: np.ndarray = designed_load * (
        1 + load_drift * np.sin(t) + load_noise_eff * np.random.randn(n_points)
    )
    settlement_base: np.ndarray = expected_settlement * (
        1
        + settlement_drift * np.sin(t + 0.5)
        + settlement_noise_eff * np.random.randn(n_points)
    )
    temp_center: float = 20 + s["temp_offset"]
    temp_amp: float = s["temp_amplitude"]
    temp_noise_eff: float = noise_level * 5 * s["noise_mult"]
    temperature_base: np.ndarray = (
        temp_center
        + temp_amp * np.sin(t * 0.7)
        + temp_noise_eff * np.random.randn(n_points)
    )
    temperature_base = np.clip(temperature_base, 5, 45)

    if inject_anomalies:
        n_anomaly_max: int = min(6, max(3, n_points // 10))
        n_anomalies_raw: int = int(np.random.randint(3, n_anomaly_max + 1))
        n_anomalies = max(
            0,
            min(
                n_points,
                int(round(n_anomalies_raw * s["anomaly_count_mult"])),
            ),
        )
        if n_anomalies > 0:
            anomaly_indices: np.ndarray = np.random.choice(
                n_points, size=n_anomalies, replace=False
            )
            for idx in anomaly_indices:
                load_base[idx] *= np.random.uniform(1.3, 1.8)
                settlement_base[idx] *= np.random.uniform(1.4, 2.0)
                temperature_base[idx] += np.random.uniform(5, 15)

    if s.get("load_spike") and inject_anomalies and n_points >= 5:
        n_spikes: int = min(2, n_points // 20 + 1)
        spike_indices: np.ndarray = np.random.choice(
            n_points, size=n_spikes, replace=False
        )
        for idx in spike_indices:
            load_base[idx] *= np.random.uniform(1.6, 2.2)

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Actual_Load_kN": np.round(load_base, 2),
            "Actual_Settlement_mm": np.round(settlement_base, 2),
            "Temperature_C": np.round(temperature_base, 1),
        }
    )


# =============================================================================
# DEVIATION CALCULATION (CACHED)
# =============================================================================


@st.cache_data(show_spinner=False)
def compute_deviations(
    df: pd.DataFrame,
    designed_load: float,
    expected_settlement: float,
    threshold_pct: float,
) -> pd.DataFrame:
    """
    Compute percentage deviation from baseline and flag threshold exceedance.
    Returns a new DataFrame; does not mutate input.
    """
    df = df.copy()
    df["Load_Deviation_Pct"] = (
        (df["Actual_Load_kN"] - designed_load) / designed_load * 100
    )
    df["Settlement_Deviation_Pct"] = (
        (df["Actual_Settlement_mm"] - expected_settlement) / expected_settlement * 100
    )
    df["Load_Threshold_Exceeded"] = np.abs(df["Load_Deviation_Pct"]) > threshold_pct
    df["Settlement_Threshold_Exceeded"] = (
        np.abs(df["Settlement_Deviation_Pct"]) > threshold_pct
    )
    return df


# =============================================================================
# AI ANOMALY DETECTION — ISOLATION FOREST (CACHED)
# =============================================================================


@st.cache_data(show_spinner=False)
def detect_anomalies(
    df: pd.DataFrame,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Train IsolationForest on sensor features and add Anomaly column (Yes/No).
    Uses primitive args for cache hashing.
    """
    df = df.copy()
    features: np.ndarray = df[
        ["Actual_Load_kN", "Actual_Settlement_mm", "Temperature_C"]
    ].values
    iso_forest: IsolationForest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    predictions: np.ndarray = iso_forest.fit_predict(features)
    df["Anomaly"] = np.where(predictions == -1, "Yes", "No")
    return df


# =============================================================================
# RISK SCORING ENGINE (CACHED)
# =============================================================================


@st.cache_data(show_spinner=False)
def compute_risk_score(
    df: pd.DataFrame,
    weight_settlement: float,
    weight_load: float,
    weight_anomaly: float,
    risk_low_threshold: float,
    risk_medium_threshold: float,
    norm_settlement_ref: float,
    norm_load_ref: float,
    norm_anomaly_ref: float,
) -> Tuple[float, str]:
    """
    Compute normalized risk score (0–1) from weighted settlement deviation,
    load deviation, and anomaly frequency. Returns (score, classification).
    """
    settlement_dev: float = float(np.abs(df["Settlement_Deviation_Pct"]).mean())
    load_dev: float = float(np.abs(df["Load_Deviation_Pct"]).mean())
    anomaly_freq: float = (df["Anomaly"] == "Yes").sum() / len(df)

    norm_settlement: float = min(settlement_dev / norm_settlement_ref, 1.0)
    norm_load: float = min(load_dev / norm_load_ref, 1.0)
    norm_anomaly: float = min(anomaly_freq / norm_anomaly_ref, 1.0)

    risk: float = (
        weight_settlement * norm_settlement
        + weight_load * norm_load
        + weight_anomaly * norm_anomaly
    )
    risk = min(max(risk, 0.0), 1.0)

    if risk < risk_low_threshold:
        classification: str = "LOW"
    elif risk <= risk_medium_threshold:
        classification = "MEDIUM"
    else:
        classification = "HIGH"

    return round(risk, 3), classification


# =============================================================================
# ADAPTIVE DESIGN RECALIBRATION ENGINE (PS-3 STYLE)
# =============================================================================


def compute_projected_risk(
    settlement_dev_pct: float,
    load_dev_pct: float,
    anomaly_freq: float,
    weight_settlement: float,
    weight_load: float,
    weight_anomaly: float,
    risk_low_threshold: float,
    risk_medium_threshold: float,
    norm_settlement_ref: float,
    norm_load_ref: float,
    norm_anomaly_ref: float,
) -> Tuple[float, str]:
    """Compute risk score and classification from given settlement/dev/anomaly metrics."""
    norm_settlement: float = min(settlement_dev_pct / norm_settlement_ref, 1.0)
    norm_load: float = min(load_dev_pct / norm_load_ref, 1.0)
    norm_anomaly: float = min(anomaly_freq / norm_anomaly_ref, 1.0)
    risk: float = (
        weight_settlement * norm_settlement
        + weight_load * norm_load
        + weight_anomaly * norm_anomaly
    )
    risk = min(max(risk, 0.0), 1.0)
    if risk < risk_low_threshold:
        classification = "LOW"
    elif risk <= risk_medium_threshold:
        classification = "MEDIUM"
    else:
        classification = "HIGH"
    return round(risk, 3), classification


def run_recalibration(
    current_settlement_dev_pct: float,
    threshold_pct: float,
    baseline_reinforcement_pct: float,
    safety_factor: float,
    max_reinforcement_change_pct: float,
) -> Optional[Dict[str, Any]]:
    """
    If |settlement deviation| > threshold, return recalibration result dict; else None.
    D = max(0, |dev| - threshold)
    Suggested change (%) = min(max_change, safety_factor * 0.6 * D)
    New reinforcement (%) = baseline * (1 + change/100)
    Projected settlement dev % = current * (1 - min(0.35, change/100))
    """
    abs_dev: float = abs(current_settlement_dev_pct)
    if abs_dev <= threshold_pct:
        return None
    D: float = max(0.0, abs_dev - threshold_pct)
    change_pct: float = min(
        max_reinforcement_change_pct,
        safety_factor * 0.6 * D,
    )
    new_reinforcement_pct: float = baseline_reinforcement_pct * (1 + change_pct / 100.0)
    projected_settlement_dev_pct: float = current_settlement_dev_pct * (
        1 - min(0.35, change_pct / 100.0)
    )
    return {
        "baseline_reinforcement_pct": baseline_reinforcement_pct,
        "suggested_change_pct": round(change_pct, 2),
        "suggested_reinforcement_pct": round(new_reinforcement_pct, 2),
        "current_settlement_dev_pct": round(current_settlement_dev_pct, 2),
        "projected_settlement_dev_pct": round(projected_settlement_dev_pct, 2),
        "change_pct": change_pct,
    }


def generate_design_options(
    recal_result: Dict[str, Any],
    max_reinforcement_change_pct: float,
    load_dev_pct: float,
    anomaly_freq: float,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate 3 candidate reinforcement options (low, Balanced, high) with projected
    settlement, risk, cost impact (0.15 * change%), and schedule impact (0, +1, +2 days).
    """
    current_dev: float = recal_result["current_settlement_dev_pct"]
    baseline_reinf: float = recal_result["baseline_reinforcement_pct"]
    suggested_change: float = recal_result["change_pct"]

    change_low: float = max(0.5, suggested_change * 0.5)
    change_med: float = suggested_change
    change_high: float = min(max_reinforcement_change_pct, suggested_change * 1.5)

    def option_row(
        label: str,
        change_pct: float,
        schedule_days: int,
    ) -> Dict[str, Any]:
        reinf_pct: float = baseline_reinf * (1 + change_pct / 100.0)
        proj_dev_pct: float = current_dev * (1 - min(0.35, change_pct / 100.0))
        risk_score, risk_level = compute_projected_risk(
            settlement_dev_pct=abs(proj_dev_pct),
            load_dev_pct=load_dev_pct,
            anomaly_freq=anomaly_freq,
            weight_settlement=cfg["weight_settlement"],
            weight_load=cfg["weight_load"],
            weight_anomaly=cfg["weight_anomaly"],
            risk_low_threshold=cfg["risk_low_threshold"],
            risk_medium_threshold=cfg["risk_medium_threshold"],
            norm_settlement_ref=cfg["risk_norm_settlement_ref"],
            norm_load_ref=cfg["risk_norm_load_ref"],
            norm_anomaly_ref=cfg["risk_norm_anomaly_ref"],
        )
        cost_impact_pct: float = 0.15 * change_pct
        return {
            "Option": label,
            "Suggested reinforcement (%)": round(reinf_pct, 2),
            "Projected settlement deviation (%)": round(proj_dev_pct, 2),
            "Projected risk score": risk_score,
            "Projected risk level": risk_level,
            "Estimated cost impact (%)": round(cost_impact_pct, 2),
            "Estimated schedule impact (days)": schedule_days,
        }

    rows: List[Dict[str, Any]] = [
        option_row("Low", change_low, 0),
        option_row("Balanced", change_med, 1),
        option_row("High", change_high, 2),
    ]
    return pd.DataFrame(rows)


# =============================================================================
# ADAPTIVE RECOMMENDATION ENGINE (QUANTIFIED ACTIONS)
# =============================================================================

# Thresholds (as %) above which we trigger specific quantified advice
SETTLEMENT_DEV_HIGH_THRESHOLD_PCT: float = 10.0
LOAD_DEV_HIGH_THRESHOLD_PCT: float = 10.0
ANOMALY_CLUSTER_WINDOW: int = 5  # indices within which anomalies count as clustered


def are_anomalies_clustered(df: pd.DataFrame) -> bool:
    """True if at least two anomalies occur within ANOMALY_CLUSTER_WINDOW indices."""
    anomaly_idx: np.ndarray = np.where(df["Anomaly"].values == "Yes")[0]
    if len(anomaly_idx) < 2:
        return False
    for i in range(len(anomaly_idx) - 1):
        if anomaly_idx[i + 1] - anomaly_idx[i] <= ANOMALY_CLUSTER_WINDOW:
            return True
    return False


def get_suggested_next_check_hrs(risk_class: str) -> int:
    """Suggested next check interval in hours based on risk level."""
    if risk_class == "HIGH":
        return 6
    if risk_class == "MEDIUM":
        return 12
    return 24


def get_quantified_recommendations(
    risk_class: str,
    settlement_dev_pct: float,
    load_dev_pct: float,
    anomaly_pct: float,
    anomalies_clustered: bool,
) -> Dict[str, Any]:
    """
    Return structured recommendations: quantified actions plus suggested next check time.
    actions: list of {"action": str, "detail": str}; suggested_next_check_hrs: int.
    """
    actions: List[Dict[str, str]] = []

    # Settlement deviation high -> reinforcement check + % range
    if settlement_dev_pct >= SETTLEMENT_DEV_HIGH_THRESHOLD_PCT:
        low_pct: float = max(5, min(10, round(settlement_dev_pct * 0.5, 1)))
        high_pct: float = max(10, min(20, round(settlement_dev_pct * 1.0, 1)))
        if high_pct <= low_pct:
            high_pct = min(20, low_pct + 5)
        actions.append(
            {
                "action": "Increase reinforcement check priority",
                "detail": f"Review design for {low_pct:.0f}–{high_pct:.0f}% reinforcement adjustment.",
            }
        )

    # Load deviation high -> sequencing / load staging + reduce peak by X%
    if load_dev_pct >= LOAD_DEV_HIGH_THRESHOLD_PCT:
        reduce_pct: float = min(25, max(5, round(load_dev_pct * 0.6, 1)))
        actions.append(
            {
                "action": "Adjust sequencing / load staging",
                "detail": f"Consider reducing peak load by ~{reduce_pct:.0f}%.",
            }
        )

    # Anomalies clustered -> sensor integrity + targeted QA
    if anomalies_clustered:
        actions.append(
            {
                "action": "Inspect sensor/measurement integrity",
                "detail": "Run targeted QA on sensors in the affected time window.",
            }
        )
        actions.append(
            {
                "action": "Run targeted QA",
                "detail": "Verify calibration and wiring for clustered anomaly period.",
            }
        )

    # Risk-level fallbacks when no driver-specific actions
    if not actions:
        if risk_class == "LOW":
            actions.append(
                {"action": "Continue routine monitoring", "detail": "No immediate action required."}
            )
        elif risk_class == "MEDIUM":
            actions.append(
                {
                    "action": "Increase inspection frequency",
                    "detail": "Daily or twice-daily checks until trend stabilizes.",
                }
            )
        else:
            actions.append(
                {
                    "action": "Trigger engineering review",
                    "detail": "Formal review and recalculation of monitoring plan.",
                }
            )

    if risk_class == "MEDIUM" and not any(
        a["action"] == "Increase inspection frequency" for a in actions
    ):
        actions.append(
            {
                "action": "Increase inspection frequency",
                "detail": "Consider daily or twice-daily checks.",
            }
        )
    if risk_class == "HIGH":
        actions.append(
            {
                "action": "Consider temporary load reduction",
                "detail": "Shoring or schedule adjustment if deviation persists.",
            }
        )

    return {
        "actions": actions,
        "suggested_next_check_hrs": get_suggested_next_check_hrs(risk_class),
    }


# =============================================================================
# EXPLAINABILITY: RISK CONTRIBUTORS & NATURAL-LANGUAGE EXPLANATION
# =============================================================================


def compute_risk_contributors(
    df: pd.DataFrame,
    weight_settlement: float,
    weight_load: float,
    weight_anomaly: float,
    norm_settlement_ref: float,
    norm_load_ref: float,
    norm_anomaly_ref: float,
) -> Dict[str, float]:
    """
    Compute raw metrics (as %) and each component's contribution to risk score.
    Returns dict with settlement_dev_pct, load_dev_pct, anomaly_pct, and
    contrib_settlement, contrib_load, contrib_anomaly (weighted, 0–1 scale).
    """
    settlement_dev_pct: float = float(np.abs(df["Settlement_Deviation_Pct"]).mean())
    load_dev_pct: float = float(np.abs(df["Load_Deviation_Pct"]).mean())
    anomaly_freq: float = (df["Anomaly"] == "Yes").sum() / len(df)
    anomaly_pct: float = anomaly_freq * 100.0

    norm_settlement: float = min(settlement_dev_pct / norm_settlement_ref, 1.0)
    norm_load: float = min(load_dev_pct / norm_load_ref, 1.0)
    norm_anomaly: float = min(anomaly_freq / norm_anomaly_ref, 1.0)

    contrib_settlement: float = weight_settlement * norm_settlement
    contrib_load: float = weight_load * norm_load
    contrib_anomaly: float = weight_anomaly * norm_anomaly

    return {
        "settlement_dev_pct": round(settlement_dev_pct, 1),
        "load_dev_pct": round(load_dev_pct, 1),
        "anomaly_pct": round(anomaly_pct, 1),
        "contrib_settlement": round(contrib_settlement, 3),
        "contrib_load": round(contrib_load, 3),
        "contrib_anomaly": round(contrib_anomaly, 3),
    }


def build_risk_explanation(
    risk_class: str,
    settlement_dev_pct: float,
    load_dev_pct: float,
    anomaly_pct: float,
    contrib_settlement: float,
    contrib_load: float,
    contrib_anomaly: float,
) -> str:
    """
    Build a short natural-language explanation of why the risk level is as shown.
    Emphasizes top contributors in order.
    """
    items: List[Tuple[str, float, float]] = [
        ("settlement deviation", settlement_dev_pct, contrib_settlement),
        ("load deviation", load_dev_pct, contrib_load),
        ("anomaly rate", anomaly_pct, contrib_anomaly),
    ]
    items.sort(key=lambda x: x[2], reverse=True)

    parts: List[str] = []
    for label, pct_val, _ in items:
        parts.append(f"{label} of {pct_val:.1f}%")

    if len(parts) == 0:
        return f"Risk is {risk_class}. No significant contributors."
    if len(parts) == 1:
        return f"Risk is {risk_class} mainly due to {parts[0]}."
    if len(parts) == 2:
        return f"Risk is {risk_class} mainly due to {parts[0]} and {parts[1]}."
    return f"Risk is {risk_class} mainly due to {parts[0]}, {parts[1]}, and {parts[2]}."


def create_contribution_barchart(
    contrib_settlement: float,
    contrib_load: float,
    contrib_anomaly: float,
) -> go.Figure:
    """Create a small horizontal bar chart of risk score contributions."""
    labels: List[str] = [
        "Settlement deviation",
        "Load deviation",
        "Anomaly frequency",
    ]
    values: List[float] = [contrib_settlement, contrib_load, contrib_anomaly]
    colors: List[str] = ["#2c3e50", "#3498db", "#7f8c8d"]

    fig: go.Figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont_size=12,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=160,
        margin=dict(l=10, r=60, t=8, b=8),
        xaxis=dict(
            range=[0, max(1.0, max(values) * 1.2)],
            title="Contribution to risk score",
            tickformat=".2f",
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


# =============================================================================
# PLOTLY TREND CHARTS WITH DIGITAL TWIN BAND
# =============================================================================

# Colors readable on both light and dark themes
CHART_BAND_FILL: str = "rgba(100, 149, 237, 0.15)"
CHART_BAND_LINE: str = "rgba(100, 149, 237, 0.7)"
CHART_BASELINE_LINE: str = "rgba(160, 160, 160, 0.95)"
CHART_MAIN_LINE: str = "#1f77b4"
CHART_OUT_OF_BAND_MARKER: str = "#f59e0b"
CHART_ANOMALY_MARKER: str = "#ef4444"


def create_trend_chart_with_band(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    baseline: float,
    deviation_threshold_pct: float,
    title: str,
    yaxis_title: str,
    anomaly_mask: pd.Series,
) -> go.Figure:
    """
    Build a trend chart with Digital Twin Expected Band: baseline (dashed),
    upper/lower tolerance bands (shaded), main series, out-of-band markers,
    and anomaly markers. Styled for readability on dark theme.
    """
    threshold: float = deviation_threshold_pct / 100.0
    upper: float = baseline * (1 + threshold)
    lower: float = baseline * (1 - threshold)

    n_pts: int = len(df)
    x_vals: pd.Series = df[x_col]
    y_vals: pd.Series = df[y_col]
    upper_arr: np.ndarray = np.full(n_pts, upper)
    lower_arr: np.ndarray = np.full(n_pts, lower)
    baseline_arr: np.ndarray = np.full(n_pts, baseline)

    out_of_band: np.ndarray = (y_vals > upper) | (y_vals < lower)
    has_out: bool = bool(out_of_band.any())

    # Trace order: band (back), baseline, main line, out-of-band, anomaly
    fig = go.Figure()

    # 1. Lower bound of band (dashed)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=lower_arr,
            mode="lines",
            line=dict(dash="dash", color=CHART_BAND_LINE, width=1.5),
            name="Lower tolerance",
            legendgroup="band",
        )
    )
    # 2. Upper bound of band (dashed) with fill to lower
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=upper_arr,
            mode="lines",
            line=dict(dash="dash", color=CHART_BAND_LINE, width=1.5),
            fill="tonexty",
            fillcolor=CHART_BAND_FILL,
            name="Expected band",
            legendgroup="band",
        )
    )
    # 3. Baseline (dashed)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=baseline_arr,
            mode="lines",
            line=dict(dash="dash", color=CHART_BASELINE_LINE, width=2),
            name="Digital twin baseline",
        )
    )
    # 4. Actual series (line + markers)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line=dict(color=CHART_MAIN_LINE, width=2),
            marker=dict(size=6),
            name="Actual",
        )
    )
    # 5. Out-of-band points (distinct marker)
    if has_out:
        fig.add_trace(
            go.Scatter(
                x=df.loc[out_of_band, x_col],
                y=df.loc[out_of_band, y_col],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=CHART_OUT_OF_BAND_MARKER,
                    line=dict(width=1.5, color="rgba(0,0,0,0.4)"),
                ),
                name="Outside tolerance band",
            )
        )
    # 6. Anomaly points (existing style)
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[anomaly_mask, x_col],
                y=df.loc[anomaly_mask, y_col],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color=CHART_ANOMALY_MARKER,
                    line=dict(width=2),
                ),
                name="Anomaly",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        title=title,
        xaxis_title="",
        yaxis_title=yaxis_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        font=dict(color="rgba(255,255,255,0.9)", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


# =============================================================================
# PLOTLY RISK GAUGE
# =============================================================================


def create_risk_gauge(
    risk_score: float,
    risk_class: str,
    risk_low_threshold: float,
    risk_medium_threshold: float,
) -> go.Figure:
    """Create a Plotly gauge chart for risk score (0–1)."""
    if risk_class == "LOW":
        color: str = "#28a745"
    elif risk_class == "MEDIUM":
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig: go.Figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#dee2e6",
                "steps": [
                    {"range": [0, risk_low_threshold], "color": "rgba(40, 167, 69, 0.2)"},
                    {
                        "range": [risk_low_threshold, risk_medium_threshold],
                        "color": "rgba(255, 193, 7, 0.2)",
                    },
                    {"range": [risk_medium_threshold, 1], "color": "rgba(220, 53, 69, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
            title={"text": f"Risk: {risk_class}", "font": {"size": 16}},
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main() -> None:
    """Entry point: render header, sidebar, run pipeline, and dashboard."""
    st.markdown(
        '<p class="main-header">AI-Driven Dynamic Construction Monitoring</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Baseline design validation, real-time sensor simulation, anomaly detection, and risk-based recommendations</p>',
        unsafe_allow_html=True,
    )

    params: Dict[str, Any] = render_sidebar()
    cfg: Dict[str, Any] = get_config(params["random_seed"])

    if params["use_simulated"]:
        scenario_key: str = params["scenario"]
        st.markdown(
            f'<div class="scenario-badge">Scenario: {scenario_key}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="scenario-desc">{SCENARIO_LABELS.get(scenario_key, "")}</p>',
            unsafe_allow_html=True,
        )
        df = simulate_sensor_data(
            params["designed_load"],
            params["expected_settlement"],
            params["n_points"],
            params["noise_level"],
            params["inject_anomalies"],
            params["random_seed"],
            params["scenario"],
        )
    else:
        st.markdown(
            '<div class="scenario-badge">Data source: Uploaded CSV</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="scenario-desc">Deviation, anomaly detection, and risk scoring run on your data.</p>',
            unsafe_allow_html=True,
        )
        if params["uploaded_file"] is None:
            st.info("Upload a CSV file in the sidebar to run analysis. Required columns: **timestamp**, **actual_load_kN**, **actual_settlement_mm**, **temperature_C**.")
            return
        df, parse_error = validate_and_parse_uploaded_csv(params["uploaded_file"])
        if parse_error is not None:
            st.error(parse_error)
            return
        assert df is not None

    # Pipeline: deviations -> anomalies -> risk (same for simulated or uploaded)
    df = compute_deviations(
        df,
        params["designed_load"],
        params["expected_settlement"],
        float(params["deviation_threshold"]),
    )

    df = detect_anomalies(
        df,
        contamination=cfg["isolation_forest_contamination"],
        n_estimators=cfg["isolation_forest_n_estimators"],
        random_state=cfg["random_seed"],
    )

    risk_score, risk_class = compute_risk_score(
        df,
        weight_settlement=cfg["weight_settlement"],
        weight_load=cfg["weight_load"],
        weight_anomaly=cfg["weight_anomaly"],
        risk_low_threshold=cfg["risk_low_threshold"],
        risk_medium_threshold=cfg["risk_medium_threshold"],
        norm_settlement_ref=cfg["risk_norm_settlement_ref"],
        norm_load_ref=cfg["risk_norm_load_ref"],
        norm_anomaly_ref=cfg["risk_norm_anomaly_ref"],
    )
    anomalies_clustered: bool = are_anomalies_clustered(df)
    current_settlement_dev_pct: float = float(np.abs(df["Settlement_Deviation_Pct"]).mean())
    load_dev_pct: float = float(np.abs(df["Load_Deviation_Pct"]).mean())
    anomaly_freq: float = (df["Anomaly"] == "Yes").sum() / len(df)
    anomaly_pct: float = anomaly_freq * 100.0

    rec_result: Dict[str, Any] = get_quantified_recommendations(
        risk_class=risk_class,
        settlement_dev_pct=current_settlement_dev_pct,
        load_dev_pct=load_dev_pct,
        anomaly_pct=anomaly_pct,
        anomalies_clustered=anomalies_clustered,
    )

    # Recalibration (trigger when settlement deviation exceeds threshold)
    recal_result: Optional[Dict[str, Any]] = run_recalibration(
        current_settlement_dev_pct=current_settlement_dev_pct,
        threshold_pct=float(params["deviation_threshold"]),
        baseline_reinforcement_pct=params["baseline_reinforcement_pct"],
        safety_factor=params["safety_factor"],
        max_reinforcement_change_pct=params["max_reinforcement_change_pct"],
    )
    projected_risk_score: Optional[float] = None
    projected_risk_class: Optional[str] = None
    if recal_result is not None:
        projected_risk_score, projected_risk_class = compute_projected_risk(
            settlement_dev_pct=abs(recal_result["projected_settlement_dev_pct"]),
            load_dev_pct=load_dev_pct,
            anomaly_freq=anomaly_freq,
            weight_settlement=cfg["weight_settlement"],
            weight_load=cfg["weight_load"],
            weight_anomaly=cfg["weight_anomaly"],
            risk_low_threshold=cfg["risk_low_threshold"],
            risk_medium_threshold=cfg["risk_medium_threshold"],
            norm_settlement_ref=cfg["risk_norm_settlement_ref"],
            norm_load_ref=cfg["risk_norm_load_ref"],
            norm_anomaly_ref=cfg["risk_norm_anomaly_ref"],
        )

    # Execution plan adjustment (demo): peak load reduction, schedule buffer, inspection
    if risk_class == "LOW":
        peak_load_reduction_pct: float = 0.0
        schedule_buffer_days: int = 0
        inspection_recommendation: str = "No change"
    elif risk_class == "MEDIUM":
        peak_load_reduction_pct = 5.0
        schedule_buffer_days = 0
        inspection_recommendation = "Increase inspection frequency"
    else:
        peak_load_reduction_pct = 10.0
        schedule_buffer_days = 2
        inspection_recommendation = "Increase inspection frequency; add schedule buffer"
    original_peak_load: float = float(df["Actual_Load_kN"].max())
    adjusted_peak_load: float = original_peak_load * (1 - peak_load_reduction_pct / 100.0)
    projected_load_curve: np.ndarray = (df["Actual_Load_kN"].values * (1 - peak_load_reduction_pct / 100.0)).astype(float)

    # -------------------------------------------------------------------------
    # KPI Row
    # -------------------------------------------------------------------------
    n_anomalies: int = int((df["Anomaly"] == "Yes").sum())
    load_exceeded: int = int(df["Load_Threshold_Exceeded"].sum())
    avg_settlement: float = float(df["Actual_Settlement_mm"].mean())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="kpi-card risk-{risk_class.lower()}">'
            f'<div class="kpi-label">Risk Level</div>'
            f'<div class="kpi-value">{risk_class}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Anomalies Detected</div>'
            f'<div class="kpi-value">{n_anomalies}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Load Threshold Exceedances</div>'
            f'<div class="kpi-value">{load_exceeded}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Avg Settlement (mm)</div>'
            f'<div class="kpi-value">{avg_settlement:.1f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Charts Row: Load trend + Settlement trend (Digital Twin Expected Band)
    # -------------------------------------------------------------------------
    c1, c2 = st.columns(2)
    deviation_pct: float = float(params["deviation_threshold"])
    anomaly_mask: pd.Series = df["Anomaly"] == "Yes"

    with c1:
        fig_load = create_trend_chart_with_band(
            df=df,
            x_col="Timestamp",
            y_col="Actual_Load_kN",
            baseline=params["designed_load"],
            deviation_threshold_pct=deviation_pct,
            title="Load Trend (kN)",
            yaxis_title="Load (kN)",
            anomaly_mask=anomaly_mask,
        )
        if params["show_projected_load_curve"] and (risk_class == "MEDIUM" or risk_class == "HIGH"):
            fig_load.add_trace(
                go.Scatter(
                    x=df["Timestamp"],
                    y=projected_load_curve,
                    mode="lines",
                    line=dict(dash="dash", color="rgba(40, 167, 69, 0.9)", width=2),
                    name="Projected (adjusted load)",
                )
            )
        st.plotly_chart(fig_load, use_container_width=True)

    with c2:
        fig_settlement = create_trend_chart_with_band(
            df=df,
            x_col="Timestamp",
            y_col="Actual_Settlement_mm",
            baseline=params["expected_settlement"],
            deviation_threshold_pct=deviation_pct,
            title="Settlement Trend (mm)",
            yaxis_title="Settlement (mm)",
            anomaly_mask=anomaly_mask,
        )
        if recal_result is not None and params["show_projected_band"]:
            proj_dev: float = recal_result["projected_settlement_dev_pct"]
            proj_level: float = params["expected_settlement"] * (1 + proj_dev / 100.0)
            fig_settlement.add_trace(
                go.Scatter(
                    x=df["Timestamp"],
                    y=np.full(len(df), proj_level),
                    mode="lines",
                    line=dict(dash="dot", color="rgba(40, 167, 69, 0.9)", width=2),
                    name="Projected band (post-recal)",
                )
            )
        st.plotly_chart(fig_settlement, use_container_width=True)

    # -------------------------------------------------------------------------
    # Design Recalibration Simulation panel
    # -------------------------------------------------------------------------
    if recal_result is not None and projected_risk_score is not None and projected_risk_class is not None:
        st.markdown("---")
        st.subheader("Design Recalibration Simulation")
        st.caption(
            "System recalibrated reinforcement due to settlement deviation exceeding threshold."
        )
        recalc_df = pd.DataFrame(
            {
                "Metric": [
                    "Reinforcement (%)",
                    "Settlement deviation (%)",
                    "Risk score",
                    "Risk level",
                ],
                "Current / Baseline": [
                    f"{recal_result['baseline_reinforcement_pct']:.2f}",
                    f"{recal_result['current_settlement_dev_pct']:.2f}",
                    f"{risk_score:.3f}",
                    risk_class,
                ],
                "Projected / Suggested": [
                    f"{recal_result['suggested_reinforcement_pct']:.2f}",
                    f"{recal_result['projected_settlement_dev_pct']:.2f}",
                    f"{projected_risk_score:.3f}",
                    projected_risk_class,
                ],
            }
        )
        st.dataframe(recalc_df, use_container_width=True, hide_index=True)

        # Generated Design Options (Trade-off) - only when recalibration triggered
        options_df: pd.DataFrame = generate_design_options(
            recal_result=recal_result,
            max_reinforcement_change_pct=params["max_reinforcement_change_pct"],
            load_dev_pct=load_dev_pct,
            anomaly_freq=anomaly_freq,
            cfg=cfg,
        )

        st.markdown("---")
        st.subheader("Generated Design Options (Trade-off)")
        st.caption("Three candidate reinforcement adjustments. Sort by column; Balanced is the recommended option.")

        def highlight_balanced(row: pd.Series) -> List[str]:
            if row["Option"] == "Balanced":
                return ["background-color: #d4edda; font-weight: 600"] * len(row)
            return [""] * len(row)

        st.dataframe(
            options_df.style.apply(highlight_balanced, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Execution Plan Adjustment (Demo)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Execution Plan Adjustment (Demo)")
    plan_df = pd.DataFrame(
        {
            "Item": [
                "Original peak load (kN)",
                "Adjusted peak load (kN)",
                "Peak load reduction",
                "Schedule buffer recommendation",
                "Inspection",
            ],
            "Value": [
                f"{original_peak_load:.1f}",
                f"{adjusted_peak_load:.1f}",
                f"{peak_load_reduction_pct:.0f}%" if peak_load_reduction_pct else "0%",
                f"+{schedule_buffer_days} days" if schedule_buffer_days else "None",
                inspection_recommendation,
            ],
        }
    )
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.caption("Toggle 'Show projected load curve on Load chart' in the sidebar to overlay the adjusted load curve.")

    # -------------------------------------------------------------------------
    # Verification (Demo Metrics)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Verification (Demo Metrics)")
    n_total: int = len(df)
    pct_outside_band: float = float(
        ((df["Load_Threshold_Exceeded"] | df["Settlement_Threshold_Exceeded"]).sum() / n_total * 100)
    ) if n_total else 0.0
    anomaly_rate_pct: float = float((df["Anomaly"] == "Yes").sum() / n_total * 100) if n_total else 0.0
    if projected_risk_score is not None and risk_score > 0:
        risk_improvement_pct: float = (risk_score - projected_risk_score) / risk_score * 100.0
        risk_improvement_str: str = f"{risk_improvement_pct:.1f}%"
    else:
        risk_improvement_str = "—"

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">% points outside tolerance band</div>'
            f'<div class="kpi-value">{pct_outside_band:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Anomaly rate</div>'
            f'<div class="kpi-value">{anomaly_rate_pct:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Risk score improvement (post-recal)</div>'
            f'<div class="kpi-value">{risk_improvement_str}</div></div>',
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Risk Gauge + Anomaly Table + Recommendations
    # -------------------------------------------------------------------------
    col_gauge, col_table, col_rec = st.columns([1, 2, 1])

    with col_gauge:
        st.plotly_chart(
            create_risk_gauge(
                risk_score,
                risk_class,
                cfg["risk_low_threshold"],
                cfg["risk_medium_threshold"],
            ),
            use_container_width=True,
        )

    with col_table:
        st.subheader("Sensor Data & Anomalies")
        display_df: pd.DataFrame = df[
            [
                "Timestamp",
                "Actual_Load_kN",
                "Actual_Settlement_mm",
                "Temperature_C",
                "Load_Deviation_Pct",
                "Settlement_Deviation_Pct",
                "Anomaly",
            ]
        ].copy()
        display_df["Load_Deviation_Pct"] = display_df["Load_Deviation_Pct"].round(2)
        display_df["Settlement_Deviation_Pct"] = (
            display_df["Settlement_Deviation_Pct"].round(2)
        )

        def highlight_anomaly_rows(row: pd.Series) -> List[str]:
            if row["Anomaly"] == "Yes":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_anomaly_rows, axis=1).format(
                {
                    "Actual_Load_kN": "{:.1f}",
                    "Actual_Settlement_mm": "{:.1f}",
                    "Temperature_C": "{:.1f}",
                }
            ),
            use_container_width=True,
            height=320,
        )

    with col_rec:
        st.subheader("Recommendations")
        risk_css: str = risk_class.lower()
        next_hrs: int = rec_result["suggested_next_check_hrs"]
        rec_actions: List[Dict[str, str]] = rec_result["actions"]
        table_rows: str = "".join(
            f'<tr><td>{a["action"]}</td><td>{a["detail"]}</td></tr>'
            for a in rec_actions
        )
        st.markdown(
            f'<div class="recommendation-panel risk-{risk_css}">'
            f'<div class="rec-next-check">Suggested next check: {next_hrs} hrs</div>'
            '<table class="rec-card-table">'
            "<thead><tr><th>Action</th><th>Detail</th></tr></thead>"
            f"<tbody>{table_rows}</tbody></table>"
            "</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Explainability / Why this risk?
    # -------------------------------------------------------------------------
    contributors: Dict[str, float] = compute_risk_contributors(
        df,
        weight_settlement=cfg["weight_settlement"],
        weight_load=cfg["weight_load"],
        weight_anomaly=cfg["weight_anomaly"],
        norm_settlement_ref=cfg["risk_norm_settlement_ref"],
        norm_load_ref=cfg["risk_norm_load_ref"],
        norm_anomaly_ref=cfg["risk_norm_anomaly_ref"],
    )
    explanation_text: str = build_risk_explanation(
        risk_class,
        contributors["settlement_dev_pct"],
        contributors["load_dev_pct"],
        contributors["anomaly_pct"],
        contributors["contrib_settlement"],
        contributors["contrib_load"],
        contributors["contrib_anomaly"],
    )
    contrib_fig: go.Figure = create_contribution_barchart(
        contributors["contrib_settlement"],
        contributors["contrib_load"],
        contributors["contrib_anomaly"],
    )

    st.markdown("---")
    st.subheader("Explainability / Why this risk?")
    expl_col1, expl_col2 = st.columns([1, 1])
    with expl_col1:
        st.markdown(
            '<div class="recommendation-panel" style="margin-bottom:0;">'
            "<p style='margin:0; font-size:0.95rem; line-height:1.5; color:#1a1a2e;'>"
            f"{explanation_text}"
            "</p></div>",
            unsafe_allow_html=True,
        )
    with expl_col2:
        st.plotly_chart(contrib_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Export: CSV + one-page HTML report
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Export")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_cols: List[str] = [
            "Timestamp",
            "Actual_Load_kN",
            "Actual_Settlement_mm",
            "Temperature_C",
            "Load_Deviation_Pct",
            "Settlement_Deviation_Pct",
            "Anomaly",
        ]
        csv_df: pd.DataFrame = df[csv_cols].copy()
        csv_df["Load_Deviation_Pct"] = csv_df["Load_Deviation_Pct"].round(2)
        csv_df["Settlement_Deviation_Pct"] = csv_df["Settlement_Deviation_Pct"].round(2)
        csv_bytes: bytes = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download sensor data (CSV)",
            data=csv_bytes,
            file_name="sensor_data.csv",
            mime="text/csv",
            key="download_csv",
        )

    with export_col2:
        fig_gauge_report: go.Figure = create_risk_gauge(
            risk_score,
            risk_class,
            cfg["risk_low_threshold"],
            cfg["risk_medium_threshold"],
        )
        gauge_html: str = fig_gauge_report.to_html(
            full_html=False, include_plotlyjs="cdn", config={"responsive": True}
        )
        load_html: str = fig_load.to_html(
            full_html=False, include_plotlyjs=False, config={"responsive": True}
        )
        settlement_html: str = fig_settlement.to_html(
            full_html=False, include_plotlyjs=False, config={"responsive": True}
        )
        report_date: str = datetime.now().strftime("%Y-%m-%d %H:%M")
        kpi_rows: str = (
            f"<tr><td>Risk Level</td><td>{risk_class}</td></tr>"
            f"<tr><td>Anomalies Detected</td><td>{n_anomalies}</td></tr>"
            f"<tr><td>Load Threshold Exceedances</td><td>{load_exceeded}</td></tr>"
            f"<tr><td>Avg Settlement (mm)</td><td>{avg_settlement:.1f}</td></tr>"
        )
        rec_rows: str = "".join(
            f'<tr><td>{a["action"]}</td><td>{a["detail"]}</td></tr>'
            for a in rec_result["actions"]
        )
        report_html: str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construction Monitoring Report</title>
<style>
body {{ font-family: sans-serif; margin: 1rem 2rem; color: #1a1a2e; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
.report-meta {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 1.5rem; }}
h2 {{ font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid #dee2e6; }}
table {{ border-collapse: collapse; width: 100%; max-width: 400px; }}
th, td {{ border: 1px solid #dee2e6; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #f8f9fa; }}
.chart-container {{ margin: 1rem 0; }}
</style>
</head>
<body>
<h1>Construction Monitoring Report</h1>
<p class="report-meta">Generated: {report_date}</p>

<h2>KPIs</h2>
<table><tbody>{kpi_rows}</tbody></table>

<h2>Risk Gauge</h2>
<div class="chart-container">{gauge_html}</div>

<h2>Load Trend</h2>
<div class="chart-container">{load_html}</div>

<h2>Settlement Trend</h2>
<div class="chart-container">{settlement_html}</div>

<h2>Recommendations</h2>
<p><strong>Suggested next check:</strong> {rec_result["suggested_next_check_hrs"]} hrs</p>
<table><thead><tr><th>Action</th><th>Detail</th></tr></thead><tbody>{rec_rows}</tbody></table>

</body>
</html>"""
        st.download_button(
            label="Download one-page HTML report",
            data=report_html.encode("utf-8"),
            file_name="construction_monitoring_report.html",
            mime="text/html",
            key="download_report",
        )


if __name__ == "__main__":
    main()

