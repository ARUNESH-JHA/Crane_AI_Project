"""
Crane AI Predictive Maintenance Platform
==========================================
Enterprise-grade Streamlit dashboard wrapped around an existing, UNCHANGED
LSTM Autoencoder anomaly-detection pipeline.

IMPORTANT: All ML logic below (preprocess_window, scale_and_pad, feature
engineering, windowing, thresholding, root-cause attribution) is copied
verbatim in behaviour from the original app.py. Only presentation/UX code
has been added or changed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import io
from datetime import timedelta, datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Crane AI Predictive Maintenance Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME / STYLE ENGINE
# =============================================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"


def inject_css(theme: str):
    if theme == "dark":
        bg0, bg1 = "#0b0f19", "#121826"
        card_bg = "rgba(255,255,255,0.045)"
        card_border = "rgba(255,255,255,0.09)"
        text_primary = "#f1f3f9"
        text_secondary = "#9aa4b8"
        accent_a, accent_b = "#6366f1", "#22d3ee"
    else:
        bg0, bg1 = "#f4f6fb", "#eef1f8"
        card_bg = "rgba(255,255,255,0.75)"
        card_border = "rgba(15,23,42,0.08)"
        text_primary = "#0f172a"
        text_secondary = "#5b6478"
        accent_a, accent_b = "#4f46e5", "#0ea5e9"

    st.markdown(f"""
    <style>
        .stApp {{
            background: radial-gradient(1200px 800px at 10% -10%, {bg1} 0%, {bg0} 55%);
            color: {text_primary};
        }}
        h1, h2, h3, h4, h5, p, span, label, div {{
            color: {text_primary};
        }}
        [data-testid="stSidebar"] {{
            background: {bg0};
            border-right: 1px solid {card_border};
        }}
        .glass-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.18);
            transition: transform .15s ease, box-shadow .15s ease;
            margin-bottom: 14px;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 36px rgba(0,0,0,0.28);
        }}
        .kpi-title {{
            font-size: 12.5px;
            letter-spacing: .04em;
            text-transform: uppercase;
            color: {text_secondary};
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 800;
            line-height: 1.1;
            background: linear-gradient(90deg, {accent_a}, {accent_b});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: {text_secondary};
            margin-top: 4px;
        }}
        .pill {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .02em;
        }}
        .pill-green {{ background: rgba(34,197,94,.15); color: #22c55e; border:1px solid rgba(34,197,94,.35); }}
        .pill-yellow {{ background: rgba(234,179,8,.15); color: #eab308; border:1px solid rgba(234,179,8,.35); }}
        .pill-orange {{ background: rgba(249,115,22,.15); color: #f97316; border:1px solid rgba(249,115,22,.35); }}
        .pill-red {{ background: rgba(239,68,68,.15); color: #ef4444; border:1px solid rgba(239,68,68,.35); }}
        .pill-blue {{ background: rgba(59,130,246,.15); color: #3b82f6; border:1px solid rgba(59,130,246,.35); }}
        .section-title {{
            font-size: 20px;
            font-weight: 800;
            margin: 6px 0 10px 0;
            display:flex; align-items:center; gap:8px;
        }}
        .subtle {{ color: {text_secondary}; font-size: 13px; }}
        .divider {{ height:1px; background:{card_border}; margin: 18px 0; border:none; }}
        .hero {{
            background: linear-gradient(135deg, {accent_a}22, {accent_b}11);
            border: 1px solid {card_border};
            border-radius: 22px;
            padding: 26px 30px;
            margin-bottom: 18px;
        }}
        .reason-chip {{
            display:inline-flex; align-items:center; gap:6px;
            background:{card_bg}; border:1px solid {card_border};
            padding:6px 12px; border-radius:12px; font-size:13px; margin:4px 6px 4px 0;
        }}
        div[data-testid="stMetricValue"] {{ font-weight: 800; }}
        .stButton>button {{
            border-radius: 12px; font-weight: 700; border: 1px solid {card_border};
        }}
        .stDataFrame {{ border-radius: 14px; overflow: hidden; }}
    </style>
    """, unsafe_allow_html=True)


inject_css(st.session_state["theme"])

PLOT_TEMPLATE = "plotly_dark" if st.session_state["theme"] == "dark" else "plotly_white"
COLOR_HEALTHY = "#22c55e"
COLOR_WARNING = "#eab308"
COLOR_ORANGE = "#f97316"
COLOR_CRITICAL = "#ef4444"
COLOR_INFO = "#3b82f6"
ACCENT = "#6366f1"

# =============================================================================
# BEGINNER-FRIENDLY LABEL MAP  (presentation only — does not touch ML logic)
# =============================================================================
FRIENDLY_LABELS = {
    "Error": "AI Health Score (raw)",
    "Threshold": "Warning Limit",
    "Top Anomaly Contributors": "Main Reasons",
    "Status": "Condition",
}


def friendly(name: str) -> str:
    return FRIENDLY_LABELS.get(name, name)


def severity_from_ratio(ratio: float):
    """ratio = error / threshold. Returns (label, color, icon)."""
    if ratio < 1.0:
        return "Healthy", COLOR_HEALTHY, "✅"
    elif ratio < 1.3:
        return "Minor", COLOR_WARNING, "🟡"
    elif ratio < 1.8:
        return "Moderate", COLOR_ORANGE, "🟠"
    elif ratio < 2.5:
        return "Major", COLOR_CRITICAL, "🔴"
    else:
        return "Critical", "#b91c1c", "🚨"


# =============================================================================
# CACHED MODEL / SCALER LOADING  (unchanged behaviour)
# =============================================================================
@st.cache_resource
def load_ml_model(model_path):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


@st.cache_data
def load_scaling_factors(scaling_path):
    if os.path.exists(scaling_path):
        return joblib.load(scaling_path)
    return None


MODEL_PATH = "Crane_Autoencoder.keras"
SCALING_PATH = "Final_Scaling.pkl"
model = load_ml_model(MODEL_PATH)
final_scaling = load_scaling_factors(SCALING_PATH)

# =============================================================================
# SIDEBAR — NAVIGATION, THEME, CONFIG, UPLOADS
# =============================================================================
with st.sidebar:
    st.markdown("## 🏗️ Crane AI Platform")
    st.caption("Predictive Maintenance · Deep Learning Autoencoder")

    theme_choice = st.radio("Appearance", ["Dark", "Light"],
                             index=0 if st.session_state["theme"] == "dark" else 1,
                             horizontal=True)
    new_theme = theme_choice.lower()
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🧠 AI Insights", "🔬 Sensor Explorer", "📈 Timeline",
         "📐 Statistics", "🔗 Correlation", "📄 Reports", "ℹ️ About Model"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    if model is not None and final_scaling is not None:
        st.markdown('<span class="pill pill-green">● Model Loaded</span>', unsafe_allow_html=True)
        st.caption(f"Expected sensor features: **{model.input_shape[2]}**")
    else:
        st.markdown('<span class="pill pill-red">● Model Missing</span>', unsafe_allow_html=True)
        st.warning("Place 'Crane_Autoencoder.keras' and 'Final_Scaling.pkl' in the app folder.")

    st.markdown("---")
    st.markdown("### 📂 Upload Datasets")
    periodic_file = st.file_uploader("1. Periodic Sensor Data (CSV)", type=["csv"],
                                      help="The main time-series export from the crane PLC.")
    event_file = st.file_uploader("2. Event Log Data (CSV - optional)", type=["csv"],
                                   help="Optional event/alarm log, merged in as extra context.")

    st.markdown("---")
    st.caption(f"🕒 Session time: {datetime.now().strftime('%H:%M:%S')}")

# =============================================================================
# MAIN HERO HEADER
# =============================================================================
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
    <div>
      <div style="font-size:26px; font-weight:800;">🏗️ Crane AI Predictive Maintenance Platform</div>
      <div class="subtle">Deep Learning powered anomaly detection, explained in plain English for plant operators and managers.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# RUN ANALYSIS TRIGGER (visible on Dashboard page, but state persists)
# =============================================================================
if periodic_file is not None:
    periodic_df = pd.read_csv(periodic_file)
    if event_file is not None:
        event_df = pd.read_csv(event_file)
    else:
        event_df = pd.DataFrame(columns=["PLC_TIMESTAMP", "TagName", "TagValue"])

    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info("📊 Dataset loaded. Press **Run Anomaly Diagnostic** to start the Deep Learning analysis.")
        with c2:
            analyze_btn = st.button("🚀 Run Anomaly Diagnostic", type="primary", use_container_width=True)

    if analyze_btn:
        if model is None or final_scaling is None:
            st.error("Cannot run analysis without a trained model and scaling dictionary!")
        else:
            progress_placeholder = st.empty()
            with st.spinner("Decoding PLC tags, merging events, and running Deep Learning predictions..."):

                # ---------------- UNCHANGED PREPROCESSING PIPELINE ----------------
                if "DATA_INGESTION_TIMESTAMP" in periodic_df.columns:
                    periodic_df = periodic_df.drop(columns=["DATA_INGESTION_TIMESTAMP"])
                p_df_clean = periodic_df.dropna(axis=1, how="all").copy()

                if "Record_Type" in event_df.columns:
                    event_df = event_df.drop(columns=["Record_Type"])
                e_df_clean = event_df.dropna(axis=1, how="all").copy()

                for df in [p_df_clean, e_df_clean]:
                    if "PLC_TIMESTAMP" in df.columns:
                        df["PLC_TIMESTAMP"] = pd.to_datetime(df["PLC_TIMESTAMP"], utc=True).dt.tz_localize(None)

                p_df_clean = p_df_clean.sort_values("PLC_TIMESTAMP").reset_index(drop=True)
                e_df_clean = e_df_clean.sort_values("PLC_TIMESTAMP").reset_index(drop=True)

                word_patterns = ["DRV_CMDWRD", "STTSWRD", "FLTWRD1", "FLTWRD2", "ALRMWRD",
                                  "SHORT_CKT_FLTWRD", "CRRNT_UNBAL_FLTWRD"]
                word_columns = [c for c in p_df_clean.columns if any(p in c.upper() for p in word_patterns)]

                if not e_df_clean.empty and "TagName" in e_df_clean.columns:
                    event_pivot = e_df_clean.pivot_table(index="PLC_TIMESTAMP", columns="TagName",
                                                           values="TagValue", aggfunc="last").sort_index()
                    event_pivot = event_pivot.ffill().reset_index()
                else:
                    event_pivot = pd.DataFrame(columns=["PLC_TIMESTAMP"])

                timestamps = p_df_clean["PLC_TIMESTAMP"]
                total_duration_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600.0
                sampling_interval = timestamps.diff().median().total_seconds()

                if total_duration_hours < 0.5:
                    window_seconds = max(10, int(sampling_interval * 10))
                elif total_duration_hours < 5.0:
                    window_seconds = 30
                elif total_duration_hours < 48.0:
                    window_seconds = 60
                else:
                    window_seconds = 300

                window = timedelta(seconds=window_seconds)
                history_size = max(30, int(1800 / window_seconds))

                def preprocess_window(periodic_df_input, event_pivot_input, word_columns_input, window_size_val):
                    bit_frames = []
                    for col in word_columns_input:
                        values = pd.to_numeric(periodic_df_input[col], errors="coerce").fillna(0).astype(np.uint16).to_numpy()
                        decoded = pd.DataFrame({
                            f"{col}_BIT_{bit}": ((values >> bit) & 1).astype(np.uint8)
                            for bit in range(16)
                        }, index=periodic_df_input.index)
                        bit_frames.append(decoded)
                    if bit_frames:
                        df_p = pd.concat([periodic_df_input] + bit_frames, axis=1)
                    else:
                        df_p = periodic_df_input.copy()
                    df_p.drop(columns=word_columns_input, inplace=True, errors="ignore")

                    common_cols = list(set(df_p.columns).intersection(set(event_pivot_input.columns)))
                    if "PLC_TIMESTAMP" in common_cols:
                        common_cols.remove("PLC_TIMESTAMP")

                    if not event_pivot_input.empty and len(event_pivot_input.columns) > 1:
                        e_ready = event_pivot_input.drop(columns=common_cols, errors="ignore")
                        merged = pd.merge_asof(
                            df_p.sort_values("PLC_TIMESTAMP"),
                            e_ready.sort_values("PLC_TIMESTAMP"),
                            on="PLC_TIMESTAMP",
                            direction="backward",
                            allow_exact_matches=True
                        )
                    else:
                        merged = df_p.copy()

                    merged["MH11_MH12_CURR_DIFF"] = (merged["MH11_MOT_CRRNT"] - merged["MH12_MOT_CRRNT"]).abs()
                    merged["MH22_MH21_CURR_DIFF"] = (merged["MH22_MOT_CRRNT"] - merged["MH21_MOT_CRRNT"]).abs()
                    merged["MH11_MH12_VOLT_DIFF"] = (merged["MH11_MOT_VOLT"] - merged["MH12_MOT_VOLT"]).abs()
                    merged["MH22_MH21_VOLT_DIFF"] = (merged["MH22_MOT_VOLT"] - merged["MH21_MOT_VOLT"]).abs()
                    merged["MH11_MH12_SPEED_DIFF"] = (merged["MH11_SPD_REF_3"] - merged["MH12_SPD_REF_3"]).abs()
                    merged["MH22_MH21_SPEED_DIFF"] = (merged["MH22_SPD_REF_3"] - merged["MH21_SPD_REF_3"]).abs()

                    if "ISU1_DC_VOLT" in merged.columns and "ISU2_DC_VOLT" in merged.columns and "ISU3_DC_VOLT" in merged.columns:
                        merged["ISU1_ISU2_VOLT_DIFF"] = (merged["ISU1_DC_VOLT"] - merged["ISU2_DC_VOLT"] - merged["ISU3_DC_VOLT"]).abs()
                    if "ISU_1_2_SEL_RQST" == 1 in merged.columns:
                        merged["ALL_ISU_VOLT_STD"] = merged[["ISU1_DC_VOLT", "ISU2_DC_VOLT"]].std(axis=1)
                    if "ISU_1_3_SEL_RQST" == 1 in merged.columns:
                        merged["ALL_ISU_VOLT_STD"] = merged[["ISU1_DC_VOLT", "ISU3_DC_VOLT"]].std(axis=1)
                    if "ISU_2_3_SEL_RQST" == 1 in merged.columns:
                        merged["ALL_ISU_VOLT_STD"] = merged[["ISU2_DC_VOLT", "ISU3_DC_VOLT"]].std(axis=1)

                    if "IDLR_VIB_SNSR_1" in merged.columns:
                        merged["IDLR_VIB_1_GRAD"] = merged["IDLR_VIB_SNSR_1"].diff().fillna(0)
                        merged["IDLR_VIB_1_ROLL_STD"] = merged["IDLR_VIB_SNSR_1"].rolling(5, min_periods=1).std().fillna(0)
                    if "IDLR_VIB_SNSR_2" in merged.columns:
                        merged["IDLR_VIB_2_GRAD"] = merged["IDLR_VIB_SNSR_2"].diff().fillna(0)
                        merged["IDLR_VIB_2_ROLL_STD"] = merged["IDLR_VIB_SNSR_2"].rolling(5, min_periods=1).std().fillna(0)

                    motor_current = ["MH11_MOT_CRRNT", "MH12_MOT_CRRNT", "MH21_MOT_CRRNT", "MH22_MOT_CRRNT"]
                    merged["MH_MOTOR_CURR_STD"] = merged[motor_current].std(axis=1)

                    motor_voltage = ["MH11_DC_VOLT", "MH12_DC_VOLT", "MH21_DC_VOLT", "MH22_DC_VOLT"]
                    merged["MH_MOTOR_VOLT_STD"] = merged[motor_voltage].std(axis=1)

                    motor_speed = ["MH11_SPD_REF_3", "MH12_SPD_REF_3", "MH21_SPD_REF_3", "MH22_SPD_REF_3"]
                    merged["MH_MOTOR_SPEED_STD"] = merged[motor_speed].std(axis=1)

                    for col in motor_current:
                        merged[col + "_GRAD"] = merged[col].diff().fillna(0)
                    for col in motor_voltage:
                        merged[col + "_GRAD"] = merged[col].diff().fillna(0)
                    for col in motor_current:
                        merged[col + "_ROLL_MEAN"] = merged[col].rolling(5, min_periods=1).mean()
                    for col in motor_current:
                        merged[col + "_ROLL_STD"] = merged[col].rolling(5, min_periods=1).std().fillna(0)

                    start_time_win = merged["PLC_TIMESTAMP"].min()
                    merged["WINDOW_TIME"] = (merged["PLC_TIMESTAMP"] - start_time_win).dt.total_seconds()
                    merged["DELTA_TIME"] = merged["PLC_TIMESTAMP"].diff().dt.total_seconds().fillna(0)

                    cols = [c for c in merged.columns if c != "PLC_TIMESTAMP"]
                    return merged, cols

                expected_features = list(final_scaling.keys())

                def scale_and_pad(df):
                    X = df.drop(columns=["PLC_TIMESTAMP"]).reindex(columns=expected_features, fill_value=0)
                    for col in expected_features:
                        if final_scaling[col]["type"] == "continuous":
                            mean = final_scaling[col]["mean"]
                            std = final_scaling[col]["std"]
                            X[col] = (X[col] - mean) / std
                    X = X.fillna(0)
                    target_len = 60
                    current_len = len(X)
                    if current_len < target_len:
                        pad_size = target_len - current_len
                        last_row = X.iloc[[-1]]
                        X = pd.concat([X] + [last_row] * pad_size, ignore_index=True)
                    elif current_len > target_len:
                        X = X.head(target_len)
                    return X

                start_time = p_df_clean["PLC_TIMESTAMP"].min()
                end_time = p_df_clean["PLC_TIMESTAMP"].max()
                current_start = start_time

                window_no = 1
                window_errors = []
                error_history = []
                anomaly_log = []
                raw_windows = {}
                feature_mse_log = {}

                expected_dim = model.input_shape[2]
                progress_bar = progress_placeholder.progress(0.0, text="Starting diagnostic sweep...")

                total_windows = int((end_time - start_time) / window) + 1

                while current_start <= end_time:
                    current_end = current_start + window
                    periodic_60s = p_df_clean[(p_df_clean["PLC_TIMESTAMP"] >= current_start) &
                                               (p_df_clean["PLC_TIMESTAMP"] < current_end)].copy()

                    if periodic_60s.empty:
                        current_start += window
                        window_no += 1
                        continue

                    w_df, _ = preprocess_window(periodic_60s, event_pivot, word_columns, window)
                    X_scaled = scale_and_pad(w_df)

                    raw_windows[window_no] = w_df.copy()

                    X_seq = X_scaled.to_numpy(dtype=np.float32)
                    X_seq_trimmed = X_seq[:, :expected_dim]
                    X_seq_trimmed_batch = np.expand_dims(X_seq_trimmed, axis=0)

                    pred_batch = model.predict(X_seq_trimmed_batch, verbose=0)
                    pred_seq = pred_batch[0]

                    window_error = np.mean(np.square(X_seq_trimmed - pred_seq))
                    window_errors.append(window_error)
                    error_history.append(window_error)

                    if len(error_history) < history_size:
                        threshold = np.mean(error_history) + 3 * np.std(error_history)
                    else:
                        recent = error_history[-history_size:]
                        threshold = np.mean(recent) + 3 * np.std(recent)

                    status = "HEALTHY"
                    top_causes_str = ""
                    causes_list_struct = []

                    squared_diff = np.square(X_seq_trimmed - pred_seq)
                    feature_mse = np.mean(squared_diff, axis=0)
                    feature_mse_log[window_no] = feature_mse

                    if window_error > threshold:
                        status = "ANOMALY"
                        top_indices = np.argsort(feature_mse)[::-1][:3]
                        total_mse_sum = np.sum(feature_mse)

                        causes_list = []
                        for idx in top_indices:
                            feat_name = expected_features[idx]
                            contribution = (feature_mse[idx] / total_mse_sum) * 100 if total_mse_sum > 0 else 0
                            causes_list.append(f"{feat_name} ({contribution:.1f}%)")
                            causes_list_struct.append((feat_name, contribution))
                        top_causes_str = ", ".join(causes_list)

                    anomaly_log.append({
                        "Window": window_no,
                        "Start Timestamp": current_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "End Timestamp": current_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "Error": float(window_error),
                        "Threshold": float(threshold),
                        "Status": status,
                        "Top Anomaly Contributors": top_causes_str,
                        "_causes_struct": causes_list_struct,
                    })

                    prog = min(1.0, window_no / total_windows)
                    progress_bar.progress(prog, text=f"Analyzing window {window_no} / {total_windows}...")

                    current_start += window
                    window_no += 1

                progress_bar.progress(1.0, text="Done.")
                df_results = pd.DataFrame(anomaly_log)
                st.session_state["df_results"] = df_results
                st.session_state["raw_windows"] = raw_windows
                st.session_state["feature_mse_log"] = feature_mse_log
                st.session_state["expected_features"] = expected_features
                st.session_state["window_seconds"] = window_seconds
                st.session_state["history_size"] = history_size
                st.session_state["sampling_interval"] = sampling_interval
                st.session_state["total_duration_hours"] = total_duration_hours
                st.session_state["n_sensors"] = len([c for c in p_df_clean.columns if c != "PLC_TIMESTAMP"])
                st.session_state["model_input_dim"] = expected_dim

            progress_placeholder.empty()
            st.success("✅ Analysis Complete!")
            st.balloons()

# =============================================================================
# HELPER: KPI CARD
# =============================================================================
def kpi_card(title, value, sub="", icon="", spark_values=None, color=ACCENT):
    spark_svg = ""
    if spark_values is not None and len(spark_values) > 1:
        vals = np.array(spark_values, dtype=float)
        vmin, vmax = vals.min(), vals.max()
        rng = (vmax - vmin) or 1.0
        pts = " ".join(f"{i * (100 / (len(vals) - 1)):.1f},{28 - ((v - vmin) / rng) * 26:.1f}"
                        for i, v in enumerate(vals))
        spark_svg = f"""<svg width="100%" height="28" viewBox="0 0 100 28" preserveAspectRatio="none">
            <polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}" opacity="0.85"/>
        </svg>"""
    st.markdown(f"""
    <div class="glass-card">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {spark_svg}
    </div>
    """, unsafe_allow_html=True)


def status_pill(status_word: str):
    mapping = {
        "Healthy": "pill-green", "Low": "pill-green",
        "Watch": "pill-yellow", "Minor": "pill-yellow", "Medium": "pill-yellow",
        "Warning": "pill-orange", "Moderate": "pill-orange", "High": "pill-orange",
        "Critical": "pill-red", "Major": "pill-red",
        "Information": "pill-blue",
    }
    cls = mapping.get(status_word, "pill-blue")
    return f'<span class="pill {cls}">{status_word}</span>'


def plotly_gauge(value, title, color, suffix="%", max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 34}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.5], "color": "rgba(239,68,68,0.18)"},
                {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(234,179,8,0.18)"},
                {"range": [max_val * 0.8, max_val], "color": "rgba(34,197,94,0.18)"},
            ],
        },
    ))
    text_color = "#f1f3f9" if st.session_state["theme"] == "dark" else "#0f172a"
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font_color=text_color, template=PLOT_TEMPLATE)
    return fig


def plotly_risk_meter(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": " / 100", "font": {"size": 26}},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#111827", "thickness": 0.001},
            "steps": [
                {"range": [0, 25], "color": COLOR_HEALTHY},
                {"range": [25, 50], "color": COLOR_WARNING},
                {"range": [50, 75], "color": COLOR_ORANGE},
                {"range": [75, 100], "color": COLOR_CRITICAL},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.9, "value": risk_score},
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", template=PLOT_TEMPLATE)
    return fig


# =============================================================================
# LOAD RESULTS (persisted in session state)
# =============================================================================
has_results = "df_results" in st.session_state

if has_results:
    df_results = st.session_state["df_results"]
    raw_windows = st.session_state["raw_windows"]
    feature_mse_log = st.session_state.get("feature_mse_log", {})
    expected_features = st.session_state.get("expected_features", [])
    window_seconds = st.session_state.get("window_seconds", 60)
    history_size = st.session_state.get("history_size", 30)
    sampling_interval = st.session_state.get("sampling_interval", np.nan)
    total_duration_hours = st.session_state.get("total_duration_hours", np.nan)
    n_sensors = st.session_state.get("n_sensors", np.nan)

    total_w = len(df_results)
    anomalous_w = int((df_results["Status"] == "ANOMALY").sum())
    healthy_w = total_w - anomalous_w
    peak_err = df_results["Error"].max()
    avg_err = df_results["Error"].mean()
    health_score = max(0, min(100, 100 * (1 - anomalous_w / total_w))) if total_w else 100
    risk_score = max(0, min(100, 100 * (anomalous_w / total_w) * 1.4)) if total_w else 0

    if risk_score < 25:
        risk_label, risk_color = "Low", COLOR_HEALTHY
    elif risk_score < 50:
        risk_label, risk_color = "Medium", COLOR_WARNING
    elif risk_score < 75:
        risk_label, risk_color = "High", COLOR_ORANGE
    else:
        risk_label, risk_color = "Critical", COLOR_CRITICAL

    if health_score >= 90:
        overall_status, status_color = "Healthy", COLOR_HEALTHY
    elif health_score >= 75:
        overall_status, status_color = "Warning", COLOR_WARNING
    else:
        overall_status, status_color = "Critical", COLOR_CRITICAL

    model_confidence = max(0, min(100, 100 - (avg_err / (df_results["Threshold"].mean() or 1)) * 40))

    df_results["Ratio"] = df_results["Error"] / df_results["Threshold"].replace(0, np.nan)
    df_results["Severity"], df_results["SeverityColor"], df_results["SeverityIcon"] = zip(
        *df_results["Ratio"].apply(lambda r: severity_from_ratio(r if pd.notna(r) else 0)))
else:
    df_results = None

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    if not has_results:
        st.info("👈 Upload a periodic sensor CSV file in the sidebar and run the diagnostic to populate the dashboard.")
    else:
        st.markdown('<div class="section-title">🚀 Executive Overview</div>', unsafe_allow_html=True)

        r1 = st.columns(4)
        with r1[0]:
            kpi_card("Total Windows Analysed", f"{total_w:,}", "Across full upload", "🧮",
                     spark_values=df_results["Error"].tolist())
        with r1[1]:
            kpi_card("Healthy Windows", f"{healthy_w:,}", f"{(healthy_w/total_w*100):.1f}% of total", "✅",
                     color=COLOR_HEALTHY)
        with r1[2]:
            kpi_card("Anomalous Windows", f"{anomalous_w:,}", f"{(anomalous_w/total_w*100):.1f}% of total", "⚠️",
                     color=COLOR_CRITICAL)
        with r1[3]:
            kpi_card("Overall Health Score", f"{health_score:.0f}", status_pill(overall_status), "💚",
                     color=status_color)

        r2 = st.columns(4)
        with r2[0]:
            kpi_card("Max AI Health Deviation", f"{peak_err:.4f}", "Highest recorded window error", "📈")
        with r2[1]:
            kpi_card("Avg AI Health Deviation", f"{avg_err:.4f}", "Mean across all windows", "📊")
        with r2[2]:
            kpi_card("Risk Level", risk_label, status_pill(risk_label), "🧭", color=risk_color)
        with r2[3]:
            kpi_card("Model Confidence", f"{model_confidence:.0f}%", "Inverse of relative error", "🤖")

        r3 = st.columns(4)
        with r3[0]:
            kpi_card("Data Duration", f"{total_duration_hours:.2f} hrs", "Full upload span", "⏱️")
        with r3[1]:
            kpi_card("Sampling Frequency", f"{sampling_interval:.1f} s", "Median PLC interval", "📡")
        with r3[2]:
            kpi_card("Sensors Monitored", f"{n_sensors}", "Raw PLC tag columns", "🛰️")
        with r3[3]:
            kpi_card("Window Size", f"{window_seconds}s", f"History: {history_size} windows", "🪟")

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        gc1, gc2, gc3 = st.columns([1, 1, 1.4])
        with gc1:
            st.markdown("**Machine Health**")
            st.plotly_chart(plotly_gauge(health_score, "Health Score", status_color), use_container_width=True,
                             config={"displayModeBar": False})
        with gc2:
            st.markdown("**Overall Risk Meter**")
            st.plotly_chart(plotly_risk_meter(risk_score), use_container_width=True,
                             config={"displayModeBar": False})
        with gc3:
            st.markdown("**Current Status**")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding-top:34px; padding-bottom:34px;">
                <div style="font-size:46px;">{'🟢' if overall_status=='Healthy' else '🟡' if overall_status=='Warning' else '🔴'}</div>
                <div style="font-size:22px; font-weight:800; margin-top:6px;">{overall_status}</div>
                <div class="subtle">Based on {anomalous_w} of {total_w} windows flagged</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎨 Machine Health Timeline</div>', unsafe_allow_html=True)
        st.caption("Click / hover any segment below to inspect that window's condition.")

        color_map = {"HEALTHY": COLOR_HEALTHY, "ANOMALY": COLOR_CRITICAL}
        fig_tl = px.bar(
            df_results, x="Window", y=[1] * len(df_results), color="Status",
            color_discrete_map=color_map, hover_data={"Start Timestamp": True, "Error": ":.4f", "Status": True},
        )
        fig_tl.update_layout(showlegend=True, height=110, template=PLOT_TEMPLATE,
                              margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(visible=False),
                              xaxis_title="Window #", bargap=0)
        st.plotly_chart(fig_tl, use_container_width=True)

# =============================================================================
# PAGE: AI INSIGHTS
# =============================================================================
elif page == "🧠 AI Insights":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to see AI insights.")
    else:
        st.markdown('<div class="section-title">🧠 Why did the AI detect this anomaly?</div>', unsafe_allow_html=True)
        df_anom = df_results[df_results["Status"] == "ANOMALY"].copy()

        if df_anom.empty:
            st.success("🎉 No anomalies detected — nothing to explain. The crane looks completely healthy.")
        else:
            sel_win = st.selectbox(
                "Choose an anomaly window to explain in plain English:",
                options=df_anom["Window"].tolist(),
                format_func=lambda w: f"Window {w} — {df_results.loc[df_results['Window']==w,'Start Timestamp'].values[0]}",
            )
            row = df_results[df_results["Window"] == sel_win].iloc[0]
            severity, sev_color, sev_icon = severity_from_ratio(row["Ratio"] if pd.notna(row["Ratio"]) else 0)

            top_left, top_right = st.columns([2, 1])
            with top_left:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="kpi-title">Plain-English Explanation</div>
                    <p style="font-size:15px; line-height:1.7;">
                    The AI compared this window against normal operating history and found it behaves differently.
                    The main contributors, in order of importance, are shown below — these are the sensor readings
                    that moved the furthest away from what the AI expects during healthy operation.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                causes = row["_causes_struct"] if "_causes_struct" in row and isinstance(row["_causes_struct"], list) else []
                if causes:
                    chips = "".join(
                        f'<span class="reason-chip">🔧 {name.replace("_", " ").title()} — {pct:.1f}%</span>'
                        for name, pct in causes
                    )
                    st.markdown(f'<div style="margin-top:10px;">{chips}</div>', unsafe_allow_html=True)

                    bar_df = pd.DataFrame(causes, columns=["Sensor", "Contribution"])
                    bar_df["Sensor"] = bar_df["Sensor"].str.replace("_", " ").str.title()
                    fig_bar = px.bar(bar_df.sort_values("Contribution"), x="Contribution", y="Sensor",
                                      orientation="h", text="Contribution",
                                      color="Contribution", color_continuous_scale=["#eab308", "#ef4444"])
                    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_bar.update_layout(template=PLOT_TEMPLATE, height=280, coloraxis_showscale=False,
                                           margin=dict(l=10, r=30, t=20, b=10), xaxis_title="Contribution (%)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No structured cause breakdown available for this window.")

            with top_right:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="kpi-title">Severity</div>
                    <div style="font-size:40px; margin:6px 0;">{sev_icon}</div>
                    <div style="font-size:20px; font-weight:800; color:{sev_color};">{severity}</div>
                    <div class="subtle" style="margin-top:6px;">Error is {row['Ratio']:.2f}× the warning limit</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card">
                    <div class="kpi-title">🛠️ Maintenance Recommendation</div>
                    <p style="font-size:13.5px; line-height:1.6;">
                    <b>Likely impact:</b> Elevated stress on the flagged component(s).<br>
                    <b>Recommended inspection:</b> Visually inspect and test the top contributing sensors listed.<br>
                    <b>Priority:</b> {severity}<br>
                    <b>Suggested action window:</b> {"Immediate" if severity in ["Major","Critical"] else "Next scheduled maintenance"}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
            st.markdown("**Culprit sensor raw trends during this window:**")
            raw_w_df = raw_windows[sel_win]
            contributors_list = [c[0] for c in causes]
            cols_to_plot = [c for c in contributors_list if c in raw_w_df.columns]
            if cols_to_plot:
                plot_df = raw_w_df.copy()
                plot_df["Time (s)"] = range(1, len(plot_df) + 1)
                fig_line = px.line(plot_df, x="Time (s)", y=cols_to_plot, template=PLOT_TEMPLATE)
                fig_line.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                                        legend_title_text="Sensor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No raw trend plot available for the custom engineered features in this window.")

# =============================================================================
# PAGE: SENSOR EXPLORER
# =============================================================================
elif page == "🔬 Sensor Explorer":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to explore sensors.")
    else:
        st.markdown('<div class="section-title">🔬 Sensor Explorer</div>', unsafe_allow_html=True)
        win_options = sorted(raw_windows.keys())
        sel_w = st.selectbox("Select a window to explore:", win_options,
                              format_func=lambda w: f"Window {w} ({df_results.loc[df_results['Window']==w,'Status'].values[0]})")
        raw_w_df = raw_windows[sel_w]
        numeric_cols = [c for c in raw_w_df.select_dtypes(include=[np.number]).columns
                        if c not in ["WINDOW_TIME", "DELTA_TIME"]]
        sensor = st.selectbox("Select sensor:", numeric_cols)

        series = raw_w_df[sensor].reset_index(drop=True)
        t = np.arange(1, len(series) + 1)
        roll_mean = series.rolling(5, min_periods=1).mean()
        roll_std = series.rolling(5, min_periods=1).std().fillna(0)
        grad = series.diff().fillna(0)
        mean_v, std_v = series.mean(), series.std()
        lower, upper = mean_v - 3 * std_v, mean_v + 3 * std_v
        anomaly_mask = (series < lower) | (series > upper)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Min", f"{series.min():.3f}")
        m2.metric("Max", f"{series.max():.3f}")
        m3.metric("Average", f"{series.mean():.3f}")
        m4.metric("Std Dev", f"{series.std():.3f}")
        trend = "▲ Rising" if series.iloc[-1] > series.iloc[0] else "▼ Falling" if series.iloc[-1] < series.iloc[0] else "→ Flat"
        m5.metric("Trend", trend)

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=t, y=series, mode="lines", name="Raw signal", line=dict(color=ACCENT)))
        fig.add_trace(go.Scatter(x=t, y=roll_mean, mode="lines", name="Moving average",
                                  line=dict(color=COLOR_INFO, dash="dot")))
        fig.add_hrect(y0=lower, y1=upper, fillcolor=COLOR_HEALTHY, opacity=0.08, line_width=0,
                      annotation_text="Normal range", annotation_position="top left")
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(x=t[anomaly_mask], y=series[anomaly_mask], mode="markers",
                                      name="Anomaly marker", marker=dict(color=COLOR_CRITICAL, size=9, symbol="x")))
        fig.update_layout(template=PLOT_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10),
                           title=f"{sensor.replace('_',' ').title()} — Raw Signal & Moving Average")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_std = px.line(x=t, y=roll_std, template=PLOT_TEMPLATE, labels={"x": "Time (s)", "y": "Rolling Std"})
            fig_std.update_layout(height=260, title="Rolling Std Dev", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_std, use_container_width=True)
        with c2:
            fig_grad = px.line(x=t, y=grad, template=PLOT_TEMPLATE, labels={"x": "Time (s)", "y": "Gradient"})
            fig_grad.update_layout(height=260, title="Gradient (rate of change)", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_grad, use_container_width=True)

# =============================================================================
# PAGE: TIMELINE
# =============================================================================
elif page == "📈 Timeline":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to view the timeline.")
    else:
        st.markdown('<div class="section-title">📈 AI Health Score vs Warning Limit</div>', unsafe_allow_html=True)
        st.caption("Zoom, pan, and use the range slider below. Red points are anomalous windows.")

        df_plot = df_results.copy()
        df_plot["Start Timestamp"] = pd.to_datetime(df_plot["Start Timestamp"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Start Timestamp"], y=df_plot["Error"], mode="lines+markers",
                                  name="AI Health Score", line=dict(color=ACCENT, width=2),
                                  marker=dict(size=5,
                                              color=[COLOR_CRITICAL if s == "ANOMALY" else ACCENT for s in df_plot["Status"]])))
        fig.add_trace(go.Scatter(x=df_plot["Start Timestamp"], y=df_plot["Threshold"], mode="lines",
                                  name="Warning Limit", line=dict(color=COLOR_WARNING, width=2, dash="dash")))

        anomalies = df_plot[df_plot["Status"] == "ANOMALY"]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies["Start Timestamp"], y=anomalies["Error"], mode="markers",
                                      name="Anomaly", marker=dict(color=COLOR_CRITICAL, size=10, symbol="triangle-up")))

        fig.update_layout(
            template=PLOT_TEMPLATE, height=520, hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(rangeslider=dict(visible=True), showspikes=True, spikemode="across"),
            yaxis=dict(title="AI Health Score (raw error)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False,
                         "modeBarButtonsToAdd": ["drawline", "eraseshape"]})

        st.download_button("⬇️ Export timeline data (CSV)",
                            df_plot.drop(columns=["_causes_struct"], errors="ignore").to_csv(index=False),
                            file_name="timeline_data.csv", mime="text/csv")

# =============================================================================
# PAGE: STATISTICS
# =============================================================================
elif page == "📐 Statistics":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to view statistics.")
    else:
        st.markdown('<div class="section-title">📐 Statistical Summary</div>', unsafe_allow_html=True)

        all_raw = pd.concat(raw_windows.values(), ignore_index=True)
        numeric_cols = [c for c in all_raw.select_dtypes(include=[np.number]).columns
                        if c not in ["WINDOW_TIME", "DELTA_TIME"]]
        sel_feats = st.multiselect("Select features to summarize:", numeric_cols,
                                    default=numeric_cols[: min(6, len(numeric_cols))])

        if sel_feats:
            summary_rows = []
            for f in sel_feats:
                s = all_raw[f].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                summary_rows.append({
                    "Feature": f, "Mean": s.mean(), "Median": s.median(),
                    "Mode": s.mode().iloc[0] if not s.mode().empty else np.nan,
                    "Variance": s.var(), "Std": s.std(), "IQR": q3 - q1,
                    "P10": s.quantile(0.10), "P90": s.quantile(0.90),
                    "Skewness": s.skew(), "Kurtosis": s.kurtosis(),
                    "Missing": int(all_raw[f].isna().sum()),
                    "Outliers (3σ)": int(((s - s.mean()).abs() > 3 * s.std()).sum()),
                })
            st.dataframe(pd.DataFrame(summary_rows).round(4), use_container_width=True)

            st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
            st.markdown("**Distribution Analysis**")
            feat_dist = st.selectbox("Feature to visualize:", sel_feats)
            d1, d2 = st.columns(2)
            with d1:
                fig_h = px.histogram(all_raw, x=feat_dist, marginal="violin", template=PLOT_TEMPLATE,
                                      color_discrete_sequence=[ACCENT])
                fig_h.update_layout(height=340, title="Histogram + Violin", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_h, use_container_width=True)
            with d2:
                fig_b = px.box(all_raw, y=feat_dist, template=PLOT_TEMPLATE, points="outliers",
                                color_discrete_sequence=[COLOR_INFO])
                fig_b.update_layout(height=340, title="Box Plot (outliers marked)", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Select at least one feature above.")

# =============================================================================
# PAGE: CORRELATION
# =============================================================================
elif page == "🔗 Correlation":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to view correlations.")
    else:
        st.markdown('<div class="section-title">🔗 Interactive Correlation Matrix</div>', unsafe_allow_html=True)
        all_raw = pd.concat(raw_windows.values(), ignore_index=True)
        numeric_cols = [c for c in all_raw.select_dtypes(include=[np.number]).columns
                        if c not in ["WINDOW_TIME", "DELTA_TIME"]]
        default_sel = numeric_cols[: min(12, len(numeric_cols))]
        sel = st.multiselect("Choose sensors to include:", numeric_cols, default=default_sel)

        if len(sel) >= 2:
            corr = all_raw[sel].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             template=PLOT_TEMPLATE, aspect="auto")
            fig.update_layout(height=min(700, 80 + 40 * len(sel)), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            pairs = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
            pairs = pairs[pairs.index.map(lambda x: x[0] < x[1])].sort_values(key=abs, ascending=False)
            st.markdown("**Strongest correlations**")
            top_pairs = pairs.head(8).reset_index()
            top_pairs.columns = ["Sensor A", "Sensor B", "Correlation"]
            st.dataframe(top_pairs.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Select at least two sensors to build the matrix.")

# =============================================================================
# PAGE: REPORTS
# =============================================================================
elif page == "📄 Reports":
    if not has_results:
        st.info("Run the diagnostic first from the sidebar to generate reports.")
    else:
        st.markdown('<div class="section-title">📄 AI Generated Executive Report</div>', unsafe_allow_html=True)

        df_anom = df_results[df_results["Status"] == "ANOMALY"]
        if not df_anom.empty:
            top_reason_series = df_anom["_causes_struct"].apply(lambda cs: cs[0][0] if cs else None).dropna()
            main_issue = top_reason_series.mode().iloc[0].replace("_", " ").title() if not top_reason_series.empty else "N/A"
        else:
            main_issue = "None detected"

        overall_priority = "Low" if anomalous_w == 0 else "Medium" if anomalous_w / total_w < 0.1 else "High"

        report_text = f"""Machine Health: {overall_status.upper()}
Overall Risk: {risk_label.upper()}
Detected anomalies: {anomalous_w}
Main issue: {main_issue}
Recommendation: {"No action needed — continue normal operation." if anomalous_w == 0 else f"Inspect components related to {main_issue}."}
Priority: {overall_priority}
Estimated urgency: {"Routine monitoring" if anomalous_w == 0 else "Within next maintenance cycle" if overall_priority != "High" else "Immediate attention recommended"}
"""

        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="kpi-title">Executive Summary</div>
                <pre style="white-space: pre-wrap; font-family: inherit; font-size:14.5px; line-height:1.9;">{report_text}</pre>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Anomaly log (searchable / sortable):**")
            search = st.text_input("🔎 Search anomaly log", "")
            df_view = df_results.drop(columns=["_causes_struct"], errors="ignore")
            if search:
                mask = df_view.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
                df_view = df_view[mask]
            st.dataframe(
                df_view.rename(columns=friendly).style.apply(
                    lambda row: [f'background-color: rgba(239,68,68,0.12)' if row.get("Condition") == "ANOMALY"
                                 else '' for _ in row], axis=1
                ) if "Condition" in df_view.rename(columns=friendly).columns else df_view,
                use_container_width=True, height=320
            )

        with rc2:
            st.markdown("**Export options**")
            csv_bytes = df_results.drop(columns=["_causes_struct"], errors="ignore").to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv_bytes, "anomaly_report.csv", "text/csv",
                                use_container_width=True)

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                df_results.drop(columns=["_causes_struct"], errors="ignore").to_excel(writer, index=False, sheet_name="Anomaly Log")
            st.download_button("⬇️ Download Excel", excel_buf.getvalue(), "anomaly_report.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

            st.download_button("⬇️ Download Executive Summary (TXT)", report_text, "executive_summary.txt",
                                "text/plain", use_container_width=True)

            st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
            st.markdown("**Severity breakdown**")
            sev_counts = df_results["Severity"].value_counts()
            fig_sev = px.pie(values=sev_counts.values, names=sev_counts.index, hole=0.55,
                              color=sev_counts.index,
                              color_discrete_map={"Healthy": COLOR_HEALTHY, "Minor": COLOR_WARNING,
                                                   "Moderate": COLOR_ORANGE, "Major": COLOR_CRITICAL,
                                                   "Critical": "#b91c1c"})
            fig_sev.update_layout(template=PLOT_TEMPLATE, height=280, margin=dict(l=10, r=10, t=10, b=10),
                                   showlegend=True)
            st.plotly_chart(fig_sev, use_container_width=True)

# =============================================================================
# PAGE: ABOUT MODEL
# =============================================================================
elif page == "ℹ️ About Model":
    st.markdown('<div class="section-title">ℹ️ About This Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card">
        <p style="font-size:14.5px; line-height:1.8;">
        This platform uses an <b>LSTM Autoencoder</b> trained on historical healthy crane operating data.
        The model learns to reconstruct normal sensor behaviour; when a new window of readings is hard for
        the model to reconstruct, the <b>reconstruction error</b> (shown here as the <i>AI Health Score</i>)
        rises above the adaptive <i>Warning Limit</i>, and the window is flagged for review.
        </p>
        <ul style="font-size:14px; line-height:1.9;">
            <li><b>Warning Limit</b>: a rolling mean + 3 standard deviations of recent error history — adapts as the crane's normal operating envelope shifts.</li>
            <li><b>Main Reasons</b>: for flagged windows, the sensors with the highest individual reconstruction error are surfaced as the likely root cause.</li>
            <li><b>Windowing</b>: sensor data is grouped into fixed-length time windows before being fed to the model, sized automatically based on total data duration.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if model is not None:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-title">Model Input Shape</div>
            <div class="kpi-value" style="font-size:20px;">{model.input_shape}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Show full model summary"):
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            st.code("\n".join(summary_lines))
    else:
        st.warning("Model not currently loaded — place the .keras and .pkl files in the app folder.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.caption("Crane AI Predictive Maintenance Platform · Deep Learning Autoencoder · Built for plant supervisors, engineers, and executives.")
