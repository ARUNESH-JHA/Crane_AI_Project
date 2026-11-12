# 🏗️ CraneGuard AI — Predictive Maintenance for Industrial Cranes

Real-time anomaly detection system for overhead cranes, built around a **CNN-LSTM Attention Autoencoder** that learns normal operating behaviour from PLC sensor telemetry and flags deviations before they escalate into mechanical or electrical failures.

Tested and validated on industrial telemetry from a production crane (IJ1 hoist group), monitoring 600+ PLC tags across hoist motors, inverters, vibration sensors, and drive command words.

---

## What it does

| Capability | Detail |
|---|---|
| **Anomaly detection** | Sliding-window reconstruction error evaluated against an adaptive 3-σ rolling threshold that shifts with normal operating envelope changes. |
| **Root-cause attribution** | Per-feature MSE breakdown isolates the top 3 contributing sensor tags for each anomaly — pinpointing the exact motor or drive component experiencing stress. |
| **Severity grading** | Classifies each flagged window into **Minor**, **Moderate**, **Major**, or **Critical** based on reconstruction error magnitude relative to threshold. |
| **Interactive dashboard** | 8-page Streamlit operations interface: executive KPIs, health gauges, risk meter, timeline heatmaps, individual sensor diagnostics, correlation matrix, statistical distributions, and automated executive report exports (CSV / Excel / TXT). |

---

## Architecture

```
PLC Sensor Stream / CSV
        │
        ▼
┌──────────────────────────────┐
│  Preprocessing Pipeline      │
│  • PLC word → 16-bit decode  │
│  • Event log merge (asof)    │
│  • Feature engineering       │
│    - Motor power (V × I)     │
│    - Master/follower ratio   │
│    - Rolling range (P2P)     │
│    - EMA smoothing           │
│    - Bit duty-cycle %        │
│  • StandardScaler (cached)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  CNN-LSTM Attention AE       │
│  Conv1D → BatchNorm → Pool   │
│  LSTM(128) → MultiHeadAttn   │
│  LSTM(64) → Dense(32) latent │
│  RepeatVector → LSTM decode  │
│  UpSample → TimeDistributed  │
└──────────┬───────────────────┘
           │
           ▼
   Reconstruction Error (MSE)
     vs adaptive threshold
           │
    ┌──────┴──────┐
    │             │
 HEALTHY      ANOMALY
              → root cause breakdown
              → severity grading
              → maintenance action
```

---

## Project structure

```
Crane_AI_Project/
├── app5.py                                          # Streamlit dashboard (main application)
├── Updated_Window_Based_PLC_Preprocessing_featuresAdded.py  # Model definition & training pipeline
├── Crane_Autoencoder.keras                          # Pretrained CNN-LSTM model weights (~10 MB)
├── Final_Scaling.pkl                                # Calibrated feature scaling dictionary
├── threshold.pkl                                    # Baseline threshold cache
├── requirements.txt                                 # Core dependencies
├── SETUP_GUIDE.txt                                  # Quick setup instructions
└── README.md                                        # Documentation
```

> **Note on Data:** Raw telemetry and event exports contain sensitive industrial plant data and are excluded from version control via `.gitignore`. You can supply your own periodic PLC exports directly through the dashboard UI.

---

## Quick start

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/ARUNESH-JHA/Crane_AI_Project.git
cd Crane_AI_Project

# 2. Set up virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\pip.exe install -r requirements.txt
# Linux / macOS:
# source venv/bin/activate && pip install -r requirements.txt

# 3. Launch the platform
.\venv\Scripts\streamlit.exe run app5.py
# Linux / macOS:
# streamlit run app5.py
```

The web dashboard opens at `http://localhost:8501`. Use the sidebar to upload a periodic sensor CSV file and click **Run Anomaly Diagnostic**.

---

## Input Data Format

The pipeline expects standard PLC/SCADA historian time-series exports:

1. **Periodic Telemetry (CSV):**
   - Must include a `PLC_TIMESTAMP` column (standard datetime string).
   - Numeric sensor channels for motor current, voltage, speed, vibration, and temperature (e.g., `MH11_MOT_CRRNT`, `MH11_MOT_VOLT`, `IDLR_VIB_SNSR_1`).
   - Binary/Word status columns for drives and fault words (e.g., `DRV_CMDWRD`, `STTSWRD`, `FLTWRD1`).
2. **Event / Alarm Log (CSV — optional):**
   - Columns: `PLC_TIMESTAMP`, `TagName`, `TagValue`.
   - Merged backward into the time-series using `pd.merge_asof` to provide operational event context.

---

## Feature engineering pipeline

Before scoring, raw signals pass through domain-specific feature engineering:

- **Motor apparent power ($P = V \times I$):** Tracks power draw on each hoist motor to reflect real mechanical load.
- **Master-follower current ratio:** Computes percentage imbalance between paired motors to flag mechanical binding or uneven load distribution before trips occur.
- **Rolling range (peak-to-peak):** Isolates short, sharp current spikes that a standard rolling variance would dampen.
- **Exponential moving average (EMA):** Weights recent timesteps heavier to capture nascent thermal/current degradation trends faster.
- **Bit duty-cycle percentage:** Converts 16-bit decoded PLC command bits into a moving % on-time signal to catch relay chattering and command instability.
- **Vibration gradients & rolling volatility:** Measures acceleration derivative and rate of change on idler bearings.

---

## Dashboard pages

| Page | Description |
|---|---|
| 📊 **Dashboard** | High-level executive overview, overall health index, risk gauge, and full-run anomaly heatmap. |
| 🧠 **AI Insights** | Plain-English explanations explaining *why* an anomaly was flagged, with per-sensor contribution charts and culprit signal plots. |
| 🔬 **Sensor Explorer** | Deep dive into individual sensors per time window — compares raw signal against rolling mean, 3σ confidence intervals, and derivative trends. |
| 📈 **Timeline** | Interactive timeline plotting reconstruction error against dynamic threshold limits with pan, zoom, and time-range filters. |
| 📐 **Statistics** | Distribution metrics (mean, median, IQR, skewness, kurtosis, outlier counts), histograms, and box plots. |
| 🔗 **Correlation** | Heatmap and top-pair ranking for identifying sensor co-movement and multi-drive synchronization issues. |
| 📄 **Reports** | Generates plant executive summaries, searchable incident logs, and downloadable reports in CSV, Excel, and TXT formats. |
| ℹ️ **About Model** | Summary of the neural architecture, tensor shapes, and input layers. |

---

## Tech stack

- **Deep Learning:** TensorFlow 2.x / Keras (Conv1D, LSTM, MultiHeadAttention)
- **UI / Frontend:** Streamlit
- **Visualization:** Plotly & Plotly Express
- **Analytics:** NumPy, pandas, scikit-learn, joblib

---

## License

This project is shared for educational, research, and portfolio demonstration purposes.

---

*Authored by [Arunesh Jha](https://github.com/ARUNESH-JHA)*
