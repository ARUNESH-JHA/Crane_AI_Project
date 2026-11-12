# # Window Based PLC Preprocessing Pipeline

# ### 5 New Feature Engineering Techniques Added
# 1. **Motor Power** (Voltage x Current) for MH11/12/21/22 and ISU1/2/3
# 2. **Master-Follower Current Ratio** (% imbalance between paired motors)
# 3. **Rolling Range (Peak-to-Peak)** for motor currents (volatility spikes)
# 4. **Exponential Moving Average (EMA)** for motor currents (trend-sensitive smoothing)
# 5. **Bit Duty-Cycle %** for decoded PLC command bits (control stability signal)

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
import joblib
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    UpSampling1D,
    BatchNormalization,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout,
    MultiHeadAttention,
    Add
)
from tensorflow.keras.optimizers import Adam

# =====================================================
# DYNAMIC CONFIGURATION ADJUSTMENT
# =====================================================
def get_dynamic_window_config(df, timestamp_col="PLC_TIMESTAMP"):
    timestamps = pd.to_datetime(df[timestamp_col])
    total_duration_seconds = (timestamps.max() - timestamps.min()).total_seconds()
    total_duration_hours = total_duration_seconds / 3600.0
    sampling_interval = timestamps.diff().median().total_seconds()
    
    if total_duration_hours < 0.5:
        window_seconds = max(10, int(sampling_interval * 10)) 
    elif total_duration_hours < 5.0:
        window_seconds = 30
    elif total_duration_hours < 48.0:
        window_seconds = 60
    else:
        window_seconds = 300
        
    decision_history = max(30, int(1800 / window_seconds))
    
    print(f"--- DYNAMIC WINDOW CONFIGURATION ---")
    print(f"Dataset Duration: {total_duration_hours:.2f} hours")
    print(f"Row Interval    : {sampling_interval:.2f} seconds")
    print(f"Decided Window  : {window_seconds} seconds")
    print(f"Decided History : {decision_history} windows")
    print(f"------------------------------------")
    
    return timedelta(seconds=window_seconds), decision_history


# =====================================================
# BUILD CNN-LSTM ATTENTION AUTOENCODER
# =====================================================
def build_autoencoder(feature_count):
    input_layer = Input(shape=(60, feature_count))

    # CNN Encoder
    x = Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    )(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same"
    )(x)

    # LSTM Encoder
    x = LSTM(
        128,
        return_sequences=True
    )(x)
    x = Dropout(0.2)(x)

    # Multi Head Attention
    att = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(x, x)
    
    # Residual Connection
    x = Add()([x, att])
    x = BatchNormalization()(x)

    x = LSTM(
        64,
        return_sequences=False
    )(x)
    latent = Dense(
        32,
        activation="relu"
    )(x)

    # Decoder
    x = RepeatVector(30)(latent)
    x = LSTM(
        64,
        return_sequences=True
    )(x)
    x = LSTM(
        128,
        return_sequences=True
    )(x)
    x = UpSampling1D(size=2)(x)

    output = TimeDistributed(
        Dense(feature_count)
    )(x)

    model = Model(input_layer, output)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse"
    )
    return model

# =====================================================
# UNIFIED PREPROCESSING & FEATURE ENGINEERING
# =====================================================
def preprocess_and_engineer(p_df, e_pivot, word_cols, is_calibration=False, window_size=timedelta(seconds=60)):
    df_p = p_df.copy()
    
    # 1. Decode PLC word columns
    bit_frames = []
    for col in word_cols:
        values = pd.to_numeric(df_p[col], errors="coerce").fillna(0).astype(np.uint16).to_numpy()
        decoded = pd.DataFrame({
            f"{col}_BIT_{bit}": ((values >> bit) & 1).astype(np.uint8)
            for bit in range(16)
        }, index=df_p.index)
        bit_frames.append(decoded)
    if bit_frames:
        df_p = pd.concat([df_p] + bit_frames, axis=1)
    df_p.drop(columns=word_cols, inplace=True, errors="ignore")

    # 2. Merge event data
    common_cols = list(set(df_p.columns).intersection(set(e_pivot.columns)))
    if "PLC_TIMESTAMP" in common_cols:
        common_cols.remove("PLC_TIMESTAMP")
    e_ready = e_pivot.drop(columns=common_cols, errors="ignore")

    merged = pd.merge_asof(
        df_p.sort_values("PLC_TIMESTAMP"),
        e_ready.sort_values("PLC_TIMESTAMP"),
        on="PLC_TIMESTAMP",
        direction="backward",
        allow_exact_matches=True
    )
    
    # For calibration, we drop completely empty columns
    if is_calibration:
        merged = merged.dropna(axis=1, how="all")

    # 3. Feature Engineering
    # Hoist Motor differences
    merged["MH11_MH12_CURR_DIFF"] = (merged["MH11_MOT_CRRNT"] - merged["MH12_MOT_CRRNT"]).abs()
    merged["MH22_MH21_CURR_DIFF"] = (merged["MH22_MOT_CRRNT"] - merged["MH21_MOT_CRRNT"]).abs()
    merged["MH11_MH12_VOLT_DIFF"] = (merged["MH11_MOT_VOLT"] - merged["MH12_MOT_VOLT"]).abs()
    merged["MH22_MH21_VOLT_DIFF"] = (merged["MH22_MOT_VOLT"] - merged["MH21_MOT_VOLT"]).abs()
    merged["MH11_MH12_SPEED_DIFF"] = (merged["MH11_SPD_REF_3"] - merged["MH12_SPD_REF_3"]).abs()
    merged["MH22_MH21_SPEED_DIFF"] = (merged["MH22_SPD_REF_3"] - merged["MH21_SPD_REF_3"]).abs()

    # ISU Voltage Differences & STD
    if "ISU1_DC_VOLT" in merged.columns and "ISU2_DC_VOLT" in merged.columns and "ISU3_DC_VOLT" in merged.columns:
        merged["ISU1_ISU2_VOLT_DIFF"] = (merged["ISU1_DC_VOLT"] - merged["ISU2_DC_VOLT"]- merged["ISU3_DC_VOLT"]).abs()
    if "ISU_1_2_SEL_RQST"==1 in merged.columns:    
        merged["ALL_ISU_VOLT_STD"] = merged[["ISU1_DC_VOLT", "ISU2_DC_VOLT"]].std(axis=1)
    if "ISU_1_3_SEL_RQST"==1 in merged.columns:   
        merged["ALL_ISU_VOLT_STD"] = merged[["ISU1_DC_VOLT", "ISU3_DC_VOLT"]].std(axis=1)
    if "ISU_2_3_SEL_RQST"==1 in merged.columns:    
        merged["ALL_ISU_VOLT_STD"] = merged[["ISU2_DC_VOLT", "ISU3_DC_VOLT"]].std(axis=1)


    # Idler Vibration Gradients & Rolling Stats
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

    # =====================================================
    # 5 NEW FEATURE ENGINEERING TECHNIQUES (added for assignment)
    # =====================================================

    # --- 1. MOTOR POWER (Apparent Power = Voltage x Current) ---
    # Physics-based feature: directly reflects load/stress on each motor
    if "MH11_MOT_CRRNT" in merged.columns and "MH11_MOT_VOLT" in merged.columns:
        merged["MH11_POWER"] = merged["MH11_MOT_CRRNT"] * merged["MH11_MOT_VOLT"]
    if "MH12_MOT_CRRNT" in merged.columns and "MH12_MOT_VOLT" in merged.columns:
        merged["MH12_POWER"] = merged["MH12_MOT_CRRNT"] * merged["MH12_MOT_VOLT"]
    if "MH21_MOT_CRRNT" in merged.columns and "MH21_MOT_VOLT" in merged.columns:
        merged["MH21_POWER"] = merged["MH21_MOT_CRRNT"] * merged["MH21_MOT_VOLT"]
    if "MH22_MOT_CRRNT" in merged.columns and "MH22_MOT_VOLT" in merged.columns:
        merged["MH22_POWER"] = merged["MH22_MOT_CRRNT"] * merged["MH22_MOT_VOLT"]

    # --- 2. MASTER-FOLLOWER RATIO (% Imbalance, not just absolute diff) ---
    # Ratio is more interpretable than a raw difference -- e.g. 1.5 means
    # the master motor is drawing 50% more current than the follower.
    if "MH11_MOT_CRRNT" in merged.columns and "MH12_MOT_CRRNT" in merged.columns:
        merged["MH11_MH12_CURR_RATIO"] = merged["MH11_MOT_CRRNT"] / (merged["MH12_MOT_CRRNT"] + 1e-6)
    if "MH22_MOT_CRRNT" in merged.columns and "MH21_MOT_CRRNT" in merged.columns:
        merged["MH22_MH21_CURR_RATIO"] = merged["MH22_MOT_CRRNT"] / (merged["MH21_MOT_CRRNT"] + 1e-6)

    # --- 3. ROLLING RANGE / PEAK-TO-PEAK (Volatility / Sudden Spikes) ---
    # Captures short sharp bursts that a plain rolling std can smooth over.
    for col in motor_current:
        roll_max = merged[col].rolling(5, min_periods=1).max()
        roll_min = merged[col].rolling(5, min_periods=1).min()
        merged[col + "_ROLL_RANGE"] = roll_max - roll_min

    # --- 4. EXPONENTIAL MOVING AVERAGE (EMA) -- Trend-Sensitive Smoothing ---
    # Unlike a simple rolling mean, EMA weights recent readings more heavily,
    # so it reacts faster to genuine degradation trends.
    for col in motor_current:
        merged[col + "_EMA"] = merged[col].ewm(span=5, adjust=False).mean()

    # --- 5. BIT DUTY-CYCLE (% ON-time within the window) ---
    # Converts raw 0/1 command bits into a % ON-time signal -- useful for
    # spotting unstable/flickering control commands (control chatter).
    # Limited to the first 100 bit columns to avoid excessive dimensionality.
    bit_cols_for_duty = [c for c in merged.columns if "_BIT_" in c][:100]
    for col in bit_cols_for_duty:
        merged[col + "_DUTY_PCT"] = merged[col].rolling(10, min_periods=1).mean() * 100

    # Defragment the dataframe after adding many new columns (performance)
    merged = merged.copy()

    # Time features
    if is_calibration:
        start_time_cal = merged["PLC_TIMESTAMP"].min()
        window_index = ((merged["PLC_TIMESTAMP"] - start_time_cal) // window_size).astype(int)
        window_start_for_row = start_time_cal + window_index * window_size
        merged["WINDOW_TIME"] = (merged["PLC_TIMESTAMP"] - window_start_for_row).dt.total_seconds()
        merged["DELTA_TIME"] = merged.groupby(window_index)["PLC_TIMESTAMP"].diff().dt.total_seconds().fillna(0)
    else:
        start_time_win = merged["PLC_TIMESTAMP"].min()
        merged["WINDOW_TIME"] = (merged["PLC_TIMESTAMP"] - start_time_win).dt.total_seconds()
        merged["DELTA_TIME"] = merged["PLC_TIMESTAMP"].diff().dt.total_seconds().fillna(0)

    cols = [c for c in merged.columns if c != "PLC_TIMESTAMP"]
    return merged, cols

# =====================================================
# LOAD REPRESENTATIVE HEALTHY CALIBRATION DATASET
# =====================================================
periodic_df = pd.read_csv("PERIODIC16-6-2024 LIMIT 5000.csv")
event_df = pd.read_csv("EVENT IJ1 16-6-24.csv")
print("Data successfully loaded!")

# Get dynamic window configuration based on loaded dataset
window, history_size = get_dynamic_window_config(periodic_df)

# =====================================================
# STEP 1 : PREPROCESS
# =====================================================
if "DATA_INGESTION_TIMESTAMP" in periodic_df.columns:
    periodic_df = periodic_df.drop(columns=["DATA_INGESTION_TIMESTAMP"])
periodic_df = periodic_df.dropna(axis=1, how="all")

if "Record_Type" in event_df.columns:
    event_df = event_df.drop(columns=["Record_Type"])
event_df = event_df.dropna(axis=1, how="all")

for df in [periodic_df, event_df]:
    df["PLC_TIMESTAMP"] = (
        pd.to_datetime(df["PLC_TIMESTAMP"], utc=True)
        .dt.tz_localize(None)
    )

periodic_df = periodic_df.sort_values("PLC_TIMESTAMP").reset_index(drop=True)
event_df = event_df.sort_values("PLC_TIMESTAMP").reset_index(drop=True)

# =====================================================
# STEP 2 : DETECT PLC WORDS
# =====================================================
word_patterns = [
    "DRV_CMDWRD",
    "STTSWRD",
    "FLTWRD1",
    "FLTWRD2",
    "ALRMWRD",
    "SHORT_CKT_FLTWRD",
    "CRRNT_UNBAL_FLTWRD"
]
word_columns = [
    c for c in periodic_df.columns
    if any(p in c.upper() for p in word_patterns)
]
print(f"PLC Word Columns : {len(word_columns)}")

# =====================================================
# STEP 3 : EVENT STATE TABLE
# =====================================================
event_pivot = (
    event_df
    .pivot_table(
        index="PLC_TIMESTAMP",
        columns="TagName",
        values="TagValue",
        aggfunc="last"
    )
    .sort_index()
)
event_pivot = event_pivot.ffill().reset_index()

# =====================================================
# CACHED SCALING FACTORS & AUTO-VALIDATION
# =====================================================
SCALING_PICKLE = "Final_Scaling.pkl"

# Run preprocessing on full calibration dataset to find current features
_, current_features = preprocess_and_engineer(periodic_df, event_pivot, word_columns, is_calibration=True)

final_scaling = None
cache_valid = False

if os.path.exists(SCALING_PICKLE):
    try:
        print("Loading pre-calculated scaling factors from cache...")
        final_scaling = joblib.load(SCALING_PICKLE)
        cached_features = list(final_scaling.keys())
        if set(cached_features) == set(current_features):
            print("Cache validation successful! Features match.")
            cache_valid = True
        else:
            print("Cache validation failed: Pipeline feature list has changed.")
    except Exception as e:
        print(f"Error loading cache: {e}")

if not cache_valid:
    print("Representative Healthy Calibration Phase starting...")
    
    # 1. Generate Calibration Dataset
    calib_df, _ = preprocess_and_engineer(periodic_df, event_pivot, word_columns, is_calibration=True)

    # 2. Load BigQuery population scaling factors (Deduplicating to prevent ValueError)
    scaling_factors_df = pd.read_csv("IJ1_SCALING_FACTORS_608 TAGS.csv")
    scaling_factors_df = scaling_factors_df.drop_duplicates(subset=["column_name"], keep="first")
    scaling_dict = scaling_factors_df.set_index("column_name")[["min_value", "max_value", "mean_value", "std_value"]].to_dict(orient="index")

    # 3. Build final_scaling dictionary with classification
    final_scaling = {}
    for col in current_features:
        is_binary = False
        if "_BIT_" in col:
            is_binary = True
        elif col in scaling_dict:
            if scaling_dict[col]["min_value"] == 0 and scaling_dict[col]["max_value"] == 1:
                is_binary = True

        if is_binary:
            final_scaling[col] = {"type": "binary", "mean": 0.0, "std": 1.0}
        else:
            if col in scaling_dict:
                mean = scaling_dict[col]["mean_value"]
                std = scaling_dict[col]["std_value"]
                if pd.isna(std) or std < 1e-8:
                    std = 1.0
                if pd.isna(mean):
                    mean = 0.0
            else:
                mean = calib_df[col].mean()
                std = calib_df[col].std()
                if pd.isna(std) or std < 1e-8:
                    std = 1.0
                if pd.isna(mean):
                    mean = 0.0
            final_scaling[col] = {"type": "continuous", "mean": float(mean), "std": float(std)}

    joblib.dump(final_scaling, SCALING_PICKLE)
    print(f"Calibration complete. Scaling factors cached in {SCALING_PICKLE}.")
    del calib_df

expected_features = list(final_scaling.keys())

# =====================================================
# STEP 4 : NON-OVERLAPPING WINDOWS (DYNAMIC)
# =====================================================
# (window duration is dynamically set above)
start_time = periodic_df["PLC_TIMESTAMP"].min()
end_time = periodic_df["PLC_TIMESTAMP"].max()
current_start = start_time
window_no = 1

# Load existing model from disk if it exists for incremental training
model_file = "Crane_Autoencoder.keras"
if os.path.exists(model_file):
    print("Loading existing model from disk for incremental training...")
    model = tf.keras.models.load_model(model_file)
else:
    model = None
window_errors = []
error_history = []
anomaly_log = []

early_stop = EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)

while current_start <= end_time:
    current_end = current_start + window

    periodic_60s = periodic_df[
        (periodic_df["PLC_TIMESTAMP"] >= current_start) &
        (periodic_df["PLC_TIMESTAMP"] < current_end)
    ].copy()

    if periodic_60s.empty:
        current_start += window
        window_no += 1
        continue

    print(f"Processing Window {window_no}")

    # Process and engineer features for this window
    window_df, _ = preprocess_and_engineer(periodic_60s, event_pivot, word_columns, is_calibration=False, window_size=window)
    X = window_df.drop(columns=["PLC_TIMESTAMP"])

    # Align columns to match expected features exactly
    X = X.reindex(columns=expected_features, fill_value=0)

    # Normalize continuous columns using cached factors
    for col in expected_features:
        if final_scaling[col]["type"] == "continuous":
            mean = final_scaling[col]["mean"]
            std = final_scaling[col]["std"]
            X[col] = (X[col] - mean) / std

    X = X.fillna(0)

    # Align sequence length to exactly 60 rows (future-proof against missing timestamps)
    target_len = 60
    current_len = len(X)
    if current_len < target_len:
        pad_size = target_len - current_len
        last_row = X.iloc[[-1]]
        X = pd.concat([X] + [last_row] * pad_size, ignore_index=True)
    elif current_len > target_len:
        X = X.head(target_len)

    # Prepare input for CNN-LSTM
    X_sequence = X.to_numpy(dtype=np.float32)
    print("Before Reshape :", X_sequence.shape)
    
    X_sequence = np.expand_dims(
        X_sequence,
        axis=0
    )
    print("After Reshape :", X_sequence.shape)

    # Build model
    if model is None:
        feature_count = X_sequence.shape[2]
        print(f"Features : {feature_count}")
        model = build_autoencoder(feature_count)
        model.summary()
        print("CNN-LSTM Autoencoder Created")    

    # Train model
    history = model.fit(
        X_sequence,
        X_sequence,
        epochs=5,
        batch_size=1,
        callbacks=[early_stop],
        verbose=0
    )

    if window_no % 100 == 0:
        model.save("Crane_Autoencoder.keras")
        print("Model Saved")
       
    if window_no % 100 == 0:
        pd.DataFrame(anomaly_log).to_csv(
            "Anomaly_Log.csv",
            index=False
        )

    # Prediction
    prediction = model.predict(
        X_sequence,
        verbose=0
    )

    # Reconstruction error
    window_error = np.mean(
        np.square(
            X_sequence -
            prediction
        )
    )
    window_errors.append(window_error)
    error_history.append(window_error)
    print("Window Error :", window_error)

    if len(error_history) < history_size:
        THRESHOLD = np.mean(error_history) + 3*np.std(error_history)
    else:
        recent = error_history[-history_size:]
        THRESHOLD = np.mean(recent) + 3*np.std(recent)
        
    if window_no % 100 == 0:
        joblib.dump(
            THRESHOLD,
            "threshold.pkl"
        )

    status = "HEALTHY"
    if window_error > THRESHOLD:
        status = "ANOMALY"
    print(status)

    anomaly_log.append({
        "Window": window_no,
        "Time": current_start,
        "Error": window_error,
        "Threshold": THRESHOLD,
        "Status": status
    })

    del X
    del X_sequence
    del prediction

    current_start += window
    window_no += 1

# =====================================================
# SAVE FINAL MODEL AND LOGS AT PIPELINE END
# =====================================================
if model is not None:
    model.save("Crane_Autoencoder.keras")
    print("Model Saved successfully to Crane_Autoencoder.keras")

if anomaly_log:
    pd.DataFrame(anomaly_log).to_csv("Anomaly_Log.csv", index=False)
    print("Anomaly Log saved successfully to Anomaly_Log.csv")

print("Pipeline Completed Successfully")

