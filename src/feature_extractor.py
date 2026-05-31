"""
SentinelMind — Feature Extractor
Loads synthetic audit logs + NSL-KDD, engineers behavioural
feature vectors, and outputs a unified matrix ready for the MPS scorer.

Usage:
    python feature_extractor.py

Outputs:
    data/features_synthetic.csv   — from your audit_logs.csv
    data/features_nslkdd.csv      — from nslkdd_train_clean.csv
    data/features_combined.csv    — merged, scaled, ready for MPS scorer
    models/scaler.pkl             — fitted StandardScaler (reuse at inference)
    models/encoders.pkl           — fitted LabelEncoders per column
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─── COLUMN DEFINITIONS ───────────────────────────────────────────────────────

# Categorical columns that need LabelEncoding
SYNTHETIC_CAT_COLS  = ["role", "department", "db_accessed", "action"]

# Numeric columns that need StandardScaling
SYNTHETIC_NUM_COLS  = [
    "session_duration_min", "records_accessed",
    "hour_of_login", "day_of_week",
]

# Binary flags — kept as-is (already 0/1)
SYNTHETIC_FLAG_COLS = [
    "off_hours", "unauthorised_db",
    "bulk_download", "privilege_escalation",
]

# Final feature columns output for MPS scorer
SYNTHETIC_FEATURE_COLS = (
    ["user_id"]
    + [f"{c}_enc" for c in SYNTHETIC_CAT_COLS]
    + SYNTHETIC_NUM_COLS
    + SYNTHETIC_FLAG_COLS
    + ["is_anomaly"]
)

# NSL-KDD numeric features that overlap with insider threat vectors
NSLKDD_NUM_COLS = [
    "duration", "src_bytes", "dst_bytes",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "is_guest_login", "count", "srv_count",
    "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
]

NSLKDD_CAT_COLS = ["protocol_type", "service", "flag"]

NSLKDD_FEATURE_COLS = (
    NSLKDD_NUM_COLS
    + [f"{c}_enc" for c in NSLKDD_CAT_COLS]
    + ["is_anomaly"]
)

# ─── SYNTHETIC PIPELINE ───────────────────────────────────────────────────────

def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = SYNTHETIC_CAT_COLS + SYNTHETIC_NUM_COLS + SYNTHETIC_FLAG_COLS + ["user_id", "is_anomaly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in audit_logs.csv: {missing}")
    print(f"  Loaded {len(df):,} rows from {path.name}")
    return df

def engineer_synthetic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Returns (feature_df, encoders_dict, scaler)
    Persist these so inference uses the same encoding.
    """
    df = df.copy()

    # — Per-user aggregate features (behavioural baseline context) —
    user_stats = df.groupby("user_id").agg(
        user_avg_records   = ("records_accessed", "mean"),
        user_max_records   = ("records_accessed", "max"),
        user_avg_duration  = ("session_duration_min", "mean"),
        user_login_std     = ("hour_of_login", "std"),   # how erratic are login times?
        user_session_count = ("user_id", "count"),
    ).reset_index()
    user_stats["user_login_std"] = user_stats["user_login_std"].fillna(0)
    df = df.merge(user_stats, on="user_id", how="left")

    # — Ratio features (catch unusual spikes relative to user's own baseline) —
    df["records_vs_user_avg"] = (
        df["records_accessed"] / (df["user_avg_records"] + 1)
    ).round(4)
    df["duration_vs_user_avg"] = (
        df["session_duration_min"] / (df["user_avg_duration"] + 1)
    ).round(4)

    # — Encode categoricals —
    encoders = {}
    for col in SYNTHETIC_CAT_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # — Scale numerics —
    extended_num_cols = SYNTHETIC_NUM_COLS + [
        "user_avg_records", "user_max_records", "user_avg_duration",
        "user_login_std", "user_session_count",
        "records_vs_user_avg", "duration_vs_user_avg",
    ]
    scaler = StandardScaler()
    df[extended_num_cols] = scaler.fit_transform(df[extended_num_cols])

    # — Final column selection —
    feature_cols = (
        ["user_id"]
        + [f"{c}_enc" for c in SYNTHETIC_CAT_COLS]
        + extended_num_cols
        + SYNTHETIC_FLAG_COLS
        + ["is_anomaly"]
    )
    return df[feature_cols], encoders, scaler

# ─── NSL-KDD PIPELINE ────────────────────────────────────────────────────────

def load_nslkdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in NSLKDD_NUM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing NSL-KDD columns: {missing}\nRun load_nslkdd.py first.")
    print(f"  Loaded {len(df):,} rows from {path.name}")
    return df

def engineer_nslkdd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categoricals (already int-encoded in load_nslkdd.py,
    # but re-encode here in case raw file is used directly)
    for col in NSLKDD_CAT_COLS:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        else:
            df[f"{col}_enc"] = df[col]

    # Scale numerics
    scaler = StandardScaler()
    df[NSLKDD_NUM_COLS] = scaler.fit_transform(df[NSLKDD_NUM_COLS])

    # Add placeholder user_id (NSL-KDD is network-level, no user concept)
    df["user_id"] = "NSL-KDD"

    available = [c for c in NSLKDD_FEATURE_COLS if c in df.columns]
    return df[available]

# ─── COMBINE ─────────────────────────────────────────────────────────────────

def align_and_combine(
    synthetic_df: pd.DataFrame,
    nslkdd_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aligns columns between both datasets and concatenates.
    Columns unique to one dataset are filled with 0 in the other.
    """
    combined = pd.concat(
        [synthetic_df, nslkdd_df],
        axis=0,
        ignore_index=True,
        sort=False
    ).fillna(0)

    # Ensure is_anomaly is int
    combined["is_anomaly"] = combined["is_anomaly"].astype(int)

    return combined

# ─── SUMMARY ─────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, name: str) -> None:
    total   = len(df)
    normal  = (df["is_anomaly"] == 0).sum()
    anomaly = (df["is_anomaly"] == 1).sum()
    print(f"\n  {name}")
    print(f"    Rows      : {total:,}")
    print(f"    Features  : {df.shape[1]}")
    print(f"    Normal    : {normal:,}  ({100*normal/total:.1f}%)")
    print(f"    Anomalous : {anomaly:,}  ({100*anomaly/total:.1f}%)")
    print(f"    Nulls     : {df.isnull().sum().sum()}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Synthetic ──────────────────────────────────────────────────
    print("\n[1/4] Processing synthetic audit logs ...")
    synthetic_path = DATA_DIR / "audit_logs.csv"

    if not synthetic_path.exists():
        print(f"  audit_logs.csv not found. Run generate_audit_logs.py first.")
        synthetic_df = None
    else:
        raw_synthetic = load_synthetic(synthetic_path)
        synthetic_df, encoders, scaler = engineer_synthetic_features(raw_synthetic)
        synthetic_df.to_csv(DATA_DIR / "features_synthetic.csv", index=False)
        print_summary(synthetic_df, "Synthetic features")

        # Persist encoder + scaler for inference
        with open(MODEL_DIR / "encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        with open(MODEL_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("\n  Saved models/encoders.pkl")
        print("  Saved models/scaler.pkl")

    # ── NSL-KDD ────────────────────────────────────────────────────
    print("\n[2/4] Processing NSL-KDD ...")
    nslkdd_path = DATA_DIR / "nslkdd_train_clean.csv"

    if not nslkdd_path.exists():
        print("  nslkdd_train_clean.csv not found. Run load_nslkdd.py first.")
        nslkdd_df = None
    else:
        raw_nslkdd = load_nslkdd(nslkdd_path)
        nslkdd_df  = engineer_nslkdd_features(raw_nslkdd)
        nslkdd_df.to_csv(DATA_DIR / "features_nslkdd.csv", index=False)
        print_summary(nslkdd_df, "NSL-KDD features")

    # ── Combine ────────────────────────────────────────────────────
    print("\n[3/4] Combining datasets ...")

    frames = [df for df in [synthetic_df, nslkdd_df] if df is not None]
    if not frames:
        print("  No data found. Run generate_audit_logs.py and load_nslkdd.py first.")
    else:
        combined = align_and_combine(*frames) if len(frames) == 2 else frames[0]
        combined.to_csv(DATA_DIR / "features_combined.csv", index=False)
        print_summary(combined, "Combined features")

    # ── Done ───────────────────────────────────────────────────────
    print("\n[4/4] Output files:")
    for f in sorted(DATA_DIR.glob("features_*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        print(f"  {f}  ({rows:,} rows)")

    print("\nNext step: python mps_scorer.py")
