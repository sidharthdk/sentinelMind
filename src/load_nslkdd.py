"""
SentinelMind — NSL-KDD Loader & Cleaner
Downloads, cleans, and maps NSL-KDD into a format compatible
with the SentinelMind feature pipeline.

Usage:
    python load_nslkdd.py

Outputs:
    data/nslkdd_clean.csv   — cleaned, feature-ready dataset
    data/nslkdd_sample.csv  — 500-row sample for quick testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH  = DATA_DIR / "KDDTest+.txt"

# All 41 features + label + difficulty score
COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Map specific attack names → 4 broad categories + normal
# These mirror the kind of insider threat categories SentinelMind targets
ATTACK_MAP = {
    "normal":          "normal",
    # DoS — analogous to bulk download / system flooding
    "neptune":         "dos", "smurf": "dos", "pod": "dos",
    "teardrop":        "dos", "back":  "dos", "land": "dos",
    "apache2":         "dos", "udpstorm": "dos",
    "processtable":    "dos", "mailbomb": "dos",
    # Probe — analogous to unauthorised DB scanning
    "ipsweep":         "probe", "nmap": "probe", "portsweep": "probe",
    "satan":           "probe", "saint": "probe", "mscan": "probe",
    # R2L — analogous to credential abuse / off-hours access
    "ftp_write":       "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop":        "r2l", "phf": "r2l", "spy": "r2l",
    "warezclient":     "r2l", "warezmaster": "r2l", "sendmail": "r2l",
    "named":           "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "worm":            "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "httptunnel":      "r2l",
    # U2R — analogous to privilege escalation
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit":         "u2r", "ps": "u2r", "sqlattack": "u2r",
    "xterm":           "u2r",
}

# Features most relevant to insider/privileged-user threat detection
# (mirrors the behavioural vectors in SentinelMind's feature extractor)
SENTINEL_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "is_guest_login",
    "count", "srv_count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate",
    "protocol_type", "service", "flag",
    "attack_category", "is_anomaly",
]

# ─── STEP 1: DOWNLOAD ─────────────────────────────────────────────────────────

def download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Found cached: {dest}")
        return
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")

# ─── STEP 2: LOAD ─────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["label"] = df["label"].str.strip().str.lower()
    return df

# ─── STEP 3: CLEAN ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Map labels to attack categories
    df["attack_category"] = df["label"].map(
        lambda x: ATTACK_MAP.get(x, "other")
    )
    df["is_anomaly"] = (df["label"] != "normal").astype(int)

    # Drop the raw difficulty score (not a feature — it's metadata)
    df = df.drop(columns=["difficulty"])

    # Encode categorical columns to integers
    for col in ["protocol_type", "service", "flag"]:
        df[col] = df[col].astype("category").cat.codes

    # Check for nulls
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"  Warning: {null_count} null values found. Filling with 0.")
        df = df.fillna(0)

    return df

# ─── STEP 4: SELECT SENTINEL FEATURES ────────────────────────────────────────

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in SENTINEL_FEATURES if c in df.columns]
    return df[available].copy()

# ─── STEP 5: PRINT SUMMARY ───────────────────────────────────────────────────

def summarise(df: pd.DataFrame, name: str) -> None:
    total = len(df)
    normal = (df["is_anomaly"] == 0).sum()
    anomaly = (df["is_anomaly"] == 1).sum()
    print(f"\n  {name}")
    print(f"    Total rows     : {total:,}")
    print(f"    Normal         : {normal:,}  ({100*normal/total:.1f}%)")
    print(f"    Anomalous      : {anomaly:,}  ({100*anomaly/total:.1f}%)")
    print(f"    Attack types   : {df['attack_category'].value_counts().to_dict()}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/4] Downloading NSL-KDD ...")
    download_if_missing(TRAIN_URL, TRAIN_PATH)
    download_if_missing(TEST_URL,  TEST_PATH)

    print("\n[2/4] Loading raw data ...")
    train_raw = load_raw(TRAIN_PATH)
    test_raw  = load_raw(TEST_PATH)
    print(f"  Train rows: {len(train_raw):,}  |  Test rows: {len(test_raw):,}")

    print("\n[3/4] Cleaning ...")
    train_clean = select_features(clean(train_raw))
    test_clean  = select_features(clean(test_raw))

    summarise(train_clean, "Train set")
    summarise(test_clean,  "Test set")

    print("\n[4/4] Saving ...")
    train_clean.to_csv(DATA_DIR / "nslkdd_train_clean.csv", index=False)
    test_clean.to_csv(DATA_DIR  / "nslkdd_test_clean.csv",  index=False)

    # Small sample for fast iteration during development
    sample = train_clean.sample(500, random_state=42)
    sample.to_csv(DATA_DIR / "nslkdd_sample.csv", index=False)

    print(f"\n  nslkdd_train_clean.csv  — {len(train_clean):,} rows")
    print(f"  nslkdd_test_clean.csv   — {len(test_clean):,} rows")
    print(f"  nslkdd_sample.csv       — 500 rows")
    print(f"\n  Features kept: {list(train_clean.columns)}")
    print("\nDone.")
