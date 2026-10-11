"""Cleaning, feature engineering and a leak-free train/test split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROTOCOLS = ["tcp", "udp", "icmp"]
AUTH_PORTS = [21, 22, 3389]            # FTP, SSH, RDP: common brute-force targets


def load_data(path="data/traffic.csv"):
    return pd.read_csv(path, parse_dates=["timestamp"])


def clean(df):
    """Remove duplicates, rows with missing values and impossible values."""
    report = {"rows_in": len(df)}
    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_in"] - len(df)
    before = len(df)
    df = df.dropna()
    report["rows_with_missing_removed"] = before - len(df)
    before = len(df)
    df = df[(df["packet_size"] > 0) & (df["duration"] >= 0) & (df["request_frequency"] >= 0)]
    report["invalid_rows_removed"] = before - len(df)
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def engineer_features(df):
    """Turn raw log columns into model-ready numeric features.

    Heavy-tailed columns are log-transformed, and a few ratios capture behaviour
    that a security analyst would look for (ports per second, who sends more data...).
    """
    out = pd.DataFrame(index=df.index)
    out["packet_size"] = df["packet_size"]
    out["request_frequency_log"] = np.log1p(df["request_frequency"])
    out["failed_logins"] = df["failed_logins"]
    out["duration"] = df["duration"]
    out["src_bytes_log"] = np.log1p(df["src_bytes"])
    out["dst_bytes_log"] = np.log1p(df["dst_bytes"])
    out["unique_ports"] = df["unique_ports"]
    out["syn_ratio"] = df["syn_ratio"]

    # Engineered features
    out["bytes_ratio"] = np.log1p(df["src_bytes"]) - np.log1p(df["dst_bytes"])
    out["traffic_volume_log"] = np.log1p(df["request_frequency"] * df["packet_size"])
    out["ports_per_second"] = df["unique_ports"] / (df["duration"] + 0.1)
    out["login_rate"] = df["failed_logins"] / (df["duration"] + 0.1)
    out["is_auth_port"] = df["dst_port"].isin(AUTH_PORTS).astype(int)
    hour = df["timestamp"].dt.hour
    out["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)

    # One-hot encode the protocol (fixed categories, so train and test always match)
    for p in PROTOCOLS:
        out[f"protocol_{p}"] = (df["protocol"] == p).astype(int)
    return out


def prepare_datasets(df, test_size=0.2, seed=42):
    """Split FIRST, then fit the scaler on the training data only (no data leakage)."""
    X = engineer_features(df)
    y_multi = df["label"]
    y_binary = df["is_attack"]

    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y_multi, y_binary, test_size=test_size, stratify=y_multi, random_state=seed)

    scaler = StandardScaler().fit(X_train)           # learns mean/std from TRAIN only
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_s, "X_test_scaled": X_test_s,
        "y_train": y_train, "y_test": y_test,
        "y_train_binary": yb_train, "y_test_binary": yb_test,
        "scaler": scaler, "feature_names": list(X.columns),
    }