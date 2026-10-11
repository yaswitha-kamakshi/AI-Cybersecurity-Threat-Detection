"""ThreatDetector: loads the saved models, scores traffic and turns detections into alerts."""
import numpy as np
import pandas as pd
from src.models import load_models, isolation_predict
from src.preprocessing import engineer_features

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Typical normal-traffic values, used when an API caller sends only a few fields.
RAW_DEFAULTS = {
    "packet_size": 500.0, "request_frequency": 5.0, "failed_logins": 0, "duration": 2.0,
    "src_bytes": 700.0, "dst_bytes": 1100.0, "unique_ports": 1, "syn_ratio": 0.05,
    "protocol": "tcp", "dst_port": 443,
}
NUMERIC_FIELDS = ["packet_size", "request_frequency", "failed_logins", "duration",
                  "src_bytes", "dst_bytes", "unique_ports", "syn_ratio", "dst_port"]


def explain(row, attack_type):
    """Plain-English reason an analyst can read at a glance."""
    if attack_type == "DoS":
        return (f"{row['request_frequency']:.0f} requests/s with {row['syn_ratio'] * 100:.0f}% "
                f"half-open (SYN) connections: possible denial-of-service flood")
    if attack_type == "BruteForce":
        return f"{int(row['failed_logins'])} failed logins against port {int(row['dst_port'])}: possible password guessing"
    if attack_type == "PortScan":
        return f"{int(row['unique_ports'])} different ports probed in one window: possible port scan"
    return "traffic pattern differs from the normal baseline (unsupervised anomaly flag)"


def severity_for(score, anomaly, flagged):
    """Base level from model confidence; both detectors agreeing escalates one level."""
    if not flagged:
        return "LOW" if anomaly else "NONE"
    level = 0 if score < 0.7 else (1 if score < 0.9 else 2)
    if anomaly:
        level = min(level + 1, 3)
    return SEVERITIES[level]


class ThreatDetector:
    def __init__(self, model_dir="models", threshold=0.5):
        self.models, self.scaler, self.features = load_models(model_dir)
        self.threshold = threshold

    def analyze(self, raw):
        """Score every row of raw traffic. Returns the rows plus detection columns."""
        X = engineer_features(raw)[self.features]
        rf = self.models["random_forest"]
        classes = list(rf.classes_)
        proba = rf.predict_proba(X)
        attack_idx = [i for i, c in enumerate(classes) if c != "Normal"]
        attack_score = 1.0 - proba[:, classes.index("Normal")]
        likely_type = np.array(classes)[attack_idx][proba[:, attack_idx].argmax(axis=1)]

        scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        anomaly = isolation_predict(self.models["isolation_forest"], scaled)
        flagged = attack_score >= self.threshold

        out = raw.copy()
        out["attack_score"] = attack_score
        out["anomaly"] = anomaly
        out["attack_type"] = np.where(flagged, likely_type, np.where(anomaly == 1, "Anomaly", "Normal"))
        out["severity"] = [severity_for(s, a, f) for s, a, f in zip(attack_score, anomaly, flagged)]
        out["reason"] = [explain(r, t) if t != "Normal" else ""
                         for (_, r), t in zip(out.iterrows(), out["attack_type"])]
        return out

    def predict_one(self, record):
        """Score a single record sent as a dict (used by the API). Missing fields get normal defaults."""
        row = dict(RAW_DEFAULTS)
        for key, value in record.items():
            if key in NUMERIC_FIELDS:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"'{key}' must be a number")
                row[key] = value
            elif key in ("protocol",):
                if value not in ("tcp", "udp", "icmp"):
                    raise ValueError("'protocol' must be tcp, udp or icmp")
                row[key] = value
        row["timestamp"] = pd.Timestamp("2026-10-01 12:00:00")
        result = self.analyze(pd.DataFrame([row])).iloc[0]
        return {
            "threat_detected": bool(result["attack_type"] != "Normal"),
            "attack_type": str(result["attack_type"]),
            "attack_score": round(float(result["attack_score"]), 3),
            "severity": str(result["severity"]),
            "reason": str(result["reason"]),
        }