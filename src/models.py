"""The three detectors: Logistic Regression (baseline), Random Forest, Isolation Forest."""
import json
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "models"


def train_baseline(X_train_scaled, y_train_binary):
    """Simple linear baseline: attack vs normal. Needs scaled features."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return model.fit(X_train_scaled, y_train_binary)


def train_random_forest(X_train, y_train):
    """Supervised multi-class detector: Normal / DoS / BruteForce / PortScan."""
    model = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42)
    return model.fit(X_train, y_train)


def train_isolation_forest(X_train_scaled, y_train_binary, contamination=0.05):
    """Unsupervised anomaly detector, trained on NORMAL traffic only.

    It learns what normal looks like and flags anything unusual, including
    attack types it has never seen. `contamination` sets how much of normal
    traffic it is allowed to flag (the false-alarm budget).
    """
    normal_only = X_train_scaled[y_train_binary == 0]
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    return model.fit(normal_only)


def isolation_predict(model, X_scaled):
    """Return 1 for anomaly (suspicious) and 0 for normal."""
    return (model.predict(X_scaled) == -1).astype(int)


def save_models(models, scaler, feature_names, folder=MODEL_DIR):
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(folder, f"{name}.joblib"), compress=3)
    joblib.dump(scaler, os.path.join(folder, "scaler.joblib"))
    with open(os.path.join(folder, "features.json"), "w") as f:
        json.dump(feature_names, f, indent=2)


def load_models(folder=MODEL_DIR):
    names = ["baseline", "random_forest", "isolation_forest"]
    models = {n: joblib.load(os.path.join(folder, f"{n}.joblib")) for n in names}
    scaler = joblib.load(os.path.join(folder, "scaler.joblib"))
    with open(os.path.join(folder, "features.json")) as f:
        features = json.load(f)
    return models, scaler, features