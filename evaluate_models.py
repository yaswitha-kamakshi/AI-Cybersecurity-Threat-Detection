"""Evaluate the saved models on the held-out test set and save charts."""
import os
import pandas as pd
from sklearn.metrics import classification_report
from src.preprocessing import load_data, clean, prepare_datasets
from src.models import load_models, isolation_predict
from src.evaluation import (CLASSES, attack_probability, binary_metrics, plot_confusion,
                            plot_feature_importance, plot_model_comparison, plot_pr_curves)

pd.set_option("display.width", 200)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

df, _ = clean(load_data())
data = prepare_datasets(df)                       # same seed -> same test set as train.py
models, _, features = load_models()
rf, lr, iso = models["random_forest"], models["baseline"], models["isolation_forest"]
Xt, Xts, y_true, y_bin = data["X_test"], data["X_test_scaled"], data["y_test"], data["y_test_binary"]

# ---- 1. Attack-type performance (Random Forest) ----
rf_pred = rf.predict(Xt)
print("=== Random Forest: per-class results (test set) ===")
print(classification_report(y_true, rf_pred, labels=CLASSES, digits=3))
cm = plot_confusion(y_true, rf_pred, "outputs/figures/04_confusion_matrix.png")

# ---- 2. Missed attacks by type ----
print("=== Missed attacks (false negatives) by type ===")
for i, name in enumerate(CLASSES[1:], start=1):
    total = cm[i].sum()
    missed = cm[i, 0]
    print(f"{name:<11} missed {missed:>3} of {total:>3}  ({100 * missed / total:.0f}% slip through as Normal)")
false_alarms = cm[0, 1:].sum()
print(f"\nFalse alarms: {false_alarms} of {cm[0].sum()} normal records flagged "
      f"({100 * false_alarms / cm[0].sum():.1f}%)")

# ---- 3. Model comparison: attack vs normal ----
rf_attack = (rf_pred != "Normal").astype(int)
results = {
    "Logistic Regression": binary_metrics(y_bin, lr.predict(Xts)),
    "Random Forest": binary_metrics(y_bin, rf_attack),
    "Isolation Forest": binary_metrics(y_bin, isolation_predict(iso, Xts)),
}
table = pd.DataFrame(results).T.round(3)
print("\n=== Attack vs normal: model comparison ===")
print(table.to_string())
plot_model_comparison(results, "outputs/figures/05_model_comparison.png")
table.to_csv("outputs/logs/model_comparison.csv")

# ---- 4. Threshold trade-off (how a SOC would tune alerts) ----
rf_score = attack_probability(rf, Xt)
print("\n=== Random Forest alert threshold trade-off ===")
rows = []
for th in [0.2, 0.3, 0.5, 0.7, 0.9]:
    m = binary_metrics(y_bin, (rf_score >= th).astype(int))
    rows.append({"threshold": th, "precision": m["Precision"], "recall": m["Recall"], "F1": m["F1"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Lower threshold = catch more attacks but more false alarms; higher = fewer alarms but more misses.")

lr_score = lr.predict_proba(Xts)[:, 1]
iso_score = -iso.score_samples(Xts)               # higher = more anomalous
plot_pr_curves(y_bin, {"Random Forest": rf_score, "Logistic Regression": lr_score, "Isolation Forest": iso_score},
               {}, "outputs/figures/06_precision_recall.png")

# ---- 5. What the model relies on ----
imp = plot_feature_importance(rf, features, "outputs/figures/07_feature_importance.png")
print("\n=== Top 5 features (Random Forest importance) ===")
print(imp.head(5).round(3).to_string())
print(f"is_night importance: {dict(zip(features, rf.feature_importances_))['is_night']:.4f} "
      "(timestamps are random in this simulation, so a low value is expected)")

print("\nSaved charts 04-07 to outputs/figures/ and outputs/logs/model_comparison.csv")