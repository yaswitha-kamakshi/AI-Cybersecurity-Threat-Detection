"""Run cleaning + feature engineering + split, print a report, save a chart."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.preprocessing import load_data, clean, engineer_features, prepare_datasets

pd.set_option("display.width", 200)

df = load_data()
df, report = clean(df)
print("=== Cleaning report ===")
for key, value in report.items():
    print(f"{key:<28}{value:>8}")

data = prepare_datasets(df)
print(f"\n=== Features ({len(data['feature_names'])}) ===")
print(", ".join(data["feature_names"]))

print("\n=== Train / test split (stratified) ===")
print(f"Train rows: {len(data['X_train'])}   Test rows: {len(data['X_test'])}")
split = pd.DataFrame({
    "train_%": data["y_train"].value_counts(normalize=True) * 100,
    "test_%": data["y_test"].value_counts(normalize=True) * 100,
}).round(1)
print(split)

print("\n=== Leakage check (scaler fitted on TRAIN only) ===")
col = "request_frequency_log"
print(f"{col} mean after scaling -> train: {data['X_train_scaled'][col].mean():+.3f}   "
      f"test: {data['X_test_scaled'][col].mean():+.3f}")
print("Train is ~0 by construction; test is close to 0 but not exactly 0, which is correct:")
print("the test set must be scaled with numbers learned from training data only.")

# Which features separate attacks from normal traffic?
X = engineer_features(df)
corr = X.corrwith(df["is_attack"]).abs().sort_values()
os.makedirs("outputs/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(corr.index, corr.values, color="#5b8def")
ax.set_title("Feature correlation with 'is_attack' (absolute)")
ax.set_xlabel("|correlation|")
fig.tight_layout()
fig.savefig("outputs/figures/03_feature_correlation.png", dpi=110)
print("\nSaved outputs/figures/03_feature_correlation.png")
print("\nTop 5 features by correlation:")
print(corr.sort_values(ascending=False).head(5).round(3).to_string())