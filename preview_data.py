"""Generate the traffic dataset, preview it, and save two charts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.simulator import save_traffic, CLASSES

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = save_traffic("data/traffic.csv")
print("=== First 8 rows ===")
preview = df.head(8).copy()
num_cols = preview.select_dtypes("number").columns
preview[num_cols] = preview[num_cols].round(2)
print(preview.to_string(index=False))
print(f"\nRows: {len(df)}   Columns: {df.shape[1]}")
print("\n=== Traffic by class ===")
counts = df["label"].value_counts()
for name in CLASSES:
    print(f"{name:<11}{counts[name]:>7}  ({100 * counts[name] / len(df):.0f}%)")
print("\n=== Average feature values by class ===")
cols = ["packet_size", "request_frequency", "failed_logins", "unique_ports", "syn_ratio"]
print(df.groupby("label")[cols].mean().loc[CLASSES].round(2))

os.makedirs("outputs/figures", exist_ok=True)
colors = {"Normal": "#4c9f70", "DoS": "#d9534f", "BruteForce": "#f0ad4e", "PortScan": "#5b8def"}

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(CLASSES, [counts[c] for c in CLASSES], color=[colors[c] for c in CLASSES])
ax.set_title("Simulated traffic by class"); ax.set_ylabel("records")
fig.tight_layout(); fig.savefig("outputs/figures/01_class_distribution.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, log) in zip(axes, [("request_frequency", True), ("failed_logins", False), ("unique_ports", False)]):
    data = [df.loc[df.label == c, col].values for c in CLASSES]
    bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(colors[c])
    ax.set_title(col.replace("_", " ").title())
    if log:
        ax.set_yscale("log")
fig.suptitle("How each attack type looks different (and where they overlap)")
fig.tight_layout(); fig.savefig("outputs/figures/02_feature_profiles.png", dpi=110); plt.close(fig)
print("\nSaved outputs/figures/01_class_distribution.png and 02_feature_profiles.png")