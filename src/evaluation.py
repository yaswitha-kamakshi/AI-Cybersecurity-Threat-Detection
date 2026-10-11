"""Evaluation helpers: metrics tables, confusion matrix, curves and feature importance."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)

CLASSES = ["Normal", "DoS", "BruteForce", "PortScan"]
COLORS = {"Logistic Regression": "#9aa5b1", "Random Forest": "#4c9f70", "Isolation Forest": "#5b8def"}


def attack_probability(rf, X):
    """Probability that a record is ANY attack = 1 - P(Normal)."""
    normal_col = list(rf.classes_).index("Normal")
    return 1.0 - rf.predict_proba(X)[:, normal_col]


def binary_metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred), "F1": f1_score(y_true, y_pred)}


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i, j]}\n({pct[i, j]:.0f}%)" for j in range(4)] for i in range(4)])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(pct, annot=labels, fmt="", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={"label": "% of true class"}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Random Forest confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return cm


def plot_model_comparison(results, path):
    df = pd.DataFrame(results).T[["Precision", "Recall", "F1"]]
    ax = df.plot(kind="bar", figsize=(8, 4.5), color=["#d9534f", "#f0ad4e", "#4c9f70"], rot=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title("Attack detection: model comparison (test set)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_feature_importance(rf, feature_names, path, top=12):
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp.index, imp.values, color="#4c9f70")
    ax.set_title("Random Forest: most important features"); ax.set_xlabel("importance")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return imp.sort_values(ascending=False)


def plot_pr_curves(y_true, scores, points, path):
    """Precision-recall curves for score-based models, plus single operating points."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=name, color=COLORS[name], lw=2)
    for name, (p, r) in points.items():
        ax.scatter([r], [p], s=90, color=COLORS[name], edgecolor="black", zorder=5, label=f"{name} (fixed threshold)")
    ax.set_xlabel("Recall (attacks caught)"); ax.set_ylabel("Precision (alerts that are real)")
    ax.set_title("Precision-recall trade-off"); ax.set_ylim(0, 1.02); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)