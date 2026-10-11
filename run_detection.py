"""Replay held-out traffic through the detector like a live feed, raise alerts, build a dashboard."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.detector import ThreatDetector, SEVERITIES
from src.preprocessing import load_data, clean, prepare_datasets

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

df, _ = clean(load_data())
data = prepare_datasets(df)                                   # same split as training
feed = df.loc[data["X_test"].index].sort_values("timestamp")  # unseen traffic, in time order
print(f"Replaying {len(feed)} unseen traffic records through the detector...\n")

detector = ThreatDetector(threshold=0.5)
res = detector.analyze(feed)
alerts = res[res["attack_type"] != "Normal"].copy()

# ---- Live-style alert log (first 12 alerts) ----
print("=== ALERT LOG (first 12) ===")
for _, a in alerts.head(12).iterrows():
    print(f"[{a['timestamp']:%m-%d %H:%M:%S}] {a['severity']:<8} {a['attack_type']:<10} "
          f"src={a['src_ip']:<16} score={a['attack_score']:.2f}  {a['reason']}")

# ---- Summary against ground truth ----
confirmed = alerts[alerts["attack_type"] != "Anomaly"]
anomaly_only = alerts[alerts["attack_type"] == "Anomaly"]
true_attacks = res[res["label"] != "Normal"]
caught = confirmed[confirmed["label"] != "Normal"]
print("\n=== SUMMARY ===")
print(f"Records analysed          : {len(res)}")
print(f"Alerts raised             : {len(alerts)}  ({len(confirmed)} classifier + {len(anomaly_only)} anomaly-only)")
print("By severity               : " + ", ".join(
    f"{s} {int((alerts['severity'] == s).sum())}" for s in reversed(SEVERITIES)))
print("By attack type            : " + ", ".join(f"{k} {v}" for k, v in alerts["attack_type"].value_counts().items()))
print(f"Real attacks in the feed  : {len(true_attacks)}")
print(f"Caught by the classifier  : {len(caught)} ({100 * len(caught) / len(true_attacks):.1f}%)")
print(f"Classifier false alarms   : {len(confirmed) - len(caught)} "
      f"({100 * (len(confirmed) - len(caught)) / (res['label'] == 'Normal').sum():.1f}% of normal traffic)")
extra = anomaly_only[anomaly_only["label"] != "Normal"]
print(f"Anomaly-only alerts       : {len(anomaly_only)}, of which {len(extra)} were real attacks the classifier missed")
crit = alerts[alerts["severity"].isin(["HIGH", "CRITICAL"])]
print(f"HIGH/CRITICAL alerts      : {len(crit)}, of which {int((crit['label'] != 'Normal').sum())} were real attacks "
      f"({100 * (crit['label'] != 'Normal').mean():.1f}% precision)")

# ---- Save the alert log ----
cols = ["timestamp", "src_ip", "severity", "attack_type", "attack_score", "reason", "label"]
alerts[cols].rename(columns={"label": "actual_label"}).to_csv("outputs/logs/alerts.csv", index=False)

# ---- Dashboard ----
palette = {"DoS": "#d9534f", "BruteForce": "#f0ad4e", "PortScan": "#5b8def", "Anomaly": "#9aa5b1"}
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

timeline = alerts.set_index("timestamp").groupby([pd.Grouper(freq="12h"), "attack_type"]).size().unstack(fill_value=0)
timeline.index = [t.strftime("%m-%d %H:%M") for t in timeline.index]      # text labels give normal-width bars
timeline.plot(kind="bar", stacked=True, ax=ax[0, 0], color=[palette.get(c, "gray") for c in timeline.columns], width=0.85)
ax[0, 0].set_title("Alerts over time (12-hour windows)")
ax[0, 0].tick_params(axis="x", rotation=60, labelsize=7)
ax[0, 0].set_xlabel(""); ax[0, 0].set_ylabel("alerts")

sev = alerts["severity"].value_counts().reindex(SEVERITIES, fill_value=0)
ax[0, 1].bar(sev.index, sev.values, color=["#9aa5b1", "#f0ad4e", "#e8742c", "#d9534f"])
ax[0, 1].set_title("Alerts by severity")
for i, v in enumerate(sev.values):
    ax[0, 1].text(i, v, str(v), ha="center", va="bottom")

top = confirmed.groupby("src_ip").agg(alerts=("src_ip", "size"), attacks=("label", lambda s: (s != "Normal").sum()))
top = top.sort_values("alerts", ascending=False).head(10).iloc[::-1]
ax[1, 0].barh(top.index, top["alerts"], color="#d9534f", label="false alarms")
ax[1, 0].barh(top.index, top["attacks"], color="#4c9f70", label="real attacks")
ax[1, 0].set_title("Top source IPs by alerts (green = confirmed attacks)")
ax[1, 0].legend(loc="lower right", fontsize=8)

types = ["DoS", "BruteForce", "PortScan"]
actual = [int((res["label"] == t).sum()) for t in types]
found = [int(((res["label"] == t) & (res["attack_type"] == t)).sum()) for t in types]
x = range(len(types))
ax[1, 1].bar([i - 0.2 for i in x], actual, width=0.4, label="actual attacks", color="#9aa5b1")
ax[1, 1].bar([i + 0.2 for i in x], found, width=0.4, label="correctly detected", color="#4c9f70")
ax[1, 1].set_xticks(list(x)); ax[1, 1].set_xticklabels(types)
ax[1, 1].set_title("Detection by attack type"); ax[1, 1].legend()

fig.suptitle("AI-Powered Cybersecurity Threat Detection: SOC dashboard (simulated traffic)", fontsize=13)
fig.tight_layout()
fig.savefig("outputs/figures/08_dashboard.png", dpi=110)
print("\nSaved outputs/logs/alerts.csv and outputs/figures/08_dashboard.png")