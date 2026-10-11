"""Train all three detectors, print headline scores, and save the models."""
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.preprocessing import load_data, clean, prepare_datasets
from src.models import (train_baseline, train_random_forest, train_isolation_forest,
                        isolation_predict, save_models)

df, _ = clean(load_data())
data = prepare_datasets(df)
print(f"Training rows: {len(data['X_train'])}   Test rows: {len(data['X_test'])}\n")

models = {}
t = time.time(); models["baseline"] = train_baseline(data["X_train_scaled"], data["y_train_binary"])
print(f"Logistic Regression trained in {time.time() - t:.1f}s")
t = time.time(); models["random_forest"] = train_random_forest(data["X_train"], data["y_train"])
print(f"Random Forest trained       in {time.time() - t:.1f}s")
t = time.time(); models["isolation_forest"] = train_isolation_forest(data["X_train_scaled"], data["y_train_binary"])
print(f"Isolation Forest trained    in {time.time() - t:.1f}s")

y_true = data["y_test_binary"]
predictions = {
    "Logistic Regression": models["baseline"].predict(data["X_test_scaled"]),
    "Random Forest": (models["random_forest"].predict(data["X_test"]) != "Normal").astype(int),
    "Isolation Forest": isolation_predict(models["isolation_forest"], data["X_test_scaled"]),
}
rows = []
for name, pred in predictions.items():
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_true, pred),
                 "Precision": precision_score(y_true, pred),
                 "Recall": recall_score(y_true, pred),
                 "F1": f1_score(y_true, pred)})
print("\n=== Attack vs normal (test set) ===")
print(pd.DataFrame(rows).round(3).to_string(index=False))

rf_pred = models["random_forest"].predict(data["X_test"])
print(f"\nRandom Forest attack-TYPE macro F1: {f1_score(data['y_test'], rf_pred, average='macro'):.3f}")

save_models(models, data["scaler"], data["feature_names"])
print("\nSaved models to the models/ folder")