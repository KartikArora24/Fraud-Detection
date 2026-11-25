import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score

MIN_VALIDATION_PRECISION = 0.70

def choose_cutoff(labels, scores, minimum_precision=MIN_VALIDATION_PRECISION):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.flatnonzero(precision >= minimum_precision)
    if eligible.size == 0:
        raise ValueError("No validation cutoff meets the precision target.")

    best_recall = recall[eligible].max()
    tied = eligible[recall[eligible] == best_recall]
    best = tied[np.argmax(precision[tied])]
    return float(thresholds[best])

def main():
    if not os.path.exists("creditcard.csv"):
        raise FileNotFoundError(
            "creditcard.csv not found in the current directory. "
            "Please place creditcard.csv here before running train_model.py."
        )

    print("Loading creditcard.csv...")
    df = pd.read_csv("creditcard.csv")
    y = df["Class"]

    # 1. Split the data
    train, rest = train_test_split(df.index, test_size=0.40, stratify=y, random_state=1)
    valid, test = train_test_split(rest, test_size=0.50, stratify=y.loc[rest], random_state=1)

    # 2. Prepare eight inputs
    X = df.drop(columns=["Class", "Time"]).copy()
    X["Amount"] = np.log1p(X["Amount"])
    
    scaler = StandardScaler().fit(X.loc[train])
    scaled = scaler.transform(X)

    feature_auc = [roc_auc_score(y.loc[valid], abs(scaled[valid, j])) for j in range(X.shape[1])]
    best_indices = np.argsort(-np.array(feature_auc))[:8]
    selected_feature_names = list(X.columns[best_indices])
    features = scaled[:, best_indices]
    print("Selected features:", ", ".join(selected_feature_names))

    # 3. Train the models
    print("Training Isolation Forest...")
    forest = IsolationForest(n_estimators=300, max_samples=256, random_state=0, n_jobs=-1)
    forest.fit(features[train])

    print("Training Local Outlier Factor...")
    sample = np.random.default_rng(0).choice(train, 40000, replace=False)
    lof = LocalOutlierFactor(n_neighbors=300, novelty=True, n_jobs=-1)
    lof.fit(features[sample])

    # 4. Compute cutoffs on validation set
    validation_labels = y.loc[valid]
    if_val_scores = -forest.score_samples(features[valid])
    lof_val_scores = -lof.score_samples(features[valid])

    if_cutoff = choose_cutoff(validation_labels, if_val_scores)
    lof_cutoff = choose_cutoff(validation_labels, lof_val_scores)

    # Save sample rows from test set so users can test real examples in the app
    sample_normal = df.loc[test][y.loc[test] == 0].iloc[0].to_dict()
    sample_fraud = df.loc[test][y.loc[test] == 1].iloc[0].to_dict()

    # Save everything
    os.makedirs("artifacts", exist_ok=True)
    artifacts = {
        "scaler": scaler,
        "all_columns": list(X.columns),
        "best_indices": best_indices,
        "selected_features": selected_feature_names,
        "forest": forest,
        "lof": lof,
        "if_cutoff": if_cutoff,
        "lof_cutoff": lof_cutoff,
        "sample_normal": sample_normal,
        "sample_fraud": sample_fraud,
    }
    joblib.dump(artifacts, "artifacts/fraud_artifacts.joblib")
    print("Saved artifacts successfully to artifacts/fraud_artifacts.joblib!")

if __name__ == "__main__":
    main()
