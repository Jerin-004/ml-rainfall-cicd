import argparse, os, json, joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()

    model = joblib.load(os.path.join(args.artifacts, "model.joblib"))
    X_test = np.load(os.path.join(args.artifacts, "X_test.npy"))
    y_test = np.load(os.path.join(args.artifacts, "y_test.npy"))

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    with open(os.path.join(args.artifacts, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.artifacts, "report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
