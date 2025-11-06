import argparse, os, json, joblib
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .data import load_dataset, split_xy
from .model import build_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.data)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y if y.nunique()>1 else None
    )

    pipe = build_pipeline(X.columns)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_train)
    acc = float(accuracy_score(y_train, y_pred))

    joblib.dump(pipe, os.path.join(args.out, "model.joblib"))
    np.save(os.path.join(args.out, "X_test.npy"), X_test.values)
    np.save(os.path.join(args.out, "y_test.npy"), y_test.values)

    meta = {
        "trained_at": datetime.utcnow().isoformat()+"Z",
        "features": list(X.columns),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_accuracy": acc,
        "model": "LogisticRegression"
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
