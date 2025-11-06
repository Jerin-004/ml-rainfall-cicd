import pandas as pd

YES = {"yes", "y", "1", "true", "t"}
NO = {"no", "n", "0", "false", "f"}

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "rainfall" not in df.columns:
        raise ValueError("rainfall column not found")
    df["rainfall"] = df["rainfall"].astype(str).str.lower().str.strip().map(
        lambda v: 1 if v in YES else (0 if v in NO else v)
    )
    if df["rainfall"].dtype == object:
        raise ValueError("rainfall must be yes/no or 1/0")
    return df

def split_xy(df: pd.DataFrame):
    y = df["rainfall"].astype(int)
    X = df.drop(columns=["rainfall"])
    return X, y
