from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_pipeline(feature_names):
    numeric_features = list(feature_names)

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features)
    ], remainder="drop")

    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    return pipe
