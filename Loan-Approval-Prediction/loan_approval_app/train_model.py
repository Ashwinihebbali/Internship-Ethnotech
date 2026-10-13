import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "loan_approval_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Replace placeholder -100000 with NaN for numeric columns and impute median.
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].replace(-100000, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns (including target)
    cat_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders, list(cat_cols), list(numeric_cols)

if __name__ == "__main__":
    df, encoders, cat_cols, num_cols = load_data()
    X = df.drop(columns=["loan_status", "loan_id"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    payload = {
        "model": model,
        "encoders": encoders,
        "categorical": cat_cols,
        "numeric": num_cols,
        "feature_names": X.columns.tolist(),
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
