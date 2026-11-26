"""
model.py
--------
Everything to do with the LINEAR REGRESSION model.

In simple words:
- Linear Regression draws the "best straight line" through the data.
  With many inputs the line becomes an equation like:
      Score = intercept + (c1 x Hours Studied) + (c2 x Previous Scores) + ...
- train_and_evaluate() -> teaches the model with 80% of the students and
                          tests it on the other 20% it has never seen
- predict_scores()     -> uses the learned equation to predict new scores
- save / load          -> stores the trained model in the 'models' folder
"""

import io
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_utils import split_feature_types, yes_no_to_number

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.joblib")


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    A 'pipeline' is a chain of steps that always run in the same order:
    1. Prepare the inputs: number columns go in as they are; text columns
       (like gender) are turned into 0/1 columns ("one-hot encoding").
    2. Linear Regression learns the equation.
    """
    steps = []
    if numeric_features:
        steps.append(("numbers", "passthrough", numeric_features))
    if categorical_features:
        steps.append(
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )
    prepare = ColumnTransformer(steps, verbose_feature_names_out=False)
    return Pipeline([("prepare", prepare), ("model", LinearRegression())])


def build_feature_info(df: pd.DataFrame, numeric: list, categorical: list, yes_no_columns: list) -> dict:
    """Remember the range of every input so the prediction form can show sensible boxes."""
    info = {}
    for col in numeric:
        series = df[col]
        is_whole = bool(np.all(np.isclose(series, series.round())))
        info[col] = {
            "kind": "yes_no" if col in yes_no_columns else "number",
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "whole_numbers": is_whole,
        }
    for col in categorical:
        values = sorted(df[col].astype(str).unique().tolist())
        info[col] = {
            "kind": "category",
            "options": values,
            "mode": str(df[col].astype(str).mode().iloc[0]),
        }
    return info


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    features: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train the Linear Regression model and measure how good it is."""
    if not features:
        raise ValueError("Please choose at least one input column.")
    if target in features:
        raise ValueError("The score column cannot also be an input column.")

    data = df[features + [target]].dropna()
    if len(data) < 10:
        raise ValueError("At least 10 rows are needed to train the model.")

    X = data[features]
    y = data[target]
    numeric, categorical = split_feature_types(data, features)

    # Split: one part to learn from (train), one part to check on (test).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    metrics = {
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    # Cross-validation: repeat the train/test check 5 times on different splits.
    if len(data) >= 25:
        folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(build_pipeline(numeric, categorical), X, y, cv=folds, scoring="r2")
        metrics["cv_r2_mean"] = float(scores.mean())
        metrics["cv_r2_std"] = float(scores.std())

    model = pipeline.named_steps["model"]
    feature_names = list(pipeline.named_steps["prepare"].get_feature_names_out())
    coefficients = pd.DataFrame(
        {"Input": feature_names, "Coefficient": model.coef_}
    )
    coefficients["Effect"] = np.where(
        coefficients["Coefficient"] >= 0, "Raises the score", "Lowers the score"
    )
    coefficients = coefficients.reindex(
        coefficients["Coefficient"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    test_table = X_test.copy()
    test_table[f"Actual {target}"] = y_test.values
    test_table[f"Predicted {target}"] = np.round(y_pred_test, 2)
    test_table["Error"] = np.round(y_test.values - y_pred_test, 2)

    return {
        "pipeline": pipeline,
        "target": target,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "metrics": metrics,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "y_test": y_test.to_numpy(),
        "y_pred_test": y_pred_test,
        "test_table": test_table.reset_index(drop=True),
        "train_data": data,
    }


def make_equation(results: dict, max_terms: int = 8) -> str:
    """Write the learned equation in plain text."""
    parts = [f"{results['intercept']:.2f}"]
    coefs = results["coefficients"].head(max_terms)
    for _, row in coefs.iterrows():
        sign = "+" if row["Coefficient"] >= 0 else "-"
        parts.append(f"{sign} {abs(row['Coefficient']):.3f} x [{row['Input']}]")
    extra = len(results["coefficients"]) - max_terms
    tail = f"  ... (+{extra} more terms)" if extra > 0 else ""
    return f"{results['target']} = " + " ".join(parts) + tail


def make_bundle(results: dict, yes_no_columns: list, source_name: str) -> dict:
    """Pack everything needed for predictions into one object we can save."""
    data = results["train_data"]
    return {
        "pipeline": results["pipeline"],
        "target": results["target"],
        "features": results["features"],
        "numeric": results["numeric"],
        "categorical": results["categorical"],
        "yes_no_columns": [c for c in yes_no_columns if c in results["features"]],
        "feature_info": build_feature_info(
            data, results["numeric"], results["categorical"], yes_no_columns
        ),
        "target_min": float(data[results["target"]].min()),
        "target_max": float(data[results["target"]].max()),
        "metrics": results["metrics"],
        "source_name": source_name,
    }


def prepare_new_data(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Clean a NEW file of students the same way the training data was cleaned."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    missing = [f for f in bundle["features"] if f not in data.columns]
    if missing:
        raise ValueError(
            "These columns are missing from the new file: " + ", ".join(missing)
        )
    data = data[bundle["features"]].copy()
    for col in bundle["numeric"]:
        info = bundle["feature_info"][col]
        data[col] = yes_no_to_number(data[col]).fillna(info["median"])
    for col in bundle["categorical"]:
        info = bundle["feature_info"][col]
        series = data[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        data[col] = series.fillna(info["mode"]).astype(str)
    return data


def predict_scores(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Use the trained equation to predict scores."""
    return bundle["pipeline"].predict(X[bundle["features"]])


def explain_prediction(bundle: dict, X_row: pd.DataFrame) -> pd.DataFrame:
    """Show how much each input pushed ONE predicted score up or down."""
    pipeline = bundle["pipeline"]
    prepared = pipeline.named_steps["prepare"].transform(X_row[bundle["features"]])
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    effect = np.asarray(prepared)[0] * coefs
    table = pd.DataFrame(
        {"Input": names, "Value used": np.asarray(prepared)[0], "Coefficient": coefs, "Points added": effect}
    )
    table = table[table["Value used"] != 0]   # hide category columns that are switched off
    return table.reindex(table["Points added"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def save_model_bundle(bundle: dict, path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    return path


def bundle_to_bytes(bundle: dict) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def load_model_bundle(path: str = MODEL_PATH):
    """Load a saved model. Returns None if there is no saved model yet."""
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception:   # an old or broken file -> just train again
        return None
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        return None
    return bundle
