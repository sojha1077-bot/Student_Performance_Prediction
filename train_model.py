"""
train_model.py
--------------
Train the model WITHOUT the dashboard (useful for checking things in the terminal).

Run:
    python train_model.py                                  # uses data/sample_students.csv
    python train_model.py --data data/your_kaggle_file.csv
    python train_model.py --data data/your_kaggle_file.csv --target "Performance Index"

It cleans the data, trains Linear Regression, prints the results and saves the
model to models/student_model.joblib (the dashboard's Predict page can use it).
"""

import argparse
import os
import sys

from src.data_utils import clean_data, guess_target_column, is_id_like, load_csv
from src.model import MODEL_PATH, make_bundle, make_equation, save_model_bundle, train_and_evaluate

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_students.csv")


def main():
    parser = argparse.ArgumentParser(description="Train the student score Linear Regression model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to the CSV file")
    parser.add_argument("--target", default=None, help="Name of the score column (guessed if not given)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Part kept for testing, e.g. 0.2 = 20%%")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"File not found: {args.data}")

    print(f"\n1) Loading {args.data}")
    raw = load_csv(args.data)
    print(f"   {raw.shape[0]} rows, {raw.shape[1]} columns")

    target = args.target or guess_target_column(raw)
    if target is None:
        sys.exit("Could not find a number column to predict. Use --target \"Column Name\".")
    print(f"   Score column (target): {target}")

    print("\n2) Cleaning")
    clean, report, info = clean_data(raw, target)
    for line in report:
        print("   -", line)
    target = str(target).strip()

    print("\n3) Training Linear Regression")
    features = [c for c in clean.columns if c != target and not is_id_like(c)]
    print("   Inputs:", ", ".join(features))
    results = train_and_evaluate(clean, target, features, test_size=args.test_size)

    m = results["metrics"]
    print("\n4) Results on test data")
    print(f"   R2 score : {m['r2_test']:.4f}")
    print(f"   MAE      : {m['mae']:.4f}")
    print(f"   MSE      : {m['mse']:.4f}")
    print(f"   RMSE     : {m['rmse']:.4f}")
    if "cv_r2_mean" in m:
        print(f"   5-fold cross-validation R2: {m['cv_r2_mean']:.4f} +/- {m['cv_r2_std']:.4f}")
    print("\n   Equation:")
    print("  ", make_equation(results))

    bundle = make_bundle(results, info["yes_no_columns"], os.path.basename(args.data))
    path = save_model_bundle(bundle, MODEL_PATH)
    print(f"\n5) Model saved to {path}\n")


if __name__ == "__main__":
    main()
