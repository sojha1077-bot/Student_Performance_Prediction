"""
data_utils.py
-------------
Everything to do with READING and CLEANING the student CSV file.

In simple words:
- load_csv()            -> opens your CSV file and turns it into a table (DataFrame)
- guess_target_column() -> guesses which column is the "score" we want to predict
- clean_data()          -> fixes common problems (empty cells, duplicates, Yes/No text ...)
- dataset_summary()     -> makes a small table that describes every column
"""

import io
import os

import numpy as np
import pandas as pd

# Words that mean "Yes" or "No" in a CSV file. These get turned into 1 and 0,
# because a Linear Regression model can only do maths with numbers.
YES_VALUES = {"yes", "y", "true", "t"}
NO_VALUES = {"no", "n", "false", "f"}

# Text that really means "nothing here" (a missing value).
MISSING_MARKERS = {"", "?", "-", "--", "na", "n/a", "nan", "null", "none", "missing"}

# Words that usually appear in the name of the score column.
# The first match wins, so the most specific words come first.
TARGET_KEYWORDS = [
    "performance index",
    "performance",
    "final grade",
    "final score",
    "final marks",
    "exam score",
    "total score",
    "g3",
    "score",
    "marks",
    "grade",
    "result",
    "cgpa",
    "gpa",
    "percentage",
]

# Column names that are just labels (like a roll number). They do not help predict.
ID_NAMES = {
    "id", "student", "student id", "studentid", "student name", "name",
    "roll", "roll no", "rollno", "roll number", "serial", "serial no",
    "s no", "sno", "sr no", "srno", "index",
}


def _normalise_name(name) -> str:
    """'Student_ID ' -> 'student id' (lower case, underscores and dots become spaces)."""
    text = str(name).strip().lower().replace("_", " ").replace(".", " ")
    return " ".join(text.split())


def is_id_like(column_name) -> bool:
    """True if a column looks like an ID / name / serial number column."""
    name = _normalise_name(column_name)
    if name.startswith("unnamed"):
        return True
    return name in ID_NAMES or name.endswith(" id")


def load_csv(file) -> pd.DataFrame:
    """
    Read a CSV file and return it as a pandas DataFrame (a table).

    `file` can be:
    - a file uploaded in Streamlit, or
    - a path on your computer, e.g. "data/sample_students.csv"
    """
    if hasattr(file, "getvalue"):          # Streamlit upload
        raw = file.getvalue()
    elif isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as handle:
            raw = handle.read()
    else:
        raise ValueError("Could not read this file. Please upload a .csv file.")

    if not raw or not raw.strip():
        raise ValueError("The file is empty. Please choose a CSV file that has data in it.")

    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            # Some CSV files use ';' instead of ',' between values.
            if df.shape[1] == 1 and b";" in raw[:5000]:
                df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=";")
            break
        except UnicodeDecodeError as err:
            last_error = err
            continue
        except pd.errors.ParserError as err:
            raise ValueError(f"This does not look like a valid CSV file ({err}).") from err
    else:
        raise ValueError(f"Could not read the text in this file ({last_error}).")

    if df.empty:
        raise ValueError("The CSV file has column names but no rows of data.")
    if df.shape[1] < 2:
        raise ValueError(
            "The CSV file needs at least 2 columns: one score column and at least one other column."
        )
    return df


def _is_text_column(series: pd.Series) -> bool:
    """True for text columns (works for both old and new versions of pandas)."""
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _to_number(series: pd.Series) -> pd.Series:
    """Turn text like ' 1,234 ' into the number 1234. Things that are not numbers become empty."""
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def numeric_like_columns(df: pd.DataFrame, threshold: float = 0.9) -> list:
    """
    Columns that are numbers, or text that is (almost) all numbers.
    Yes/No and True/False columns are NOT counted here.
    """
    columns = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            columns.append(col)
            continue
        non_empty = series.dropna()
        if len(non_empty) == 0:
            continue
        if _to_number(non_empty).notna().mean() >= threshold:
            columns.append(col)
    return columns


def guess_target_column(df: pd.DataFrame):
    """Guess which number column is the score to predict (the user can change it)."""
    candidates = [c for c in numeric_like_columns(df) if not is_id_like(c)]
    if not candidates:
        return None
    for keyword in TARGET_KEYWORDS:
        for col in candidates:
            if keyword in _normalise_name(col):
                return col
    return candidates[-1]   # in most datasets the score is the last column


def yes_no_to_number(series: pd.Series) -> pd.Series:
    """Yes/True -> 1, No/False -> 0. Numbers stay numbers. Anything else becomes empty."""

    def convert(value):
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in YES_VALUES:
                return 1
            if text in NO_VALUES:
                return 0
            try:
                return float(text)
            except ValueError:
                return np.nan
        return value

    return pd.to_numeric(series.map(convert), errors="coerce")


def _is_yes_no_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if not _is_text_column(series):
        return False
    values = {str(v).strip().lower() for v in series.dropna().unique()}
    return bool(values) and values <= (YES_VALUES | NO_VALUES)


def clean_data(
    df: pd.DataFrame,
    target: str,
    remove_duplicates: bool = True,
    remove_outliers: bool = False,
):
    """
    Clean the data step by step and write down what was done.

    Returns three things:
    1. the cleaned table
    2. a list of sentences explaining every change (shown on the dashboard)
    3. extra info the model needs later (which columns were Yes/No)
    """
    report = []
    info = {"yes_no_columns": [], "dropped_columns": []}
    data = df.copy()
    target = str(target).strip()

    # Step 1: tidy the column names (remove extra spaces at the start/end).
    data.columns = [str(c).strip() for c in data.columns]
    report.append("Removed extra spaces from the column names.")

    # Step 2: drop leftover index columns such as 'Unnamed: 0'.
    unnamed = [c for c in data.columns if c.lower().startswith("unnamed")]
    if unnamed:
        data = data.drop(columns=unnamed)
        info["dropped_columns"].extend(unnamed)
        report.append(f"Removed {len(unnamed)} empty index column(s): {', '.join(unnamed)}.")

    # Step 3: tidy text cells and mark fake values like '?' or 'N/A' as missing.
    for col in data.columns:
        if _is_text_column(data[col]):
            cleaned = data[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            cleaned = cleaned.map(
                lambda v: np.nan if isinstance(v, str) and v.lower() in MISSING_MARKERS else v
            )
            data[col] = cleaned
    report.append("Removed extra spaces inside text cells and treated '?', 'N/A', 'null' etc. as missing.")

    # Step 4: remove rows and columns that are completely empty.
    rows_before = len(data)
    data = data.dropna(how="all")
    empty_rows = rows_before - len(data)
    if empty_rows:
        report.append(f"Removed {empty_rows} completely empty row(s).")
    empty_cols = [c for c in data.columns if data[c].isna().all()]
    if empty_cols:
        data = data.drop(columns=empty_cols)
        info["dropped_columns"].extend(empty_cols)
        report.append(f"Removed {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}.")

    if target not in data.columns:
        raise ValueError(f"The score column '{target}' was not found after cleaning.")

    # Step 5: numbers that were saved as text are turned back into real numbers.
    for col in numeric_like_columns(data):
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = _to_number(data[col])
            report.append(f"Converted '{col}' from text into numbers.")

    # Step 6: Yes/No (or True/False) columns become 1/0.
    for col in list(data.columns):
        if col != target and _is_yes_no_column(data[col]):
            data[col] = yes_no_to_number(data[col])
            info["yes_no_columns"].append(col)
            report.append(f"Changed '{col}' from Yes/No into 1/0 (Yes = 1, No = 0).")

    # Step 7: the score column must be numbers.
    if not pd.api.types.is_numeric_dtype(data[target]):
        raise ValueError(
            f"The score column '{target}' contains text, not numbers. "
            "Please choose a column that holds the marks / score."
        )

    # Step 8: rows without a score cannot be used to learn, so they are removed.
    missing_target = int(data[target].isna().sum())
    if missing_target:
        data = data.dropna(subset=[target])
        report.append(f"Removed {missing_target} row(s) that had no value in the score column '{target}'.")

    # Step 9: remove exact duplicate rows (the same student copied twice).
    if remove_duplicates:
        dupes = int(data.duplicated().sum())
        if dupes:
            data = data.drop_duplicates()
            report.append(f"Removed {dupes} duplicate row(s).")
        else:
            report.append("No duplicate rows were found.")

    # Step 10: fill the remaining empty cells.
    filled_any = False
    for col in data.columns:
        missing = int(data[col].isna().sum())
        if not missing:
            continue
        filled_any = True
        if col in info["yes_no_columns"]:
            value = int(data[col].mode().iloc[0])
            data[col] = data[col].fillna(value)
            report.append(
                f"Filled {missing} empty cell(s) in '{col}' with the most common answer "
                f"'{'Yes' if value == 1 else 'No'}'."
            )
        elif pd.api.types.is_numeric_dtype(data[col]):
            value = data[col].median()
            data[col] = data[col].fillna(value)
            report.append(f"Filled {missing} empty cell(s) in '{col}' with the median (middle value) {value:g}.")
        else:
            mode = data[col].mode()
            value = mode.iloc[0] if not mode.empty else "Unknown"
            data[col] = data[col].fillna(value)
            report.append(f"Filled {missing} empty cell(s) in '{col}' with the most common value '{value}'.")
    if not filled_any:
        report.append("No empty cells were left to fill.")

    # Step 11 (optional): remove extreme values (outliers) using the IQR rule.
    if remove_outliers:
        rows_before = len(data)
        keep = pd.Series(True, index=data.index)
        for col in data.columns:
            series = data[col]
            if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10:
                continue  # skip text columns and small 0/1 or 1-5 style columns
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            keep &= series.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        data = data[keep]
        removed = rows_before - len(data)
        report.append(f"Removed {removed} row(s) with extreme values (outliers) using the IQR rule.")

    data = data.reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} row(s) are left after cleaning. At least 10 rows are needed to train a model."
        )
    report.append(f"Final clean data: {len(data)} rows and {data.shape[1]} columns.")
    return data, report, info


def split_feature_types(df: pd.DataFrame, features: list):
    """Split the chosen input columns into number columns and text (category) columns."""
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    categorical = [f for f in features if f not in numeric]
    return numeric, categorical


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """A simple table that describes every column in plain words."""
    rows = []
    for col in df.columns:
        series = df[col]
        non_empty = series.dropna()
        rows.append(
            {
                "Column": col,
                "Type": "Number" if pd.api.types.is_numeric_dtype(series) else "Text",
                "Empty cells": int(series.isna().sum()),
                "Different values": int(series.nunique()),
                "Example": str(non_empty.iloc[0]) if len(non_empty) else "",
            }
        )
    return pd.DataFrame(rows)
