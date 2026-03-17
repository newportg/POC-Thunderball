from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["draw_date", "n1", "n2", "n3", "n4", "n5", "thunderball"]
OFFICIAL_COLUMN_MAP = {
    "DrawDate": "draw_date",
    "Ball 1": "n1",
    "Ball 2": "n2",
    "Ball 3": "n3",
    "Ball 4": "n4",
    "Ball 5": "n5",
    "Thunderball": "thunderball",
}


class DataValidationError(ValueError):
    pass


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Support the official export format while preserving compatibility with internal sample data.
    if "draw_date" in df.columns and all(col in df.columns for col in ["n1", "n2", "n3", "n4", "n5", "thunderball"]):
        return df

    mapped = df.rename(columns=OFFICIAL_COLUMN_MAP)
    return mapped


def _validate_ranges(df: pd.DataFrame) -> None:
    for col in ["n1", "n2", "n3", "n4", "n5"]:
        if not df[col].between(1, 39).all():
            raise DataValidationError(f"Column {col} has values outside 1..39")

    if not df["thunderball"].between(1, 14).all():
        raise DataValidationError("Column thunderball has values outside 1..14")


def _validate_unique_main_numbers(df: pd.DataFrame) -> None:
    for idx, row in df.iterrows():
        values = [int(row["n1"]), int(row["n2"]), int(row["n3"]), int(row["n4"]), int(row["n5"])]
        if len(set(values)) != 5:
            raise DataValidationError(
                f"Row index {idx} has duplicate main numbers. Each draw must contain 5 unique numbers."
            )


def load_draw_history(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    _ensure_columns(df)

    # Keep only the known schema so extra CSV columns do not break downstream logic.
    df = df[EXPECTED_COLUMNS].copy()
    # Official downloads use values like 14-Mar-2026; infer mixed formats safely.
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce", dayfirst=True)

    if df["draw_date"].isna().any():
        raise DataValidationError("draw_date contains invalid values. Use ISO date format YYYY-MM-DD.")

    for col in ["n1", "n2", "n3", "n4", "n5", "thunderball"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["n1", "n2", "n3", "n4", "n5", "thunderball"]].isna().any().any():
        raise DataValidationError("Number columns contain non-numeric values.")

    for col in ["n1", "n2", "n3", "n4", "n5", "thunderball"]:
        df[col] = df[col].astype(int)

    _validate_ranges(df)
    _validate_unique_main_numbers(df)

    return df.sort_values("draw_date", ascending=False).reset_index(drop=True)
