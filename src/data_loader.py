"""Load and validate the UCI Default of Credit Card Clients dataset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.constants import (
    DATA_DIR,
    FEATURE_COLUMNS,
    ID_COLUMN,
    LOCAL_CSV_NAME,
    TARGET_COLUMN,
    UCI_DOWNLOAD_URL,
    UCI_XLS_NAME,
)

REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded or fails schema checks."""


def _read_uci_excel(path: Path) -> pd.DataFrame:
    # Official UCI file uses two header rows; named columns sit on row 2.
    return pd.read_excel(path, header=1)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns[: len(REQUIRED_COLUMNS)]) != REQUIRED_COLUMNS:
        # Some mirrors keep a blank first column from the Excel export.
        if df.columns[0] in {"Unnamed: 0", ""} and list(df.columns[1 : 1 + len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS:
            df = df.iloc[:, 1 : 1 + len(REQUIRED_COLUMNS)].copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {missing}")
    out = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_dataset(df: pd.DataFrame) -> dict:
    """Return a quality report. Raises DatasetError on fatal schema issues."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise DatasetError("Dataset is empty.")
    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise DatasetError("Unexpected missing values in features or target.")
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise DatasetError("Target must be binary {0, 1}.")
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features": len(FEATURE_COLUMNS),
        "n_duplicates": int(df.duplicated().sum()),
        "n_duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
        "class_counts": {
            int(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "positive_rate": float(df[TARGET_COLUMN].mean()),
    }


def save_local_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATA_DIR / LOCAL_CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def download_uci_dataset(dest_dir: Path | None = None, timeout: int = 60) -> Path:
    """Download the official UCI zip, extract the .xls, and return its path."""
    dest_dir = dest_dir or DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(UCI_DOWNLOAD_URL, timeout=timeout) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        xls_name = next((n for n in names if n.lower().endswith(".xls")), None)
        if xls_name is None:
            raise DatasetError(f"UCI zip did not contain an .xls file. Contents: {names}")
        archive.extract(xls_name, dest_dir)
        extracted = dest_dir / xls_name
    target = dest_dir / UCI_XLS_NAME
    if extracted.resolve() != target.resolve():
        extracted.replace(target)
    return target


def load_credit_dataset(*, download_if_missing: bool = True) -> pd.DataFrame:
    """Load the dataset from local CSV, local Excel, or the official UCI zip."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / LOCAL_CSV_NAME
    xls_candidates = [
        DATA_DIR / UCI_XLS_NAME,
        PROJECT_ROOT_XLS(),
    ]

    if csv_path.exists():
        df = _normalize(pd.read_csv(csv_path))
        validate_dataset(df)
        return df

    for path in xls_candidates:
        if path is not None and path.exists():
            df = _normalize(_read_uci_excel(path))
            validate_dataset(df)
            save_local_csv(df, csv_path)
            return df

    if not download_if_missing:
        raise DatasetError(
            "Dataset file not found. Place the UCI CSV or XLS in the data/ folder, "
            f"or allow a download from {UCI_DOWNLOAD_URL}."
        )

    xls_path = download_uci_dataset(DATA_DIR)
    df = _normalize(_read_uci_excel(xls_path))
    validate_dataset(df)
    save_local_csv(df, csv_path)
    return df


def PROJECT_ROOT_XLS() -> Path | None:
    from src.constants import PROJECT_ROOT

    extracted = PROJECT_ROOT / "_uci350_extract" / UCI_XLS_NAME
    return extracted if extracted.exists() else None


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictive features X and target y. ID is never used as a feature."""
    x = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return x, y
