"""Hypothetical client presets. Values follow the UCI schema; predictions are not stored."""

from src.constants import FEATURE_COLUMNS

# Constructed examples, not copied from identifiable people.
PRESETS = {
    "The Consistent Payer": {
        "LIMIT_BAL": 180000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 36,
        "PAY_0": -1,
        "PAY_2": -1,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -1,
        "PAY_6": -1,
        "BILL_AMT1": 4200,
        "BILL_AMT2": 3900,
        "BILL_AMT3": 4100,
        "BILL_AMT4": 3800,
        "BILL_AMT5": 3600,
        "BILL_AMT6": 3400,
        "PAY_AMT1": 4200,
        "PAY_AMT2": 3900,
        "PAY_AMT3": 4100,
        "PAY_AMT4": 3800,
        "PAY_AMT5": 3600,
        "PAY_AMT6": 3400,
    },
    "The Irregular Payer": {
        "LIMIT_BAL": 50000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 29,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 2,
        "PAY_5": 0,
        "PAY_6": 1,
        "BILL_AMT1": 28500,
        "BILL_AMT2": 27200,
        "BILL_AMT3": 25100,
        "BILL_AMT4": 24800,
        "BILL_AMT5": 23000,
        "BILL_AMT6": 22100,
        "PAY_AMT1": 0,
        "PAY_AMT2": 1200,
        "PAY_AMT3": 0,
        "PAY_AMT4": 800,
        "PAY_AMT5": 1500,
        "PAY_AMT6": 0,
    },
    "The High-Balance Client": {
        "LIMIT_BAL": 500000,
        "SEX": 1,
        "EDUCATION": 1,
        "MARRIAGE": 1,
        "AGE": 44,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 410000,
        "BILL_AMT2": 398000,
        "BILL_AMT3": 387000,
        "BILL_AMT4": 372000,
        "BILL_AMT5": 365000,
        "BILL_AMT6": 351000,
        "PAY_AMT1": 15000,
        "PAY_AMT2": 14000,
        "PAY_AMT3": 16000,
        "PAY_AMT4": 12000,
        "PAY_AMT5": 11000,
        "PAY_AMT6": 13000,
    },
}


def default_client() -> dict:
    return dict(PRESETS["The Consistent Payer"])


def ensure_complete(record: dict) -> dict:
    base = default_client()
    base.update({k: record[k] for k in FEATURE_COLUMNS if k in record})
    return base
