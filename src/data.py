"""
Loading the data and putting it into a SQLite database.

Dataset: "Credit Card Fraud Detection" (European cardholders, September 2013,
ULB Machine Learning Group). 284,807 card transactions over two days, 492 frauds (0.17%).
Columns: V1..V28 (anonymised PCA components), Amount (EUR), Class (1 = fraud).
The rows are in chronological order (the OpenML copy drops the original Time
column, so the row order is used as the time order).
"""

import os
import sqlite3

import numpy as np
import pandas as pd

CSV_PATH = "data/creditcard.csv"
DB_PATH = "data/transactions.db"


def load_raw():
    """Read the CSV if it is already in data/, otherwise download it from OpenML (id 1597)."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        from sklearn.datasets import fetch_openml

        print("Downloading the dataset from OpenML (about 150 MB, only the first time)...")
        df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
        df.to_csv(CSV_PATH, index=False)
    df["Class"] = df["Class"].astype(int)
    return df


def build_database(df, db_path=DB_PATH):
    """Write the transactions into a SQLite table, adding an id that keeps the time order."""
    out = df.drop(columns=["Time"], errors="ignore").copy()
    out.insert(0, "id", np.arange(len(out)))  # row order = time order
    with sqlite3.connect(db_path) as conn:
        out.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON transactions(id)")
    return db_path


def connect(db_path=DB_PATH):
    return sqlite3.connect(db_path)
