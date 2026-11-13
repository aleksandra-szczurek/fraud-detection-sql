"""
Run only the SQL analysis (no model training).

Run:  python run_sql.py
Needs data/transactions.db, created the first time you run main.py.
"""

import pandas as pd

from src.data import connect
from src.sql_analysis import run_all

pd.set_option("display.width", 120)

conn = connect()
for name, table in run_all(conn).items():
    print(f"\n--- {name} ---")
    print(table.to_string(index=False))
conn.close()
