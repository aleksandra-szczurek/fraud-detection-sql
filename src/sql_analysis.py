"""
Run the named SQL queries in sql/analysis.sql and return them as pandas tables.
"""

import pandas as pd

SQL_PATH = "sql/analysis.sql"


def read_queries(path=SQL_PATH):
    """Split the .sql file into {name: query} using the '-- name:' markers."""
    queries, name, lines = {}, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-- name:"):
                if name:
                    queries[name] = "".join(lines)
                name, lines = line.split(":", 1)[1].strip(), []
            else:
                lines.append(line)
    if name:
        queries[name] = "".join(lines)
    return queries


def run_all(conn, path=SQL_PATH):
    return {name: pd.read_sql_query(q, conn) for name, q in read_queries(path).items()}


def load_model_table(conn):
    """The modelling table, ordered in time (we must never train on the future)."""
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY id", conn)
