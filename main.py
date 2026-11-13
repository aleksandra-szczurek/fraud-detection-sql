"""
Card fraud detection with SQL + scikit-learn.

Run:  python main.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data import load_raw, build_database, connect
from src.sql_analysis import run_all, load_model_table
from src.model import make_features, time_split, build_models
from src.evaluate import best_threshold, summary, business_cost, REVIEW_COST

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def main():
    # 1. Data -> SQLite
    df = load_raw()
    build_database(df)
    conn = connect()

    # 2. Exploratory analysis in SQL
    for name, table in run_all(conn).items():
        print(f"\n--- {name} ---")
        print(table.to_string(index=False))

    # 3. Modelling table (ordered in time) and chronological split
    data = load_model_table(conn)
    conn.close()
    X, y, amount = make_features(data)
    tr, va, te = time_split(len(y))
    print(f"\nTrain {tr.stop:,} | validation {va.stop - va.start:,} | test {te.stop - te.start:,} transactions")
    print(f"Frauds: train {y[tr].sum()}, validation {y[va].sum()}, test {y[te].sum()}")

    # 4. Train, choose the threshold on validation, evaluate on test
    rows, curves = [], {}
    for name, model in build_models().items():
        model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        p_te = model.predict_proba(X.iloc[te])[:, 1]
        thr, grid, costs = best_threshold(y[va], p_va, amount[va])
        res = summary(y[te], p_te, amount[te], thr)
        res["cost at 0.5"] = business_cost(y[te], p_te, amount[te], 0.5)
        rows.append({"model": name, **res})
        curves[name] = (p_te, grid, costs, thr)

    results = pd.DataFrame(rows).set_index("model")
    print(f"\n=== Test set (last 20% of time), review cost = {REVIEW_COST:.0f} EUR per false alarm ===")
    print(results.round(3).T.to_string())
    no_model = float(np.sum(amount[te][y[te] == 1]))
    print(f"\nCost with no model (every fraud missed): {no_model:,.0f} EUR")

    # 5. Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (p_te, grid, costs, thr) in curves.items():
        prec, rec, _ = precision_recall_curve(y[te], p_te)
        axes[0].plot(rec, prec, lw=2, label=f"{name} (PR-AUC {results.loc[name, 'PR-AUC']:.2f})")
        axes[1].plot(grid, costs, lw=2, label=f"{name} (best threshold {thr:.2f})")
    axes[0].set(xlabel="Recall (share of frauds caught)", ylabel="Precision (share of alerts that are fraud)",
                title="Precision-recall curve - test set")
    axes[1].set(xlabel="Decision threshold", ylabel="Business cost on validation (EUR)",
                title="Choosing the threshold by cost")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/fraud_results.png", dpi=150)
    plt.show()
    print("\nPlot saved to results/fraud_results.png")


if __name__ == "__main__":
    main()
