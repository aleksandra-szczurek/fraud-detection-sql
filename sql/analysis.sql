-- name: overview
-- How many transactions, how many frauds, and how much money is involved
SELECT
    COUNT(*)                                   AS n_transactions,
    SUM(Class)                                 AS n_frauds,
    ROUND(100.0 * AVG(Class), 3)               AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 0) AS fraud_amount_eur
FROM transactions;

-- name: amount_by_class
-- Are frauds bigger or smaller than normal payments?
SELECT
    CASE WHEN Class = 1 THEN 'fraud' ELSE 'legitimate' END AS type,
    COUNT(*)                     AS n,
    ROUND(AVG(Amount), 2)        AS avg_amount,
    ROUND(MAX(Amount), 2)        AS max_amount,
    ROUND(100.0 * AVG(CASE WHEN Amount < 1 THEN 1 ELSE 0 END), 1) AS pct_under_1_eur
FROM transactions
GROUP BY Class;

-- name: fraud_by_amount_band
-- Fraud rate by size of the payment
SELECT
    CASE
        WHEN Amount < 1    THEN '1) < 1 EUR'
        WHEN Amount < 10   THEN '2) 1-10 EUR'
        WHEN Amount < 100  THEN '3) 10-100 EUR'
        WHEN Amount < 500  THEN '4) 100-500 EUR'
        ELSE                    '5) 500+ EUR'
    END                              AS amount_band,
    COUNT(*)                         AS n_transactions,
    SUM(Class)                       AS n_frauds,
    ROUND(100.0 * AVG(Class), 3)     AS fraud_rate_pct
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;

-- name: fraud_over_time
-- Fraud rate in ten consecutive blocks of transactions (rows are in time order)
SELECT
    1 + id * 10 / (SELECT COUNT(*) FROM transactions) AS time_block,
    COUNT(*)                         AS n_transactions,
    SUM(Class)                       AS n_frauds,
    ROUND(100.0 * AVG(Class), 3)     AS fraud_rate_pct
FROM transactions
GROUP BY time_block
ORDER BY time_block;

-- name: fraud_trend_cumulative
-- CTE + window functions: frauds per time block, running total and change vs the previous block
WITH blocks AS (
    SELECT
        1 + id * 10 / (SELECT COUNT(*) FROM transactions) AS time_block,
        Class,
        Amount
    FROM transactions
),
per_block AS (
    SELECT
        time_block,
        SUM(Class)                                            AS n_frauds,
        ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 0) AS fraud_eur
    FROM blocks
    GROUP BY time_block
)
SELECT
    time_block,
    n_frauds,
    n_frauds - LAG(n_frauds) OVER (ORDER BY time_block)       AS change_vs_prev,
    SUM(n_frauds) OVER (ORDER BY time_block)                  AS cumulative_frauds,
    SUM(fraud_eur) OVER (ORDER BY time_block)                 AS cumulative_fraud_eur
FROM per_block
ORDER BY time_block;

-- name: top_frauds_per_band
-- ROW_NUMBER() OVER (PARTITION BY ...): the 3 largest frauds in each amount band
WITH frauds AS (
    SELECT
        id,
        Amount,
        CASE
            WHEN Amount < 10   THEN '1) < 10 EUR'
            WHEN Amount < 100  THEN '2) 10-100 EUR'
            WHEN Amount < 500  THEN '3) 100-500 EUR'
            ELSE                    '4) 500+ EUR'
        END AS amount_band
    FROM transactions
    WHERE Class = 1
),
ranked AS (
    SELECT
        amount_band,
        id,
        Amount,
        ROW_NUMBER() OVER (PARTITION BY amount_band ORDER BY Amount DESC) AS rank_in_band
    FROM frauds
)
SELECT amount_band, rank_in_band, id AS transaction_id, Amount
FROM ranked
WHERE rank_in_band <= 3
ORDER BY amount_band, rank_in_band;

-- name: blocks_vs_average
-- CTE: which periods are above the overall fraud rate, and by how much
WITH per_block AS (
    SELECT
        1 + id * 10 / (SELECT COUNT(*) FROM transactions) AS time_block,
        AVG(Class) AS fraud_rate
    FROM transactions
    GROUP BY time_block
),
overall AS (
    SELECT AVG(Class) AS avg_rate FROM transactions
)
SELECT
    p.time_block,
    ROUND(100.0 * p.fraud_rate, 3)                  AS fraud_rate_pct,
    ROUND(100.0 * o.avg_rate, 3)                    AS overall_pct,
    ROUND(p.fraud_rate / o.avg_rate, 2)             AS ratio_to_average,
    CASE WHEN p.fraud_rate > o.avg_rate THEN 'above' ELSE 'below' END AS vs_average,
    RANK() OVER (ORDER BY p.fraud_rate DESC)        AS risk_rank
FROM per_block p
CROSS JOIN overall o
ORDER BY p.time_block;
