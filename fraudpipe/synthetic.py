"""Synthetic PaySim-schema generator (used when the real 6.3M-row CSV is absent).

It reproduces the PaySim columns, the fraud pattern (fraud only in TRANSFER /
CASH_OUT, account drain, mule rings) AND the post-transaction balance leakage
(e.g. newbalanceOrig == 0 and a stale newbalanceDest on fraud rows), so the
leakage-removal step in features.py is meaningful.

For the real dataset download PaySim from Kaggle
(ealaxi/paysim1 -> PS_20174392719_1491204439457_Log.csv) and pass --csv.
"""
import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]


def generate(n_tx: int = 300_000, fraud_rate: float = 0.0082, seed: int = 42,
             n_steps: int = 743) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud_pairs = max(int(n_tx * fraud_rate / 2), 4)
    n_legit = n_tx - 2 * n_fraud_pairs

    n_cust = max(n_tx // 3, 1000)
    cust = np.array([f"C{i}" for i in range(n_cust)])
    merch = np.array([f"M{i}" for i in range(max(n_cust // 20, 50))])
    bal = rng.lognormal(9.5, 1.6, n_cust)  # starting balances

    # ---- legit traffic -------------------------------------------------
    t = rng.choice(TYPES, n_legit, p=[0.34, 0.22, 0.35, 0.08, 0.01])
    step = np.sort(rng.integers(1, n_steps + 1, n_legit))
    o = rng.integers(0, n_cust, n_legit)
    d_c = rng.integers(0, n_cust, n_legit)
    old_o = bal[o] * rng.uniform(0.2, 1.5, n_legit)
    amt = np.minimum(rng.lognormal(8.5, 1.4, n_legit), old_o * rng.uniform(0.05, 1.0, n_legit))
    # a few legit users also empty their account -> drain alone is not a perfect signal
    drain = (rng.random(n_legit) < 0.015) & np.isin(t, ["TRANSFER", "CASH_OUT"])
    amt = np.where(drain, old_o, amt)
    dest = np.where(t == "PAYMENT", rng.choice(merch, n_legit), cust[d_c])
    old_d = np.where(t == "PAYMENT", 0.0, bal[d_c] * rng.uniform(0.0, 1.5, n_legit))
    old_d = np.where(rng.random(n_legit) < 0.25, 0.0, old_d)
    legit = pd.DataFrame(dict(step=step, type=t, amount=amt, nameOrig=cust[o],
                              oldbalanceOrg=old_o, nameDest=dest, oldbalanceDest=old_d,
                              isFraud=0))

    # ---- fraud rings: victim --TRANSFER--> mule --CASH_OUT--> cash ------
    n_mules = max(n_fraud_pairs // 6, 3)
    mules = np.array([f"C9{i:07d}" for i in range(n_mules)])
    victims = np.array([f"C8{i:07d}" for i in range(n_fraud_pairs)])
    f_step = np.sort(rng.integers(1, n_steps, n_fraud_pairs))
    v_bal = rng.lognormal(10.5, 1.5, n_fraud_pairs)
    frac = np.where(rng.random(n_fraud_pairs) < 0.85, 1.0, rng.uniform(0.5, 0.99, n_fraud_pairs))
    f_amt = v_bal * frac
    mule = rng.choice(mules, n_fraud_pairs)
    tr = pd.DataFrame(dict(step=f_step, type="TRANSFER", amount=f_amt, nameOrig=victims,
                           oldbalanceOrg=v_bal, nameDest=mule, oldbalanceDest=0.0, isFraud=1))
    co = pd.DataFrame(dict(step=f_step + rng.integers(0, 3, n_fraud_pairs), type="CASH_OUT",
                           amount=f_amt, nameOrig=mule, oldbalanceOrg=f_amt,
                           nameDest=[f"C7{i:07d}" for i in range(n_fraud_pairs)],
                           oldbalanceDest=0.0, isFraud=1))

    df = pd.concat([legit, tr, co], ignore_index=True).sort_values("step", kind="stable")
    df = df.reset_index(drop=True)

    # ---- PaySim-style post-transaction balances (the leaky columns) -----
    df["newbalanceOrig"] = np.maximum(df.oldbalanceOrg - df.amount, 0.0)
    df["newbalanceDest"] = df.oldbalanceDest + df.amount
    is_m = df.nameDest.str.startswith("M")
    df.loc[is_m, "newbalanceDest"] = 0.0
    # fraud rows: destination balance often not updated (known PaySim artefact)
    stale = (df.isFraud == 1) & (rng.random(len(df)) < 0.9)
    df.loc[stale, "newbalanceDest"] = df.loc[stale, "oldbalanceDest"]
    df["isFlaggedFraud"] = ((df.type == "TRANSFER") & (df.amount > 200_000) & (df.isFraud == 1)
                            & (rng.random(len(df)) < 0.002)).astype(int)
    cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    return df[cols]
