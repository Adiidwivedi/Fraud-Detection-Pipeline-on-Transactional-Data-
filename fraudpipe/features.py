"""Leakage-free tabular feature engineering.

Leakage: ``newbalanceOrig`` / ``newbalanceDest`` are written *after* the
transaction (and PaySim's fraud logic zeroes / forgets them), so they encode the
label. They are dropped and replaced by features built only from pre-transaction
state and strictly *past* activity.
"""
import numpy as np
import pandas as pd

TYPE_MAP = {"PAYMENT": 0, "CASH_IN": 1, "CASH_OUT": 2, "TRANSFER": 3, "DEBIT": 4}
LEAKY_COLS = ["newbalanceOrig", "newbalanceDest"]
VELOCITY_WINDOW = 24  # steps (hours)

TABULAR_FEATURES = [
    "type_code", "amount", "log_amount", "hour",
    "amt_to_old_orig", "amt_to_old_dest", "balance_delta_orig", "overdraft",
    "drains_account", "orig_zero_before", "dest_zero_before", "dest_is_merchant",
    "orig_tx_count_24h", "orig_time_since_last",
    "dest_tx_count_24h", "dest_time_since_last",
]


def _past_activity(names: pd.Series, step: pd.Series, window: int = VELOCITY_WINDOW):
    """Per account: #prior txns in the last `window` steps and steps since the last one.

    Strictly causal (only rows earlier in the sorted order count). Fully
    vectorised via a composite (account, step) key -> scales to millions of rows.
    """
    codes, _ = pd.factorize(names)
    order = np.lexsort((np.arange(len(names)), step.to_numpy(), codes))
    c, s = codes[order], step.to_numpy()[order].astype(np.int64)
    big = int(s.max()) + window + 2
    key = c.astype(np.int64) * big + s
    idx = np.arange(len(key))
    first_in_window = np.searchsorted(key, key - window, side="left")
    first_of_acct = np.searchsorted(c, c, side="left")
    first_in_window = np.maximum(first_in_window, first_of_acct)
    count = idx - first_in_window
    prev = np.where(idx > first_of_acct, np.roll(s, 1), -1)
    since = np.where(prev >= 0, s - prev, 10_000)  # 10_000 = "never seen before"
    out_count = np.empty(len(key), dtype=np.int32)
    out_since = np.empty(len(key), dtype=np.int32)
    out_count[order], out_since[order] = count, since
    return out_count, out_since


def build_tabular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    x = pd.DataFrame(index=df.index)
    x["type_code"] = df["type"].map(TYPE_MAP).fillna(-1).astype(int)
    x["amount"] = df["amount"]
    x["log_amount"] = np.log1p(df["amount"])
    x["hour"] = df["step"] % 24
    oo, od = df["oldbalanceOrg"], df["oldbalanceDest"]
    x["amt_to_old_orig"] = df["amount"] / (oo + 1.0)
    x["amt_to_old_dest"] = df["amount"] / (od + 1.0)
    x["balance_delta_orig"] = oo - df["amount"]  # expected, computed pre-commit
    x["overdraft"] = (df["amount"] > oo).astype(int)
    x["drains_account"] = ((df["amount"] >= oo * 0.999) & (oo > 0)).astype(int)
    x["orig_zero_before"] = (oo == 0).astype(int)
    x["dest_zero_before"] = (od == 0).astype(int)
    x["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    x["orig_tx_count_24h"], x["orig_time_since_last"] = _past_activity(df["nameOrig"], df["step"])
    x["dest_tx_count_24h"], x["dest_time_since_last"] = _past_activity(df["nameDest"], df["step"])
    return x[TABULAR_FEATURES]
