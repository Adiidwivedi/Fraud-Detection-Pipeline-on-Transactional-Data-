"""Scoring + SHAP explanations shared by the API and the dashboard."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from . import graph
from .features import TYPE_MAP, TABULAR_FEATURES

MODELS = Path(__file__).resolve().parent.parent / "models"


class Scorer:
    def __init__(self, model_dir: Path = MODELS):
        art = joblib.load(model_dir / "model.joblib")
        self.model, self.features, self.threshold = art["model"], art["features"], art["threshold"]
        # history snapshot of the transaction graph (accounts with >=2 past edges); others = cold start
        self.nodes = pd.read_parquet(model_dir / "node_features.parquet")
        self.explainer = shap.TreeExplainer(self.model)

    def tier(self, p: float) -> str:
        if p >= self.threshold:
            return "HIGH"
        return "MEDIUM" if p >= self.threshold * 0.4 else "LOW"

    def _row(self, t: dict) -> pd.DataFrame:
        oo, od, amt = t["oldbalanceOrg"], t["oldbalanceDest"], t["amount"]
        f = {
            "type_code": TYPE_MAP.get(t["type"], -1), "amount": amt, "log_amount": np.log1p(amt),
            "hour": t["step"] % 24,
            "amt_to_old_orig": amt / (oo + 1.0), "amt_to_old_dest": amt / (od + 1.0),
            "balance_delta_orig": oo - amt, "overdraft": int(amt > oo),
            "drains_account": int(amt >= oo * 0.999 and oo > 0),
            "orig_zero_before": int(oo == 0), "dest_zero_before": int(od == 0),
            "dest_is_merchant": int(str(t["nameDest"]).startswith("M")),
            # per-account velocity comes from the caller's feature store (defaults = no history)
            "orig_tx_count_24h": t.get("orig_tx_count_24h", 0),
            "orig_time_since_last": t.get("orig_time_since_last", 10_000),
            "dest_tx_count_24h": t.get("dest_tx_count_24h", 0),
            "dest_time_since_last": t.get("dest_time_since_last", 10_000),
        }
        for prefix, name in (("orig", t["nameOrig"]), ("dest", t["nameDest"])):
            for c in graph.NODE_COLS:
                f[f"{prefix}_{c}"] = float(self.nodes[c].get(name, 0.0))
        return pd.DataFrame([f])[self.features]

    def score(self, t: dict, explain: bool = True, top_k: int = 6) -> dict:
        x = self._row(t)
        p = float(self.model.predict_proba(x)[0, 1])
        out = {"fraud_probability": p, "risk_tier": self.tier(p),
               "is_fraud_predicted": p >= self.threshold, "threshold": self.threshold}
        if explain:
            sv = self.explainer.shap_values(x)[0]
            order = np.argsort(-np.abs(sv))[:top_k]
            out["explanation"] = [{"feature": self.features[i], "value": float(x.iloc[0, i]),
                                   "shap": float(sv[i])} for i in order]
            out["base_value"] = float(np.ravel(self.explainer.expected_value)[0])
        return out
