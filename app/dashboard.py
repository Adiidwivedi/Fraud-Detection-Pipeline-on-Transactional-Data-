"""Streamlit fraud console.   streamlit run app/dashboard.py
Scores via the FastAPI service if API_URL is reachable, else with the local model."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Risk Console", layout="wide")
st.title("Transaction Fraud Risk Console")
tab_score, tab_batch, tab_model = st.tabs(["Score a transaction", "Held-out PaySim sample", "Model report"])


@st.cache_resource
def local_scorer():
    from fraudpipe.scoring import Scorer
    return Scorer()


@st.cache_data
def load_sample():
    p = ROOT / "data" / "sample_transactions.csv"
    return pd.read_csv(p) if p.exists() else None


def score(tx, explain=True):
    try:
        r = requests.post(f"{API_URL}/predict", params={"explain": explain}, json=tx, timeout=5)
        r.raise_for_status()
        return r.json(), "FastAPI"
    except Exception:
        return local_scorer().score(tx, explain=explain), "local model"


FIELDS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest",
          "orig_tx_count_24h", "orig_time_since_last", "dest_tx_count_24h", "dest_time_since_last"]
DEFAULT = dict(step=500, type="TRANSFER", amount=250_000.0, nameOrig="C123", oldbalanceOrg=250_000.0,
               nameDest="C90000001", oldbalanceDest=0.0, orig_tx_count_24h=0, orig_time_since_last=10_000,
               dest_tx_count_24h=0, dest_time_since_last=10_000)
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)

sample = load_sample()


def show(res, src):
    colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{res['fraud_probability']:.2%}")
    c2.markdown(f"### Risk tier: :{colors[res['risk_tier']]}[{res['risk_tier']}]")
    c3.metric("Decision threshold", f"{res['threshold']:.3f}")
    st.caption(f"scored via {src}")
    st.subheader("Why? Top SHAP contributions (log-odds; positive = pushes towards fraud)")
    ex = pd.DataFrame(res["explanation"])
    st.bar_chart(ex.set_index("feature")["shap"])
    st.dataframe(ex, use_container_width=True, hide_index=True)


with tab_score:
    if sample is not None:
        c = st.columns([3, 1])
        i = c[0].number_input("Load a real held-out transaction (row of the sample)", 0, len(sample) - 1, 0)
        if c[1].button("Load row"):
            row = sample.iloc[int(i)]
            for k in FIELDS:
                v = row[k]
                st.session_state[k] = v.item() if hasattr(v, "item") else v
            st.session_state["_label"] = int(row["isFraud"])
    left, right = st.columns(2)
    tx = dict(
        step=left.number_input("Step (hour)", 0, 10_000, key="step"),
        type=left.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], key="type"),
        amount=left.number_input("Amount", 0.0, 1e10, key="amount"),
        nameOrig=left.text_input("Sender", key="nameOrig"),
        oldbalanceOrg=left.number_input("Sender balance before", 0.0, 1e10, key="oldbalanceOrg"),
        nameDest=right.text_input("Receiver", key="nameDest"),
        oldbalanceDest=right.number_input("Receiver balance before", 0.0, 1e10, key="oldbalanceDest"),
        orig_tx_count_24h=right.number_input("Sender txns in last 24h", 0, 1000, key="orig_tx_count_24h"),
        orig_time_since_last=right.number_input("Sender hours since last txn", 0, 10_000, key="orig_time_since_last"),
        dest_tx_count_24h=right.number_input("Receiver txns in last 24h", 0, 1000, key="dest_tx_count_24h"),
        dest_time_since_last=right.number_input("Receiver hours since last txn", 0, 10_000, key="dest_time_since_last"),
    )
    if st.button("Score transaction", type="primary"):
        res, src = score(tx)
        if "_label" in st.session_state:
            st.write("Ground truth of the loaded row:", "**FRAUD**" if st.session_state["_label"] else "legit")
        show(res, src)

with tab_batch:
    if sample is None:
        st.info("Run train.py to generate data/sample_transactions.csv")
    else:
        if st.button("Score whole sample (no SHAP)"):
            sc = local_scorer()
            probs = [sc.score(r, explain=False)["fraud_probability"]
                     for r in sample[FIELDS].to_dict("records")]
            d = sample.assign(fraud_probability=probs)
            d["flagged"] = d["fraud_probability"] >= sc.threshold
            st.write(pd.crosstab(d["isFraud"], d["flagged"], rownames=["actual fraud"], colnames=["flagged"]))
            st.dataframe(d.sort_values("fraud_probability", ascending=False).head(200), use_container_width=True)

with tab_model:
    mp = ROOT / "reports" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        rows = {k: {kk: v for kk, v in d.items() if kk in
                    ("pr_auc", "precision", "recall", "f1", "alert_rate", "tp", "fp", "fn")}
                for k, d in m.items() if k not in ("meta", "WITH_LEAKY_COLUMNS_do_not_use")}
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
        st.caption(str(m["meta"]))
    for img in ("pr_curve_and_ablation.png", "shap_summary.png"):
        if (ROOT / "reports" / img).exists():
            st.image(str(ROOT / "reports" / img))
