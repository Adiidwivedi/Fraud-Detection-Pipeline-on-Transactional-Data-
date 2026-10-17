import numpy as np
import pandas as pd

from fraudpipe import graph
from fraudpipe.features import build_tabular, TABULAR_FEATURES, LEAKY_COLS, _past_activity
from fraudpipe.synthetic import generate


def test_no_leaky_columns_in_features():
    x = build_tabular(generate(5000))
    assert not set(LEAKY_COLS) & set(x.columns)
    assert list(x.columns) == TABULAR_FEATURES


def test_velocity_is_strictly_causal():
    names = pd.Series(["a", "a", "b", "a", "a"])
    step = pd.Series([1, 2, 2, 10, 40])
    cnt, since = _past_activity(names, step, window=24)
    assert cnt.tolist() == [0, 1, 0, 2, 0]          # only past txns inside the window
    assert since.tolist() == [10_000, 1, 10_000, 8, 30]


def test_graph_clustering_triangle():
    df = pd.DataFrame({"nameOrig": ["a", "b", "c", "d"], "nameDest": ["b", "c", "a", "a"]})
    nodes = graph.build_node_table(df)
    assert abs(nodes.loc["b", "clustering"] - 1.0) < 1e-6
    assert abs(nodes.loc["a", "clustering"] - 1 / 3) < 1e-6
    assert nodes.loc["a", "in_deg"] == 2


def test_attach_unseen_nodes_zero():
    nodes = graph.build_node_table(pd.DataFrame({"nameOrig": ["a"], "nameDest": ["b"]}))
    g = graph.attach(pd.DataFrame({"nameOrig": ["zzz"], "nameDest": ["b"]}), nodes)
    assert g["orig_out_deg"].iloc[0] == 0 and g["dest_in_deg"].iloc[0] == 1


def test_synthetic_fraud_rate():
    df = generate(20000)
    assert 0.004 < df.isFraud.mean() < 0.02
    assert set(df[df.isFraud == 1].type) <= {"TRANSFER", "CASH_OUT"}
