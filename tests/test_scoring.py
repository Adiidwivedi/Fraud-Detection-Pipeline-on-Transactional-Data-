"""Training/serving consistency: the scorer's row builder must equal the training feature builder."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fraudpipe.features import build_tabular
from fraudpipe.synthetic import generate

MODEL = Path(__file__).parent.parent / "models" / "model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run train.py first")


def test_serving_features_match_training_features():
    from fraudpipe.scoring import Scorer
    sc = Scorer()
    df = generate(4000, seed=3).sort_values("step", kind="stable").reset_index(drop=True)
    tab = build_tabular(df)
    for i in (5, 100, 2000, 3500):
        r = df.iloc[i]
        t = r.to_dict()
        for c in ("orig_tx_count_24h", "orig_time_since_last", "dest_tx_count_24h", "dest_time_since_last"):
            t[c] = tab.iloc[i][c]
        x = sc._row(t)
        for c in tab.columns:
            if c in x.columns:
                assert np.isclose(x.iloc[0][c], tab.iloc[i][c]), c
