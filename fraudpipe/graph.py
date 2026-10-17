"""Transaction-network layer: sender->receiver graph features.

Degree centrality and clustering coefficient per account, computed with sparse
linear algebra (triangles = diag(A^3)/2) so it scales to millions of edges
where networkx would not. Fraud rings (many victims -> few mules, mules
cross-linked) show up as hub-like degree and unusual clustering that
row-level tabular features cannot see.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

NODE_COLS = ["out_deg", "in_deg", "uniq_receivers", "uniq_senders",
             "deg_centrality", "clustering"]
GRAPH_FEATURES = [f"{p}_{c}" for p in ("orig", "dest") for c in NODE_COLS]


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account graph features from the edges in `df` (nameOrig -> nameDest)."""
    if len(df) == 0:
        return pd.DataFrame(columns=NODE_COLS, dtype=float).rename_axis("account")
    names = pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
    codes, uniques = pd.factorize(names)
    n = len(uniques)
    src, dst = codes[: len(df)], codes[len(df):]

    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)

    # unique directed edges -> unique receivers / senders per node
    e = np.unique(np.stack([src, dst], 1), axis=0)
    uniq_rec = np.bincount(e[:, 0], minlength=n)
    uniq_snd = np.bincount(e[:, 1], minlength=n)

    # undirected simple graph for clustering coefficient
    und = e[e[:, 0] != e[:, 1]]
    a = sp.coo_matrix((np.ones(len(und), dtype=np.float32), (und[:, 0], und[:, 1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    k = np.asarray(a.sum(1)).ravel()
    tri = np.asarray((a @ a).multiply(a).sum(1)).ravel() / 2.0
    clustering = np.divide(2.0 * tri, k * (k - 1), out=np.zeros(n), where=k > 1)

    return pd.DataFrame({
        "out_deg": out_deg, "in_deg": in_deg,
        "uniq_receivers": uniq_rec, "uniq_senders": uniq_snd,
        "deg_centrality": k / max(n - 1, 1), "clustering": clustering,
    }, index=pd.Index(uniques, name="account"))


def attach(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Join node features onto each transaction for both endpoints (unseen -> 0)."""
    out = {}
    for prefix, col in (("orig", "nameOrig"), ("dest", "nameDest")):
        joined = nodes.reindex(df[col].to_numpy())
        for c in NODE_COLS:
            out[f"{prefix}_{c}"] = joined[c].fillna(0.0).to_numpy()
    return pd.DataFrame(out, index=df.index)[GRAPH_FEATURES]


def causal_features(df: pd.DataFrame, n_blocks: int = 20):
    """Graph features with NO look-ahead.

    `df` must be sorted by time. Rows are cut into `n_blocks` consecutive time
    blocks; block b is featurised from the graph of all edges in blocks < b only
    (block 0 -> cold start, all zeros). This mirrors production: the graph is a
    batch-refreshed snapshot of *history*, never of the future. No labels are used.

    Returns (features, block_bounds).
    """
    bounds = np.linspace(0, len(df), n_blocks + 1).astype(int)
    parts = []
    for b in range(n_blocks):
        blk = df.iloc[bounds[b]:bounds[b + 1]]
        nodes = build_node_table(df.iloc[:bounds[b]])
        parts.append(attach(blk, nodes))
    return pd.concat(parts), bounds
