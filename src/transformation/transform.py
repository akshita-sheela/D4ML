"""Stage 4 - Data transformation.

* time-based train/test split (most recent `test_fraction` of interactions -> test)
* derived datasets built from TRAIN only (no leakage):
    user_activity_summary, item_popularity, user-item interaction matrices (sparse .npz)
"""
from __future__ import annotations

import json

import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, path_for

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float):
    cutoff = inter["timestamp"].quantile(1 - test_fraction)
    return inter[inter["timestamp"] < cutoff].copy(), inter[inter["timestamp"] >= cutoff].copy(), cutoff


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    return g.agg(n_interactions=("item_id", "size"), n_positive=("is_positive", "sum"),
                 first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max"),
                 active_days=("timestamp", lambda s: s.dt.date.nunique())).reset_index()


def item_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id").agg(n_interactions=("user_id", "size"), n_positive=("is_positive", "sum"),
                                     mean_rating=("rating", "mean")).reset_index()
    out = items[["item_id", "category"]].merge(g, on="item_id", how="left")
    out[["n_interactions", "n_positive"]] = out[["n_interactions", "n_positive"]].fillna(0).astype(int)
    out["popularity_rank"] = out["n_interactions"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank")


def build_matrices(train: pd.DataFrame, items: pd.DataFrame):
    users_idx = pd.DataFrame({"user_id": sorted(train["user_id"].unique())})
    users_idx["user_idx"] = range(len(users_idx))
    items_idx = pd.DataFrame({"item_id": sorted(items["item_id"].unique())})
    items_idx["item_idx"] = range(len(items_idx))
    t = train.merge(users_idx, on="user_id").merge(items_idx, on="item_id")
    shape = (len(users_idx), len(items_idx))
    ratings = sp.csr_matrix((t["rating"], (t["user_idx"], t["item_idx"])), shape=shape)
    seen = sp.csr_matrix(([1.0] * len(t), (t["user_idx"], t["item_idx"])), shape=shape)
    pos = t[t["is_positive"]]
    positive = sp.csr_matrix(([1.0] * len(pos), (pos["user_idx"], pos["item_idx"])), shape=shape)
    return users_idx, items_idx, ratings, seen, positive


def run(cfg: dict) -> dict:
    proc = path_for(cfg, "processed")
    inter = pd.read_parquet(proc / "clean_interactions.parquet")
    items = pd.read_parquet(proc / "clean_items.parquet")
    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)
    user_activity_summary(train).to_parquet(proc / "user_activity_summary.parquet", index=False)
    item_popularity(train, items).to_parquet(proc / "item_popularity.parquet", index=False)
    u_idx, i_idx, R, S, P = build_matrices(train, items)
    u_idx.to_parquet(proc / "user_index.parquet", index=False)
    i_idx.to_parquet(proc / "item_index.parquet", index=False)
    sp.save_npz(proc / "train_rating_matrix.npz", R)
    sp.save_npz(proc / "train_seen_matrix.npz", S)
    sp.save_npz(proc / "train_positive_matrix.npz", P)
    meta = {"cutoff": str(cutoff), "train_rows": len(train), "test_rows": len(test),
            "train_users": len(u_idx), "catalog_items": len(i_idx),
            "matrix_shape": list(R.shape), "matrix_density": round(R.nnz / (R.shape[0] * R.shape[1]), 5),
            "test_users_unseen_in_train": int((~test["user_id"].isin(train["user_id"])).sum())}
    (proc / "split_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Split: %s", meta)
    return meta
