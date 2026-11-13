"""Stage 5 - Feature management: a small custom offline feature layer.

The SAME `FeatureStore` code path produces features for training (as_of = train cutoff,
history = train only) and for prediction (as_of = reference date, history = all clean data,
saved as a 'serving' snapshot).  Models and the serving helper read features only through
`get_user_features / get_item_features / pair_features`, so training and prediction can't
drift apart.  Feature definitions live in `definitions.py` and are exported as a catalog.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, path_for
from src.features.definitions import write_catalog

log = get_logger("features")


class FeatureStore:
    def __init__(self, user_features, item_features, user_category_affinity, global_mean, as_of):
        self.user_features = user_features.set_index("user_id")
        self.item_features = item_features.set_index("item_id")
        self.user_category_affinity = user_category_affinity  # index (user_id, category) -> share
        self.global_mean = float(global_mean)
        self.as_of = pd.Timestamp(as_of)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, history: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, as_of, m: float = 5):
        as_of = pd.Timestamp(as_of)
        df = history.merge(items[["item_id", "category"]], on="item_id")
        gmean = df["rating"].mean()

        # ---- user features
        g = df.groupby("user_id")
        uf = g.agg(user_activity_count=("rating", "size"), user_avg_rating=("rating", "mean"),
                   user_rating_std=("rating", "std"), user_positive_rate=("is_positive", "mean"),
                   last_ts=("timestamp", "max")).reset_index()
        cc = pd.concat([df.groupby(["user_id", "category"]).size().rename("n_all"),
                        df[df["is_positive"]].groupby(["user_id", "category"]).size().rename("n_pos")],
                       axis=1).fillna(0)
        cc["score"] = cc["n_pos"] + 1e-3 * cc["n_all"]
        best = cc.groupby(level=0)["score"].idxmax()
        pref = pd.DataFrame({"user_id": best.index, "preferred_category": [t[1] for t in best.values]})
        share = (cc["n_all"] / cc.groupby(level=0)["n_all"].transform("sum")).rename("share")
        pref["preferred_category_share"] = [share.loc[t] for t in best.values]
        uf = uf.merge(pref, on="user_id")
        uf["days_since_last_interaction"] = (as_of - uf["last_ts"]).dt.total_seconds() / 86400
        uf = users[["user_id", "age", "signup_date"]].merge(uf.drop(columns="last_ts"), on="user_id", how="left")
        uf["user_tenure_days"] = (as_of - uf["signup_date"]).dt.total_seconds() / 86400
        uf = uf.drop(columns="signup_date")
        uf["user_activity_count"] = uf["user_activity_count"].fillna(0).astype(int)   # cold-start defaults
        uf["user_avg_rating"] = uf["user_avg_rating"].fillna(gmean)
        uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0)
        uf["user_positive_rate"] = uf["user_positive_rate"].fillna(0.0)
        uf["preferred_category"] = uf["preferred_category"].fillna("Unknown")
        uf["preferred_category_share"] = uf["preferred_category_share"].fillna(0.0)
        uf["days_since_last_interaction"] = uf["days_since_last_interaction"].fillna(uf["user_tenure_days"])

        # ---- item features
        gi = df.groupby("item_id").agg(item_popularity=("user_id", "size"), item_positive_count=("is_positive", "sum"),
                                       mean_r=("rating", "mean"), item_rating_std=("rating", "std"),
                                       last_ts=("timestamp", "max")).reset_index()
        itf = items[["item_id", "category", "price", "release_year"]].merge(gi, on="item_id", how="left")
        itf["item_popularity"] = itf["item_popularity"].fillna(0).astype(int)
        itf["item_positive_count"] = itf["item_positive_count"].fillna(0).astype(int)
        itf["item_avg_rating"] = (itf["item_popularity"] * itf["mean_r"].fillna(0) + m * gmean) / (itf["item_popularity"] + m)
        itf["item_rating_std"] = itf["item_rating_std"].fillna(0.0)
        itf["item_popularity_percentile"] = itf["item_popularity"].rank(pct=True)
        itf["item_age_years"] = as_of.year - itf["release_year"]
        itf["item_days_since_last_interaction"] = ((as_of - itf["last_ts"]).dt.total_seconds() / 86400).fillna(9999.0)
        itf = itf.drop(columns=["mean_r", "last_ts", "release_year"])
        return cls(uf, itf, share, gmean, as_of)

    # ----------------------------------------------------------------- access
    def get_user_features(self, user_ids) -> pd.DataFrame:
        f = self.user_features.reindex(user_ids)
        return f.fillna({"user_activity_count": 0, "user_avg_rating": self.global_mean, "user_rating_std": 0.0,
                         "user_positive_rate": 0.0, "preferred_category": "Unknown", "preferred_category_share": 0.0})

    def get_item_features(self, item_ids) -> pd.DataFrame:
        f = self.item_features.reindex(item_ids)
        return f.fillna({"item_popularity": 0, "item_positive_count": 0, "item_avg_rating": self.global_mean})

    def pair_features(self, user_ids, item_ids) -> pd.DataFrame:
        uf, itf = self.get_user_features(user_ids), self.get_item_features(item_ids)
        cats = itf["category"].to_numpy()
        keys = pd.MultiIndex.from_arrays([np.asarray(user_ids), cats])
        out = pd.DataFrame({"user_id": np.asarray(user_ids), "item_id": np.asarray(item_ids)})
        out["user_item_category_match"] = (uf["preferred_category"].to_numpy() == cats).astype(int)
        out["user_category_affinity"] = self.user_category_affinity.reindex(keys).fillna(0.0).to_numpy()
        out["item_popularity_percentile"] = itf["item_popularity_percentile"].fillna(0.0).to_numpy()
        out["baseline_rating_estimate"] = np.clip(
            uf["user_avg_rating"].to_numpy() + itf["item_avg_rating"].to_numpy() - self.global_mean, 1, 5)
        return out

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.user_features.reset_index().to_parquet(directory / "user_features.parquet", index=False)
        self.item_features.reset_index().to_parquet(directory / "item_features.parquet", index=False)
        self.user_category_affinity.reset_index().to_parquet(directory / "user_category_affinity.parquet", index=False)
        (directory / "meta.json").write_text(json.dumps({"global_mean": self.global_mean, "as_of": str(self.as_of)}))

    @classmethod
    def load(cls, directory: Path) -> "FeatureStore":
        meta = json.loads((directory / "meta.json").read_text())
        aff = pd.read_parquet(directory / "user_category_affinity.parquet").set_index(["user_id", "category"])["share"]
        return cls(pd.read_parquet(directory / "user_features.parquet"), pd.read_parquet(directory / "item_features.parquet"),
                   aff, meta["global_mean"], meta["as_of"])


def run(cfg: dict) -> dict:
    proc, feat, rep = path_for(cfg, "processed"), path_for(cfg, "features"), path_for(cfg, "reports")
    users, items = pd.read_parquet(proc / "clean_users.parquet"), pd.read_parquet(proc / "clean_items.parquet")
    train = pd.read_parquet(proc / "train.parquet")
    inter = pd.read_parquet(proc / "clean_interactions.parquet")
    cutoff = json.loads((proc / "split_meta.json").read_text())["cutoff"]
    m = cfg["features"]["bayes_smoothing_m"]
    train_fs = FeatureStore.build(train, users, items, cutoff, m)            # offline / training
    train_fs.save(feat / "train")
    serve_fs = FeatureStore.build(inter, users, items, cfg["reference_date"], m)  # online / prediction snapshot
    serve_fs.save(feat / "serving")
    write_catalog(rep / "feature_catalog.md", rep / "feature_catalog.csv")
    out = {"user_features": train_fs.user_features.shape, "item_features": train_fs.item_features.shape,
           "serving_users": serve_fs.user_features.shape[0]}
    log.info("Feature tables built: %s", out)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}
