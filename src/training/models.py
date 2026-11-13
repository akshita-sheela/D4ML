"""Recommenders.  All share: fit(...), recommend(user_idx, k) -> (n, k) item indices,
predict_rating(user_ids, item_ids) -> np.ndarray | None.

1. PopularityRecommender  - baseline: ranks by item_positive_count feature
2. ItemKNNRecommender     - item-item collaborative filtering (cosine on positive matrix)
3. SVDRecommender         - matrix factorisation (TruncatedSVD) for ranking + biased-baseline
                            + latent-residual model for rating prediction (RMSE/MAE)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def _top_k(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    scores = scores.copy()
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf                       # never re-recommend what was seen in train
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


class BaseRecommender:
    name = "base"

    def score_matrix(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_idx: np.ndarray, k: int, batch: int = 500) -> np.ndarray:
        out = []
        for s in range(0, len(user_idx), batch):
            u = user_idx[s:s + batch]
            out.append(_top_k(self.score_matrix(u), self.seen[u], k))
        return np.vstack(out)

    def predict_rating(self, user_ids, item_ids):
        return None


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def fit(self, seen, items_idx, fs, **_):
        self.seen, self.fs = seen, fs
        self.pop = fs.get_item_features(items_idx["item_id"].to_numpy())["item_positive_count"].to_numpy(float)
        return self

    def score_matrix(self, user_idx):
        return np.tile(self.pop, (len(user_idx), 1))

    def predict_rating(self, user_ids, item_ids):
        return self.fs.get_item_features(item_ids)["item_avg_rating"].to_numpy()

    def params(self):
        return {"ranking": "item_positive_count", "rating": "smoothed_item_avg_rating"}


class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"

    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

    def fit(self, seen, positive, **_):
        self.seen, self.P = seen, positive.tocsr()
        norms = np.sqrt(np.asarray(self.P.multiply(self.P).sum(axis=0)).ravel()) + 1e-9
        sim = (self.P.T @ self.P).toarray() / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        k = min(self.n_neighbors, sim.shape[0] - 1)
        drop = np.argpartition(-sim, k, axis=1)[:, k:]            # zero all but top-k neighbours
        np.put_along_axis(sim, drop, 0.0, axis=1)
        self.sim = sim
        return self

    def score_matrix(self, user_idx):
        return np.asarray(self.P[user_idx] @ self.sim)

    def params(self):
        return {"n_neighbors": self.n_neighbors, "similarity": "cosine"}


class SVDRecommender(BaseRecommender):
    name = "svd_mf"

    def __init__(self, n_factors=32, n_factors_rating=16, seed=42):
        self.n_factors, self.n_factors_rating, self.seed = n_factors, n_factors_rating, seed

    def fit(self, seen, positive, ratings, users_idx, items_idx, fs, **_):
        self.seen, self.fs = seen, fs
        self.users_idx, self.items_idx = users_idx, items_idx
        self.u_pos = dict(zip(users_idx["user_id"], users_idx["user_idx"]))
        self.i_pos = dict(zip(items_idx["item_id"], items_idx["item_idx"]))
        # ranking factorisation on implicit-positive matrix
        self.svd = TruncatedSVD(self.n_factors, random_state=self.seed).fit(positive)
        self.Uf = self.svd.transform(positive)
        # rating model: baseline (from feature store) + low-rank factorisation of residuals
        r = ratings.tocoo()
        uid, iid = users_idx["user_id"].to_numpy()[r.row], items_idx["item_id"].to_numpy()[r.col]
        base = fs.pair_features(uid, iid)["baseline_rating_estimate"].to_numpy()
        resid = sp.csr_matrix((r.data - base, (r.row, r.col)), shape=ratings.shape)
        self.svd_r = TruncatedSVD(self.n_factors_rating, random_state=self.seed).fit(resid)
        self.Ur = self.svd_r.transform(resid)
        return self

    def score_matrix(self, user_idx):
        return self.Uf[user_idx] @ self.svd.components_

    def predict_rating(self, user_ids, item_ids):
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        base = self.fs.pair_features(user_ids, item_ids)["baseline_rating_estimate"].to_numpy()
        ui = np.array([self.u_pos.get(u, -1) for u in user_ids])
        ii = np.array([self.i_pos.get(i, -1) for i in item_ids])
        known = (ui >= 0) & (ii >= 0)
        adj = np.zeros(len(base))
        adj[known] = np.einsum("ij,ji->i", self.Ur[ui[known]], self.svd_r.components_[:, ii[known]])
        return np.clip(base + adj, 1, 5)

    def params(self):
        return {"n_factors": self.n_factors, "n_factors_rating": self.n_factors_rating, "algorithm": "TruncatedSVD"}
