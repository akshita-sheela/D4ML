"""Ranking and rating metrics (binary relevance)."""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: np.ndarray, relevant: list[set], k: int) -> float:
    return float(np.mean([len(set(r[:k]) & rel) / k for r, rel in zip(recs, relevant)]))


def recall_at_k(recs, relevant, k) -> float:
    return float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in zip(recs, relevant)]))


def ndcg_at_k(recs, relevant, k) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for r, rel in zip(recs, relevant):
        gains = np.array([1.0 if i in rel else 0.0 for i in r[:k]])
        ideal = disc[: min(len(rel), k)].sum()
        out.append((gains * disc).sum() / ideal)
    return float(np.mean(out))


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    return float(len(np.unique(recs)) / n_items)


def diversity(recs: np.ndarray, item_category: np.ndarray) -> float:
    """Mean intra-list diversity = share of item pairs in a list that have different categories."""
    k = recs.shape[1]
    out = []
    for r in recs:
        c = item_category[r]
        same = (c[:, None] == c[None, :]).sum() - k          # remove diagonal
        out.append(1.0 - same / (k * (k - 1)))
    return float(np.mean(out))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
