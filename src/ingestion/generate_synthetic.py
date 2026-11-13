"""Synthetic data generator (documented, seeded, deliberately 'dirty').

Why synthetic?  The sandbox the project was built in has no internet access, and a
generator lets us inject *known* data-quality problems so the validation stage has
something real to catch.  The CSV schema below is the contract: a public dataset
(e.g. MovieLens) converted to the same three files can be dropped into data/raw/ and
used with `data.source: csv`.

Schema
  users.csv         user_id, age, gender, country, signup_date
  items.csv         item_id, title, category, release_year, price
  interactions.csv  user_id, item_id, rating (1-5), timestamp

Ground-truth model: users and items have latent vectors (+ a per-category vector, so
categories carry signal).  Which items a user interacts with depends on item popularity
(Pareto) x user-item affinity; the rating is a noisy function of the same affinity.

Injected problems (rates approximate): duplicate rows, duplicate keys, ratings out of
range / missing / non-numeric, bad & future & pre-signup timestamps, orphan user/item ids,
non-integer ids, missing/invalid ages, inconsistent gender/category spelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller",
          "Documentary", "Animation", "Horror", "Fantasy", "Mystery", "Family"]
COUNTRIES = ["IN", "US", "UK", "DE", "FR", "CA", "AU", "SG"]


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_u, n_i, n_int = d["n_users"], d["n_items"], d["n_interactions"]
    dim = 8

    cat_idx = rng.choice(len(GENRES), n_i, p=_zipf(len(GENRES), 0.6))
    items = pd.DataFrame({
        "item_id": np.arange(1, n_i + 1),
        "title": [f"Item {i:04d}" for i in range(1, n_i + 1)],
        "category": np.array(GENRES)[cat_idx],
        "release_year": rng.integers(1995, 2026, n_i),
        "price": np.round(rng.lognormal(3.0, 0.6, n_i), 2),
    })

    start, end = pd.Timestamp(d["start_date"]), pd.Timestamp(d["end_date"])
    span = (end - start).days
    signup = start + pd.to_timedelta(rng.integers(0, int(span * 0.6), n_u), unit="D")
    users = pd.DataFrame({
        "user_id": np.arange(1, n_u + 1),
        "age": rng.integers(15, 70, n_u).astype(float),
        "gender": rng.choice(["male", "female", "other"], n_u, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(COUNTRIES, n_u, p=_zipf(len(COUNTRIES), 0.8)),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    U = rng.normal(size=(n_u, dim))
    cat_vec = rng.normal(size=(len(GENRES), dim))
    V = rng.normal(size=(n_i, dim)) * 0.8 + cat_vec[cat_idx]
    affinity = U @ V.T / np.sqrt(dim)
    item_pop = rng.pareto(1.1, n_i) + 1.0
    user_act = rng.pareto(1.4, n_u) + 1.0
    user_bias = rng.normal(0, 0.35, n_u)
    item_bias = rng.normal(0, 0.35, n_i)

    counts = rng.multinomial(n_int, user_act / user_act.sum())
    counts = np.clip(counts, 6, int(n_i * 0.4))
    rows = []
    for u in range(n_u):
        p = item_pop * np.exp(1.5 * affinity[u])
        chosen = rng.choice(n_i, size=counts[u], replace=False, p=p / p.sum())
        lo = signup[u]
        ts = lo + pd.to_timedelta(rng.integers(0, max((end - lo).days, 1), counts[u]), unit="D") \
                + pd.to_timedelta(rng.integers(0, 86400, counts[u]), unit="s")
        score = 2.4 + 1.1 * affinity[u, chosen] + user_bias[u] + item_bias[chosen] \
            + rng.normal(0, 0.6, counts[u])
        rows.append(pd.DataFrame({"user_id": u + 1, "item_id": chosen + 1,
                                  "rating": np.clip(np.rint(score), 1, 5), "timestamp": ts}))
    inter = pd.concat(rows, ignore_index=True)
    inter["timestamp"] = inter["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    users, items, inter = _dirty(rng, users, items, inter, n_u, n_i)
    return {"users": users, "items": items, "interactions": inter}


def _zipf(n: int, a: float) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    return w / w.sum()


def _pick(rng, df: pd.DataFrame, frac: float) -> np.ndarray:
    return rng.choice(df.index.to_numpy(), size=max(1, int(len(df) * frac)), replace=False)


def _dirty(rng, users, items, inter, n_u, n_i):
    users, items, inter = (x.astype(object).copy() for x in (users, items, inter))

    users.loc[_pick(rng, users, 0.010), "age"] = np.nan
    users.loc[_pick(rng, users, 0.003), "age"] = 150
    users.loc[_pick(rng, users, 0.010), "country"] = np.nan
    g = users["gender"].to_numpy(copy=True)
    for i in _pick(rng, users, 0.15):
        g[i] = {"male": rng.choice(["M", "Male", " male"]), "female": rng.choice(["F", "FEMALE", "female "]),
                "other": "Other"}[g[i]]
    users["gender"] = g
    users = pd.concat([users, users.loc[_pick(rng, users, 0.005)]], ignore_index=True)

    items.loc[_pick(rng, items, 0.010), "category"] = np.nan
    lc = _pick(rng, items, 0.05)
    items.loc[lc, "category"] = items.loc[lc, "category"].map(
        lambda c: c.lower() if isinstance(c, str) else c)
    items.loc[_pick(rng, items, 0.004), "release_year"] = 2099
    items = pd.concat([items, items.loc[_pick(rng, items, 0.005)]], ignore_index=True)

    inter.loc[_pick(rng, inter, 0.005), "rating"] = rng.choice([0, 6, -1, 11])
    inter.loc[_pick(rng, inter, 0.003), "rating"] = np.nan
    inter.loc[_pick(rng, inter, 0.002), "rating"] = "five"
    inter.loc[_pick(rng, inter, 0.003), "timestamp"] = "not-a-date"
    inter.loc[_pick(rng, inter, 0.003), "timestamp"] = "2031-05-17 10:00:00"
    inter.loc[_pick(rng, inter, 0.002), "timestamp"] = "2019-01-01 00:00:00"
    inter.loc[_pick(rng, inter, 0.004), "user_id"] = n_u + int(rng.integers(100, 900))
    inter.loc[_pick(rng, inter, 0.004), "item_id"] = n_i + int(rng.integers(100, 900))
    inter.loc[_pick(rng, inter, 0.002), "user_id"] = "abc"
    inter = pd.concat([inter, inter.loc[_pick(rng, inter, 0.010)]], ignore_index=True)
    re = inter.loc[_pick(rng, inter, 0.008)].copy()
    re["rating"] = rng.integers(1, 6, len(re))
    re["timestamp"] = "2025-12-30 12:00:00"
    inter = pd.concat([inter, re], ignore_index=True)
    inter = inter.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return users.reset_index(drop=True), items.reset_index(drop=True), inter
