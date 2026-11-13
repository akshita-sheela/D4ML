"""Stage 3 - Data preparation (cleaning / standardisation).

* users: standardise gender, impute missing age (median) and country ('Unknown')
* items: standardise category casing, impute missing category ('Unknown')
* interactions: drop rows whose user/item did not survive validation, resolve repeated
  (user, item) pairs by keeping the most recent interaction, add `is_positive`
"""
from __future__ import annotations

import pandas as pd

from src.common import get_logger, path_for

log = get_logger("preparation")
GENDER_MAP = {"m": "male", "male": "male", "f": "female", "female": "female", "other": "other"}


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].astype("string").str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    df["age"] = df["age"].fillna(df["age"].median()).round().astype("int64")
    df["country"] = df["country"].astype("string").str.strip().str.upper().fillna("Unknown")
    return df


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["category"].astype("string").str.strip().str.title().fillna("Unknown")
    df["release_year"] = df["release_year"].astype("int64")
    return df


def clean_interactions(df, users, items, positive_threshold: int) -> pd.DataFrame:
    df = df[df["user_id"].isin(users["user_id"]) & df["item_id"].isin(items["item_id"])].copy()
    before = len(df)
    df = (df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
            .sort_values("timestamp").reset_index(drop=True))
    log.info("Resolved %d repeated (user,item) pairs (kept latest)", before - len(df))
    df["rating"] = df["rating"].astype(float)
    df["is_positive"] = df["rating"] >= positive_threshold
    return df


def run(cfg: dict) -> dict:
    proc = path_for(cfg, "processed")
    users = clean_users(pd.read_parquet(proc / "validated_users.parquet"))
    items = clean_items(pd.read_parquet(proc / "validated_items.parquet"))
    inter = clean_interactions(pd.read_parquet(proc / "validated_interactions.parquet"), users, items,
                               cfg["evaluation"]["positive_threshold"])
    users.to_parquet(proc / "clean_users.parquet", index=False)
    items.to_parquet(proc / "clean_items.parquet", index=False)
    inter.to_parquet(proc / "clean_interactions.parquet", index=False)
    out = {"users": len(users), "items": len(items), "interactions": len(inter),
           "positive_rate": round(float(inter["is_positive"].mean()), 4)}
    log.info("Clean data: %s", out)
    return out
