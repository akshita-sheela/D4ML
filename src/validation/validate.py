"""Stage 2 - Data validation.

Rule-based checks (missing values, duplicates, wrong dtypes, invalid ratings, invalid
timestamps, referential integrity).  Rows failing any *error* rule are moved to
data/errors/<table>_errors.csv with an `error_reason` column; passing rows are written
with proper dtypes to data/processed/validated_<table>.parquet.  *Warnings* (e.g. missing
optional fields that preparation will impute) do not remove rows but are counted.
A JSON + Markdown report is written to reports/.
"""
from __future__ import annotations

import json

import pandas as pd

from src.common import get_logger, path_for

log = get_logger("validation")


class Rules:
    """Collects rule violations per row."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.reasons = pd.Series("", index=df.index)
        self.counts: dict[str, int] = {}
        self.warnings: dict[str, int] = {}

    def error(self, mask: pd.Series, name: str) -> None:
        mask = mask.fillna(False).astype(bool)
        self.counts[name] = int(mask.sum())
        self.reasons.loc[mask] = self.reasons.loc[mask].map(lambda r: name if not r else f"{r};{name}")

    def warn(self, mask: pd.Series, name: str) -> None:
        self.warnings[name] = int(mask.fillna(False).sum())

    def split(self):
        bad = self.reasons != ""
        errors = self.df[bad].copy()
        errors["error_reason"] = self.reasons[bad]
        return self.df[~bad].copy(), errors


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def validate_users(df, cfg):
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    r = Rules(df)
    r.error(_blank(df["user_id"]), "missing_user_id")
    r.error(~_blank(df["user_id"]) & _num(df["user_id"]).isna(), "invalid_dtype_user_id")
    r.error(df["user_id"].duplicated(keep="first") & ~_blank(df["user_id"]), "duplicate_user_id")
    age = _num(df["age"])
    r.error(df["age"].notna() & age.isna(), "invalid_dtype_age")
    r.error(age.notna() & ((age < v["age_min"]) | (age > v["age_max"])), "invalid_age_range")
    signup = pd.to_datetime(df["signup_date"], errors="coerce")
    r.error(df["signup_date"].notna() & signup.isna(), "invalid_signup_date")
    r.error(signup > ref, "future_signup_date")
    r.warn(df["age"].isna(), "missing_age (imputed in preparation)")
    r.warn(df["gender"].isna(), "missing_gender")
    r.warn(df["country"].isna(), "missing_country (imputed in preparation)")
    ok, err = r.split()
    ok = ok.assign(user_id=_num(ok["user_id"]).astype("int64"), age=_num(ok["age"]),
                   signup_date=pd.to_datetime(ok["signup_date"]))
    return ok, err, r


def validate_items(df, cfg):
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    r = Rules(df)
    r.error(_blank(df["item_id"]), "missing_item_id")
    r.error(~_blank(df["item_id"]) & _num(df["item_id"]).isna(), "invalid_dtype_item_id")
    r.error(df["item_id"].duplicated(keep="first") & ~_blank(df["item_id"]), "duplicate_item_id")
    yr = _num(df["release_year"])
    r.error(df["release_year"].notna() & yr.isna(), "invalid_dtype_release_year")
    r.error(yr.notna() & ((yr < v["release_year_min"]) | (yr > ref.year)), "invalid_release_year")
    price = _num(df["price"])
    r.error(df["price"].notna() & price.isna(), "invalid_dtype_price")
    r.error(price < 0, "negative_price")
    r.warn(df["category"].isna(), "missing_category (imputed 'Unknown')")
    ok, err = r.split()
    ok = ok.assign(item_id=_num(ok["item_id"]).astype("int64"), release_year=_num(ok["release_year"]),
                   price=_num(ok["price"]))
    return ok, err, r


def validate_interactions(df, cfg, valid_users: pd.DataFrame, valid_items: pd.DataFrame):
    v, ref = cfg["validation"], pd.Timestamp(cfg["reference_date"])
    r = Rules(df)
    for c in ["user_id", "item_id", "rating", "timestamp"]:
        r.error(_blank(df[c]), f"missing_{c}")
    uid, iid, rating = _num(df["user_id"]), _num(df["item_id"]), _num(df["rating"])
    r.error(~_blank(df["user_id"]) & uid.isna(), "invalid_dtype_user_id")
    r.error(~_blank(df["item_id"]) & iid.isna(), "invalid_dtype_item_id")
    r.error(~_blank(df["rating"]) & rating.isna(), "invalid_dtype_rating")
    r.error(rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])), "invalid_rating_range")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    r.error(~_blank(df["timestamp"]) & ts.isna(), "invalid_timestamp_format")
    r.error(ts > ref, "future_timestamp")
    signup = uid.map(valid_users.set_index("user_id")["signup_date"])
    r.error(ts < signup, "timestamp_before_signup")
    r.error(uid.notna() & ~uid.isin(valid_users["user_id"]), "orphan_user_id")
    r.error(iid.notna() & ~iid.isin(valid_items["item_id"]), "orphan_item_id")
    r.error(df.duplicated(keep="first"), "exact_duplicate_row")
    ok, err = r.split()
    ok = ok.assign(user_id=_num(ok["user_id"]).astype("int64"), item_id=_num(ok["item_id"]).astype("int64"),
                   rating=_num(ok["rating"]), timestamp=pd.to_datetime(ok["timestamp"]))
    return ok, err, r


def run(cfg: dict) -> dict:
    proc, errdir, rep = path_for(cfg, "processed"), path_for(cfg, "errors"), path_for(cfg, "reports")
    raw = {t: pd.read_parquet(proc / f"ingested_{t}.parquet") for t in ["users", "items", "interactions"]}
    users, u_err, ru = validate_users(raw["users"], cfg)
    items, i_err, ri = validate_items(raw["items"], cfg)
    inter, x_err, rx = validate_interactions(raw["interactions"], cfg, users, items)

    summary = {}
    for name, ok, err, r, total in [("users", users, u_err, ru, len(raw["users"])),
                                    ("items", items, i_err, ri, len(raw["items"])),
                                    ("interactions", inter, x_err, rx, len(raw["interactions"]))]:
        ok.to_parquet(proc / f"validated_{name}.parquet", index=False)
        err.to_csv(errdir / f"{name}_errors.csv", index=False)
        summary[name] = {"total_rows": total, "valid_rows": len(ok), "invalid_rows": len(err),
                         "pct_invalid": round(100 * len(err) / total, 3),
                         "rule_violations": {k: v for k, v in r.counts.items() if v},
                         "warnings": {k: v for k, v in r.warnings.items() if v}}
        log.info("%-12s total=%d valid=%d invalid=%d", name, total, len(ok), len(err))
    (rep / "data_validation_report.json").write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, rep / "data_validation_report.md", cfg)
    return summary


def _write_markdown(summary: dict, out, cfg) -> None:
    L = ["# Data Validation Report", "",
         f"Reference date: `{cfg['reference_date']}`. Invalid rows are written to `data/errors/<table>_errors.csv` "
         "with an `error_reason` column (a row can violate several rules).", ""]
    L += ["| Table | Total | Valid | Invalid | % invalid |", "|---|---:|---:|---:|---:|"]
    for t, s in summary.items():
        L.append(f"| {t} | {s['total_rows']} | {s['valid_rows']} | {s['invalid_rows']} | {s['pct_invalid']} |")
    for t, s in summary.items():
        L += ["", f"## {t}", "", "| Rule | Rows violating |", "|---|---:|"]
        L += [f"| {k} | {v} |" for k, v in sorted(s["rule_violations"].items(), key=lambda kv: -kv[1])]
        if s["warnings"]:
            L += ["", "Warnings (rows kept):", ""] + [f"- {k}: {v}" for k, v in s["warnings"].items()]
    out.write_text("\n".join(L) + "\n")
