import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.validation.validate import validate_interactions, validate_items, validate_users  # noqa: E402
from src.transformation.transform import time_split  # noqa: E402

CFG = load_config()


def test_metrics_known_values():
    recs = np.array([[1, 2, 3]])
    rel = [{1, 3, 9}]
    assert abs(M.precision_at_k(recs, rel, 3) - 2 / 3) < 1e-9
    assert abs(M.recall_at_k(recs, rel, 3) - 2 / 3) < 1e-9
    ideal = 1 + 1 / np.log2(3) + 1 / 2
    assert abs(M.ndcg_at_k(recs, rel, 3) - (1 + 0.5) / ideal) < 1e-9
    assert M.coverage(np.array([[0, 1], [1, 2]]), 6) == 0.5
    assert M.diversity(np.array([[0, 1]]), np.array([0, 1])) == 1.0


def _tables():
    users = pd.DataFrame({"user_id": ["1", "2", "2", "3"], "age": ["30", "150", "20", None],
                          "gender": ["m"] * 4, "country": ["IN"] * 4, "signup_date": ["2024-01-01"] * 4})
    items = pd.DataFrame({"item_id": ["1", "2"], "title": "t", "category": ["A", None],
                          "release_year": ["2000", "2099"], "price": ["5", "6"]})
    return users, items


def test_user_and_item_validation():
    users, items = _tables()
    ok, err, _ = validate_users(users, CFG)
    assert set(ok["user_id"]) == {1, 3}                      # dup id and age=150 rejected
    ok_i, err_i, _ = validate_items(items, CFG)
    assert list(ok_i["item_id"]) == [1] and "invalid_release_year" in err_i["error_reason"].iloc[0]


def test_interaction_validation_rules():
    users, items = _tables()
    vu, _, _ = validate_users(users, CFG)
    vi, _, _ = validate_items(items, CFG)
    df = pd.DataFrame({
        "user_id": ["1", "1", "1", "1", "1", "99", "1"],
        "item_id": ["1", "1", "1", "1", "1", "1", "55"],
        "rating": ["5", "9", "x", "4", "4", "3", "3"],
        "timestamp": ["2024-06-01 10:00:00", "2024-06-01 10:00:00", "2024-06-01 10:00:00",
                      "bad", "2031-01-01 00:00:00", "2024-06-01 10:00:00", "2024-06-01 10:00:00"]})
    ok, err, r = validate_interactions(df, CFG, vu, vi)
    assert len(ok) == 1
    for rule in ["invalid_rating_range", "invalid_dtype_rating", "invalid_timestamp_format",
                 "future_timestamp", "orphan_user_id", "orphan_item_id"]:
        assert r.counts[rule] == 1, rule


def test_time_split_has_no_overlap():
    ts = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"user_id": 1, "item_id": range(100), "timestamp": ts})
    train, test, cutoff = time_split(df, 0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(test) == 20
