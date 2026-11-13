"""Stage 7 - Model evaluation on the time-based test set.

Protocol: for every train-known user with >=1 positive (rating >= threshold) interaction in
the test period, produce top-K items excluding items seen in train, then compare against the
user's test positives.  Rating metrics (RMSE/MAE) use all test interactions; unknown
users/items fall back to the smoothed global/item mean from the feature store.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.common import get_logger, path_for
from src.evaluation import metrics as M
from src.training.train import load_training_inputs, setup_mlflow

log = get_logger("evaluation")
MODELS = ["popularity", "item_knn", "svd_mf"]


def run(cfg: dict) -> pd.DataFrame:
    setup_mlflow(cfg)
    k = cfg["evaluation"]["k"]
    proc, models_dir, rep = path_for(cfg, "processed"), path_for(cfg, "models"), path_for(cfg, "reports")
    inputs = load_training_inputs(cfg)
    u_idx, i_idx, fs = inputs["users_idx"], inputs["items_idx"], inputs["fs"]
    test = pd.read_parquet(proc / "test.parquet")
    run_ids = json.loads((models_dir / "run_ids.json").read_text())

    pos = test[test["is_positive"]].merge(u_idx, on="user_id").merge(i_idx, on="item_id")
    rel_map = pos.groupby("user_idx")["item_idx"].apply(set)
    users = rel_map.index.to_numpy()
    relevant = [rel_map[u] for u in users]
    category = fs.get_item_features(i_idx["item_id"].to_numpy())["category"].to_numpy()
    log.info("Evaluating on %d users, K=%d, catalog=%d items", len(users), k, len(i_idx))

    rows = []
    for name in MODELS:
        model = joblib.load(models_dir / f"{name}.joblib")
        recs = model.recommend(users, k)
        res = {"model": name,
               f"precision@{k}": M.precision_at_k(recs, relevant, k),
               f"recall@{k}": M.recall_at_k(recs, relevant, k),
               f"ndcg@{k}": M.ndcg_at_k(recs, relevant, k),
               "coverage": M.coverage(recs, len(i_idx)),
               "diversity": M.diversity(recs, pd.factorize(category)[0])}
        pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
        res["rmse"] = M.rmse(test["rating"], pred) if pred is not None else float("nan")
        res["mae"] = M.mae(test["rating"], pred) if pred is not None else float("nan")
        rows.append(res)
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({kk.replace("@", "_at_"): v for kk, v in res.items()
                                if kk != "model" and v == v})
            mlflow.log_param("eval_users", len(users))
        log.info("%s", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res.items()})
    df = pd.DataFrame(rows)
    df.to_csv(rep / "model_comparison.csv", index=False)
    df.round(4).to_markdown(rep / "model_comparison.md", index=False)
    with mlflow.start_run(run_id=run_ids["svd_mf"]):
        mlflow.log_artifact(str(rep / "model_comparison.csv"))
    _plot(df, k, rep / "model_comparison.png")
    return df


def _plot(df, k, out):
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity"]
    ax = df.set_index("model")[cols].plot.bar(figsize=(9, 4), rot=0, width=0.8)
    ax.set_title(f"Model comparison (K={k}, time-based test split)")
    ax.set_ylabel("score")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
