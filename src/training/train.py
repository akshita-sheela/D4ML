"""Stage 6 - Model training with MLflow tracking (params, artifacts; metrics are added by evaluation)."""
from __future__ import annotations

import json

import joblib
import mlflow
import pandas as pd
import scipy.sparse as sp

from src.common import ROOT, get_logger, path_for
from src.features.store import FeatureStore
from src.training.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender

log = get_logger("training")


def setup_mlflow(cfg):
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(ROOT / uri.replace("sqlite:///", ""))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def load_training_inputs(cfg):
    proc, feat = path_for(cfg, "processed"), path_for(cfg, "features")
    return dict(
        users_idx=pd.read_parquet(proc / "user_index.parquet"),
        items_idx=pd.read_parquet(proc / "item_index.parquet"),
        ratings=sp.load_npz(proc / "train_rating_matrix.npz"),
        seen=sp.load_npz(proc / "train_seen_matrix.npz"),
        positive=sp.load_npz(proc / "train_positive_matrix.npz"),
        fs=FeatureStore.load(feat / "train"),
    )


def run(cfg: dict) -> dict:
    setup_mlflow(cfg)
    models_dir, proc = path_for(cfg, "models"), path_for(cfg, "processed")
    inputs = load_training_inputs(cfg)
    mc, seed = cfg["models"], cfg["seed"]
    models = [PopularityRecommender(), ItemKNNRecommender(**mc["item_knn"]),
              SVDRecommender(**mc["svd_mf"], seed=seed)]
    split = json.loads((proc / "split_meta.json").read_text())
    run_ids = {}
    for model in models:
        with mlflow.start_run(run_name=model.name) as r:
            model.fit(**inputs)
            mlflow.log_params({**model.params(), "seed": seed, "train_rows": split["train_rows"],
                               "split_cutoff": split["cutoff"], "positive_threshold": cfg["evaluation"]["positive_threshold"]})
            mlflow.set_tags({"model": model.name, "stage": "training"})
            f = models_dir / f"{model.name}.joblib"
            joblib.dump(model, f)
            mlflow.log_artifact(str(f), artifact_path="model")
            mlflow.log_artifact(str(path_for(cfg, "reports") / "feature_catalog.md"))
            run_ids[model.name] = r.info.run_id
            log.info("Trained %-10s run_id=%s", model.name, r.info.run_id[:8])
    (models_dir / "run_ids.json").write_text(json.dumps(run_ids, indent=2))
    return run_ids
