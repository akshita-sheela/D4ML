"""Prefect flow: Data Ingestion -> Validation -> Preparation -> Transformation ->
Feature Management -> Training -> Evaluation.

    python dags/reco_flow.py            # run once
    python dags/reco_flow.py --serve    # run on a schedule (daily 02:00) with `flow.serve`
    prefect server start                # (separate terminal) UI at http://127.0.0.1:4200

Dependencies are explicit: each task receives the previous task's result, so Prefect only
starts a stage after its upstream stage succeeded (a failure skips everything downstream).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402

from src.common import load_config, path_for, reset_run_log, seed_everything  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import store as feature_store  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402

RETRY = dict(retries=1, retry_delay_seconds=5)


def _timed(name, fn, cfg):
    t0 = time.time()
    get_run_logger().info("running %s", name)
    fn(cfg)
    return {"stage": name, "status": "SUCCESS", "start": t0, "seconds": round(time.time() - t0, 2)}


@task(name="1_data_ingestion", **RETRY)
def t_ingest(cfg): return _timed("ingestion", ingest.run, cfg)

@task(name="2_data_validation", **RETRY)
def t_validate(cfg, upstream): return _timed("validation", validate.run, cfg)

@task(name="3_data_preparation", **RETRY)
def t_prepare(cfg, upstream): return _timed("preparation", prepare.run, cfg)

@task(name="4_data_transformation", **RETRY)
def t_transform(cfg, upstream): return _timed("transformation", transform.run, cfg)

@task(name="5_feature_management", **RETRY)
def t_features(cfg, upstream): return _timed("feature_management", feature_store.run, cfg)

@task(name="6_model_training", **RETRY)
def t_train(cfg, upstream): return _timed("model_training", train.run, cfg)

@task(name="7_model_evaluation", **RETRY)
def t_evaluate(cfg, upstream): return _timed("model_evaluation", evaluate.run, cfg)


@flow(name="recommendation-pipeline", log_prints=True)
def reco_flow(config_path: str | None = None):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    reset_run_log()
    r1 = t_ingest(cfg)
    r2 = t_validate(cfg, r1)
    r3 = t_prepare(cfg, r2)
    r4 = t_transform(cfg, r3)
    r5 = t_features(cfg, r4)
    r6 = t_train(cfg, r5)
    r7 = t_evaluate(cfg, r6)
    record = [r1, r2, r3, r4, r5, r6, r7]
    (path_for(cfg, "reports") / "run_summary.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    if "--serve" in sys.argv:
        reco_flow.serve(name="daily-reco-pipeline", cron="0 2 * * *")
    else:
        reco_flow()
