"""Plain-Python orchestration (also the single source of stage order for the Prefect flow).

    python -m src.pipeline            # run every stage in dependency order
    python -m src.pipeline --from validation
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from src.common import load_config, path_for, reset_run_log, seed_everything, get_logger
from src.evaluation import evaluate
from src.features import store as feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

log = get_logger("pipeline")

# (stage name, callable) in dependency order: each stage reads the previous stage's output
STAGES = [("ingestion", ingest.run), ("validation", validate.run), ("preparation", prepare.run),
          ("transformation", transform.run), ("feature_management", feature_store.run),
          ("model_training", train.run), ("model_evaluation", evaluate.run)]


def run_all(cfg: dict | None = None, start_from: str | None = None) -> list[dict]:
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    if start_from is None:
        reset_run_log()
    names = [n for n, _ in STAGES]
    todo = STAGES[names.index(start_from):] if start_from else STAGES
    record = []
    for name, fn in todo:
        t0 = time.time()
        log.info("=== START %s ===", name)
        try:
            fn(cfg)
            status = "SUCCESS"
        except Exception:
            log.error("Stage %s FAILED\n%s", name, traceback.format_exc())
            status = "FAILED"
        record.append({"stage": name, "status": status, "start": t0, "seconds": round(time.time() - t0, 2)})
        log.info("=== END %s [%s] %.1fs ===", name, status, record[-1]["seconds"])
        if status == "FAILED":
            break                      # downstream stages depend on this one
    (path_for(cfg, "reports") / "run_summary.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start_from", choices=[n for n, _ in STAGES])
    ap.add_argument("--config")
    a = ap.parse_args()
    out = run_all(load_config(a.config), a.start_from)
    raise SystemExit(0 if all(r["status"] == "SUCCESS" for r in out) else 1)
