"""Stage 1 - Data ingestion.

Reads users / items / interactions CSVs (generating synthetic ones first when
`data.source: synthetic`), keeps every column as *string* so type problems are caught by
the validation stage instead of being silently coerced, and writes a manifest
(row counts + MD5) for lineage/reproducibility.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.common import get_logger, path_for
from src.ingestion.generate_synthetic import generate

log = get_logger("ingestion")
TABLES = ["users", "items", "interactions"]


def run(cfg: dict) -> dict:
    raw, processed = path_for(cfg, "raw"), path_for(cfg, "processed")
    if cfg["data"]["source"] == "synthetic":
        log.info("Generating synthetic data (seed=%s)", cfg["seed"])
        for name, df in generate(cfg).items():
            df.to_csv(raw / f"{name}.csv", index=False)
    manifest = {}
    for t in TABLES:
        f = raw / f"{t}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - provide it or use data.source: synthetic")
        df = pd.read_csv(f, dtype=str)
        df.to_parquet(processed / f"ingested_{t}.parquet", index=False)
        manifest[t] = {"rows": len(df), "columns": list(df.columns),
                       "md5": hashlib.md5(f.read_bytes()).hexdigest()}
        log.info("Ingested %-12s rows=%d md5=%s", t, len(df), manifest[t]["md5"][:8])
    (processed / "ingestion_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
