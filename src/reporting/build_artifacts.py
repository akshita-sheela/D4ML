"""Generates reports/architecture_diagram.png, reports/workflow_run.png (from the real
run_summary.json written by the last pipeline run) and reports/final_report.pdf."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common import load_config, path_for

STAGES = [("Data Ingestion", "CSV / synthetic\nraw -> parquet\nmanifest + MD5"),
          ("Data Validation", "missing, dupes, dtypes\nrating/timestamp, FK\nerrors -> data/errors"),
          ("Data Preparation", "standardise, impute\ndedupe (user,item)\nis_positive flag"),
          ("Transformation", "time-based split\nactivity / popularity\nsparse matrices"),
          ("Feature Mgmt", "custom feature store\nuser/item/pair feats\ncatalog + snapshots"),
          ("Model Training", "Popularity, ItemKNN,\nSVD-MF\nMLflow params+artifacts"),
          ("Model Evaluation", "P/R/NDCG@10, coverage\ndiversity, RMSE/MAE\nMLflow metrics")]


def architecture(out):
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
    w, gap = 1.8, 0.35
    for i, (t, s) in enumerate(STAGES):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.7), w, 1.8, boxstyle="round,pad=0.04", fc="#e8f1fb", ec="#2b6cb0"))
        ax.text(x + w / 2, 3.2, t, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, 2.4, s, ha="center", va="center", fontsize=7.5)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 2.6), (x + w + gap - 0.02, 2.6), arrowstyle="-|>", mutation_scale=12))
    ax.add_patch(FancyBboxPatch((0.2, 0.15), 14.5, 0.9, boxstyle="round,pad=0.04", fc="#fdf3e1", ec="#b7791f"))
    ax.text(7.45, 0.6, "Storage: data/raw  ->  data/processed (parquet, npz)  ->  data/features/{train,serving}  |  data/errors  |  "
            "models/*.joblib  |  mlflow.db  |  reports/", ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((0.2, 3.9), 14.5, 0.55, boxstyle="round,pad=0.04", fc="#e6f6ec", ec="#2f855a"))
    ax.text(7.45, 4.18, "Orchestration: Prefect flow (dags/reco_flow.py) - sequential task dependencies, retries, cron schedule  |  "
            "fallback: python -m src.pipeline", ha="center", va="center", fontsize=8.5)
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close()


def workflow_run(summary_path, out):
    rec = json.loads(summary_path.read_text())
    fig, ax = plt.subplots(figsize=(11, 4))
    t0 = rec[0]["start"]
    for i, r in enumerate(rec):
        ok = r["status"] == "SUCCESS"
        ax.barh(i, r["seconds"], left=r["start"] - t0, color="#38a169" if ok else "#e53e3e")
        ax.text(r["start"] - t0 + r["seconds"] + 0.05, i, f"{'OK' if ok else 'FAILED'}  {r['seconds']}s", va="center", fontsize=8)
    ax.set_yticks(range(len(rec))); ax.set_yticklabels([f"{i+1}. {r['stage']}" for i, r in enumerate(rec)])
    ax.invert_yaxis(); ax.set_xlabel("seconds since flow start")
    ax.set_title("Prefect flow 'recommendation-pipeline' - run summary (all tasks completed)" if all(r["status"] == "SUCCESS" for r in rec)
                 else "Pipeline run summary (FAILED)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def final_report(cfg, rep):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    val = json.loads((rep / "data_validation_report.json").read_text())
    res = pd.read_csv(rep / "model_comparison.csv")
    split = json.loads((path_for(cfg, "processed") / "split_meta.json").read_text())
    k = cfg["evaluation"]["k"]
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=11.5, spaceAfter=4)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3, keepWithNext=1)
    doc = SimpleDocTemplate(str(rep / "final_report.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.4 * cm, title="Recommendation Pipeline - Final Report")
    S = []
    P = lambda t, s=body: S.append(Paragraph(t, s))

    def table(data, widths=None, fs=7.5):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        S.extend([t, Spacer(1, 5)])

    P("End-to-End Data Management Pipeline for a Recommendation System", ss["Title"])
    P("Name / Student ID: ______________________ &nbsp;&nbsp; Course: ______________________ &nbsp;&nbsp; Repository: &lt;paste GitHub link&gt;")
    P("1. Overview and design decisions", h)
    P("The pipeline recommends items to users from past interactions. It runs seven stages - ingestion, validation, preparation, "
      "transformation, feature management, training and evaluation - orchestrated by a Prefect flow. Each stage is an independent module "
      "that reads the previous stage's files from disk (parquet / npz), so any stage can be re-run alone and every intermediate dataset can be inspected. "
      "Behaviour is driven by <i>configs/config.yaml</i> and a fixed seed and reference date, which makes runs reproducible.")
    P(f"<b>Data.</b> The build environment had no internet access, so the data are documented, seeded synthetic data "
      f"({cfg['data']['n_users']} users, {cfg['data']['n_items']} items, about {cfg['data']['n_interactions']} interactions, 1-5 ratings) generated from a latent-factor "
      "model with popularity skew, and with known data-quality defects injected so that validation has real work to do. The CSV schema "
      "(users, items, interactions) is the contract: a public dataset such as MovieLens converted to it can be dropped in with <i>data.source: csv</i>. "
      "Results below are therefore for synthetic data and should not be read as MovieLens-level numbers.")
    S.append(Image(str(rep / "architecture_diagram.png"), width=17.2 * cm, height=17.2 * cm * 4.6 / 15)); S.append(Spacer(1, 4))
    P("2. Stages and tools", h)
    P("<b>Validation</b> (pandas rule engine): missing values, duplicate keys and rows, dtype errors, rating range [1,5], unparseable / future / pre-signup "
      "timestamps and referential integrity (orphan users/items). Failing rows go to <i>data/errors/&lt;table&gt;_errors.csv</i> with an error_reason; "
      "optional-field gaps are warnings and are imputed later. All columns are ingested as strings so type problems are detected rather than silently coerced.")
    rows = [["Table", "Total", "Valid", "Invalid", "Main rules triggered"]]
    for t, s in val.items():
        top = ", ".join(f"{k_}={v}" for k_, v in sorted(s["rule_violations"].items(), key=lambda kv: -kv[1])[:3])
        rows.append([t, s["total_rows"], s["valid_rows"], s["invalid_rows"], top])
    table(rows, [2.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 10 * cm])
    P(f"<b>Preparation / transformation.</b> Standardise gender and category spelling, impute age (median) and country, keep the latest interaction per (user,item), "
      f"flag positives (rating &gt;= {cfg['evaluation']['positive_threshold']}). A <b>time-based split</b> at the 80th percentile of timestamps "
      f"({split['cutoff']}) gives {split['train_rows']} train / {split['test_rows']} test rows. Derived datasets (user activity summary, item popularity, "
      f"sparse user-item matrices, density {split['matrix_density']}) are built from the training period only to avoid leakage.")
    P("<b>Feature management.</b> A small custom offline feature store computes 21 documented features (user, item and user-item pair; see reports/feature_catalog.md), e.g. "
      "user average rating, activity count, preferred category, days since last interaction, item popularity and smoothed average rating. The same code builds a 'train' "
      "snapshot (history before the cutoff) and a 'serving' snapshot (all data), and models read features only through its accessors, so training and prediction share one definition. "
      "I chose a custom layer over Feast because the project is batch-only and a full feature-store deployment would add infrastructure without adding insight.")
    P("<b>Models</b> (scikit-learn / SciPy): (1) popularity baseline ranking by positive-interaction count; (2) item-based collaborative filtering (cosine similarity, 50 neighbours); "
      "(3) matrix factorisation with TruncatedSVD (32 factors) for ranking, plus a bias baseline with low-rank residual model for rating prediction. "
      "<b>MLflow</b> (SQLite backend) records parameters, metrics and model artifacts for every model.")
    P("3. Results", h)
    cols = ["model", f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "rmse", "mae"]
    data = [cols] + [[("-" if (isinstance(v, float) and v != v) else (f"{v:.4f}" if isinstance(v, float) else v)) for v in r] for r in res[cols].values.tolist()]
    table(data, fs=8)
    S.append(Image(str(rep / "model_comparison.png"), width=12.5 * cm, height=12.5 * cm * 4 / 9)); S.append(Spacer(1, 3))
    best = res.set_index("model")
    P(f"Evaluation uses the time-based test period: for every train-known user with at least one positive test interaction, top-{k} items not seen in training are compared with "
      f"the user's test positives. The personalised models clearly beat the popularity baseline: SVD-MF reaches NDCG@{k} {best.loc['svd_mf', f'ndcg@{k}']:.3f} vs "
      f"{best.loc['popularity', f'ndcg@{k}']:.3f} for popularity. Popularity recommends the same few items to everyone (coverage "
      f"{best.loc['popularity','coverage']:.2f}) while item-KNN covers most of the catalogue ({best.loc['item_knn','coverage']:.2f}) but with lower category diversity "
      f"({best.loc['item_knn','diversity']:.2f}), i.e. the accuracy/diversity trade-off is visible. The bias+residual rating model improves RMSE from "
      f"{best.loc['popularity','rmse']:.3f} to {best.loc['svd_mf','rmse']:.3f} (MAE {best.loc['popularity','mae']:.3f} to {best.loc['svd_mf','mae']:.3f}). Item-KNN only ranks, so it has no RMSE/MAE.")
    P("4. Orchestration and reproducibility", h)
    P("<i>dags/reco_flow.py</i> defines a Prefect flow of seven tasks; each task receives its upstream task's result, so a stage starts only after its predecessor succeeded and a failure "
      "skips everything downstream. Tasks retry once on failure and the flow can be scheduled with a cron expression (<i>flow.serve</i>). "
      "<i>python -m src.pipeline</i> runs the same stages without Prefect. A successful run is shown in reports/workflow_run.png (generated from the flow's recorded task results) and logged in reports/pipeline_run.log.")
    S.append(Image(str(rep / "workflow_run.png"), width=12 * cm, height=12 * cm * 4 / 11))
    P("5. Limitations", h)
    P("(a) Synthetic data: the ground-truth structure is simple and the injected errors are artificial; conclusions about absolute accuracy do not transfer to real catalogues. "
      "(b) Cold start is only handled with fallbacks (global/item means); new users get no personalised ranking. "
      "(c) Item-KNN and SVD build dense item/user factors in memory, fine for 10<super>3</super>-10<super>4</super> items but not for large catalogues (use ALS/ANN there). "
      "(d) Single time split and single seed: no confidence intervals or hyper-parameter search. "
      "(e) The rating model treats unobserved residuals as zero, a simplification versus a proper SGD/ALS fit. "
      "(f) Offline metrics do not capture online behaviour, and diversity is measured only by category. "
      "(g) The Prefect run uses a local ephemeral server; production would need a deployed server/worker and alerting. "
      "Future work: MovieLens run, content/hybrid model, hyper-parameter tuning in MLflow, data drift checks, Great Expectations suites.")
    doc.build(S)


def run(cfg: dict | None = None):
    cfg = cfg or load_config()
    rep = path_for(cfg, "reports")
    architecture(rep / "architecture_diagram.png")
    workflow_run(rep / "run_summary.json", rep / "workflow_run.png")
    final_report(cfg, rep)
    return str(rep / "final_report.pdf")


if __name__ == "__main__":
    print(run())
