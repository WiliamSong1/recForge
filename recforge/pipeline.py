from pathlib import Path
import json
import subprocess
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import load, chronological
from .training import train_models
from .engine import Engine
from .ranking import train_ranker
from .evaluation import evaluate, rating_metrics, retrieval_benchmark, environment, write_report, comparisons


def fit(data, config):
    # Build reranker labels with strictly earlier base features.
    base, rank_period = chronological(data.train, config["rank_train_fraction"])
    mf, tower, early_logs = train_models(base, data.genres, len(data.users), config)
    print("Building temporal-base retrieval index", flush=True)
    early = Engine(base, data.genres, data.movies, mf, tower, config)
    print("Training temporal reranker", flush=True)
    ranker, rank_metadata = train_ranker(early, rank_period, config)
    rank_metadata.update({"feature_cutoff": int(base.timestamp.max()), "label_start": int(rank_period.timestamp.min()),
                          "label_end": int(rank_period.timestamp.max()), "feature_distribution_shift": "Final base models are refit on full train; outer validation checks blend usefulness."})
    # All final models receive identical full training history.
    print("Refitting full-history models", flush=True)
    mf, tower, logs = train_models(data.train, data.genres, len(data.users), config, data.validation)
    engine = Engine(data.train, data.genres, data.movies, mf, tower, config)
    engine.ranker = ranker
    print("Selecting validation blend", flush=True)
    blend_scores = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        engine.alpha = alpha
        frame, summary, _ = evaluate(engine, data.validation, models=("two_stage",))
        warm = frame[frame.cohort == "established"]
        score = float(warm["ndcg@10"].mean()) if len(warm) else float(frame["ndcg@10"].mean())
        blend_scores.append({"alpha": alpha, "validation_ndcg@10": score})
    # Prefer less reranking on tied validation quality.
    engine.alpha = max(blend_scores, key=lambda r: r["validation_ndcg@10"])["alpha"]
    return engine, mf, tower, {"base_training": early_logs, "final_training": logs, "ranker": rank_metadata, "blend_selection": blend_scores}


def run(data_path, output, config, tracking_uri=None, final_test=False):
    if output.exists():
        raise FileExistsError("Use a new run directory to preserve existing artifacts")
    started = time.perf_counter()
    data = load(data_path, config)
    with threadpool_limits(limits=config["threads"]):
        engine, mf, tower, training = fit(data, config)
        fit_seconds = time.perf_counter() - started
        heldout = data.test if final_test else data.validation
        frame, quality, exclusions = evaluate(engine, heldout)
        # Also measure pure reranking, even when validation prefers retrieval alone.
        selected_alpha = engine.alpha
        engine.alpha = 1.0
        _, pure_quality, _ = evaluate(engine, heldout, models=("two_stage",))
        engine.alpha = selected_alpha
        users = frame.loc[frame.cohort == "established", "user"].unique()[:500]
        retrieval = retrieval_benchmark(engine, users) if len(users) else []
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    report = {"dataset": data.metadata, "config": config, "split": "test" if final_test else "validation",
              "quality": quality, "pure_reranker_quality": pure_quality, "rating_prediction_mf": rating_metrics(engine, heldout),
              "paired_ndcg_comparisons": comparisons(frame, config["seed"]), "retrieval": retrieval,
              "exclusions": exclusions, "training_seconds": fit_seconds, "selected_alpha": selected_alpha,
              "training": training, "git_revision": revision, "environment": environment(),
              "protocol": {"universe": "all items observed in train", "relevance": "rating >= 4", "history": "fixed training snapshot", "bootstrap_unit": "user", "training_seeds": 1}}
    engine.save(output, {"dataset": data.metadata, "user_ids_by_index": data.users, "training": training, "git_revision": revision})
    torch.save({"mf": mf.state_dict(), "tower": tower.state_dict()}, output / "training.pt")
    begin = time.perf_counter()
    Engine.load(output)
    report["model_load_seconds"] = time.perf_counter() - begin
    report["artifact_version"] = engine.version
    result_path = Path("benchmarks/results") / output.name
    write_report(result_path, report, frame)
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("RecForge")
        with mlflow.start_run(run_name=output.name):
            mlflow.log_params(config)
            mlflow.set_tags({"dataset_sha256": data.metadata["ratings_sha256"], "split": report["split"], "artifact_version": engine.version,
                             "git_revision": revision or "uncommitted"})
            mlflow.log_metric("training_seconds", fit_seconds)
            for row in quality:
                mlflow.log_metric(f"{row['cohort']}.{row['model']}.ndcg10", row["ndcg@10"])
            mlflow.log_artifacts(str(result_path), artifact_path="evaluation")
            mlflow.log_artifacts(str(output), artifact_path="model")
    print(json.dumps({"report": str(result_path / "report.json"), "quality": quality, "paired": report["paired_ndcg_comparisons"]}, indent=2), flush=True)
    return report
