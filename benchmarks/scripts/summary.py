"""Generate the readable benchmark report and comparison chart from measured runs."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recforge-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    reports = [json.loads(Path(f"benchmarks/results/run-{s}/report.json").read_text()) for s in (42, 43, 44)]
    rows = []
    for report in reports:
        for row in report["quality"]:
            if row["cohort"] == "established":
                rows.append({"model": row["model"], "seed": report["config"]["seed"], "ndcg": row["ndcg@10"]})
        pure = next(r for r in report["pure_reranker_quality"] if r["cohort"] == "established")
        rows.append({"model": "pure_reranker", "seed": report["config"]["seed"], "ndcg": pure["ndcg@10"]})
    order = ["popularity", "mf", "two_tower", "two_tower_ann", "two_stage", "pure_reranker"]
    labels = ["Popularity", "Rating MF", "Two-tower\nexact", "Two-tower\nANN", "Two-stage\nblend", "Pure\nreranker"]
    summary = pd.DataFrame(rows).groupby("model").ndcg.agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    colors = ["#98a8b7", "#98a8b7", "#3c82b5", "#3c82b5", "#188574", "#c28a68"]
    ax.bar(np.arange(len(order)), summary["mean"], yerr=summary["std"], color=colors, capsize=5)
    for index, value in enumerate(summary["mean"]):
        ax.text(index, value + summary.iloc[index]["std"] + .004, f"{value:.4f}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(order)), labels)
    ax.set_ylim(0, .25)
    ax.set_ylabel("Test NDCG@10")
    ax.set_title("RecForge · MovieLens 1M\n985 established users · three seeds · mean ± sample SD", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    Path("benchmarks/plots").mkdir(exist_ok=True)
    fig.savefig("benchmarks/plots/three-seed-comparison.png", dpi=180)
    fig.savefig("benchmarks/plots/three-seed-comparison.svg")
    plt.close(fig)
    lift = summary.loc["two_stage", "mean"] / summary.loc["two_tower_ann", "mean"] - 1
    text = ["# Measured benchmark results", "", "Python 3.12, MovieLens 1M, CPU execution. Global temporal 80/10/10 split; full training-observed catalog, seen items excluded. Seeds 42, 43, 44 use the same configuration except random seed. Each seed selects its blend on validation before test scoring.", "", "## Established-user quality", "", "985 users with at least five training interactions and eligible test positives. Values are mean ± sample standard deviation across three independently trained seeds.", "", "| Model | Test NDCG@10 |", "|---|---:|"]
    for model, row in summary.iterrows():
        text.append(f"| {model} | {row['mean']:.4f} ± {row['std']:.4f} |")
    text += ["", f"The validation-selected two-stage blend improves mean NDCG@10 by **{lift:.1%} relative** to ANN retrieval. Pure reranking hurts quality. Two-tower retrieval beats rating MF but is close to popularity. This is not evidence that neural retrieval consistently beats popularity.", "", "![Three-seed comparison](plots/three-seed-comparison.png)", "", "## Paired uncertainty and selection", "", "Paired user-bootstrap intervals for the absolute NDCG@10 difference (blend minus ANN). These are conditional on each fitted model; the table above separately reports training-seed variability.", "", "| Seed | Validation-selected rank weight | Test difference | 95% bootstrap interval |", "|---|---:|---:|---|"]
    for r in reports:
        pair = r["paired_ndcg_comparisons"]["established:two_stage-minus-two_tower_ann"]
        text.append(f"| {r['config']['seed']} | {r['selected_alpha']:.2f} | {pair['mean_delta']:.5f} | [{pair['ci95_low']:.5f}, {pair['ci95_high']:.5f}] |")
    r = reports[0]
    text += ["", "All five systems use popularity fallback for the 202 cold users, giving identical NDCG@10 of 0.1900. Another 22 test users have no eligible positive and are excluded; 50 positive test interactions are outside the eligible unseen catalog. These exclusions are visible in every raw report.", "", f"Seed 42 MF rating prediction: RMSE **{r['rating_prediction_mf']['rmse']:.4f}**, MAE **{r['rating_prediction_mf']['mae']:.4f}**, on {r['rating_prediction_mf']['observations']:,} warm user-item ratings. Rating accuracy does not establish ranking quality.", "", "## Retrieval and serving", "", "Seed 42 retrieval compares the first 500 established test users. Timings include seen-item filtering and single-query search.", "", "| Candidates | ANN recall against exact | Exact mean ms | ANN mean ms |", "|---:|---:|---:|---:|"]
    for row in r["retrieval"]:
        text.append(f"| {row['candidate_pool']} | {row['ann_recall_vs_exact']:.4f} | {row['exact_mean_ms']:.3f} | {row['ann_mean_ms']:.3f} |")
    text += ["", "ANN provides no convincing speed advantage at this catalog size. Recall at 300 drops with fixed efSearch=128; that configuration is not suitable for high-recall large candidate pools. This is a measured limit, not a scale claim.", ""]
    serving_path = Path("benchmarks/results/serving.json")
    if serving_path.exists():
        serving = json.loads(serving_path.read_text())
        text += ["Real loopback HTTP, one Uvicorn process, 100 requests per row, ten results per response, three rated movies per application profile. SQLite reads and event writes are included. The cold/warm distinction refers to the application recommendation cache, not OS page caches. No training jobs ran during the timed workload; an earlier stalled diagnostic process remained resident, so machine isolation is not guaranteed.", "", "| Two-stage concurrency | Cache | Mean ms | P95 ms | Requests/sec | Hit rate |", "|---:|---|---:|---:|---:|---:|"]
        for row in serving["results"]:
            if row["model"] == "two_stage":
                text.append(f"| {row['concurrency']} | {row['cache']} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} | {row['rps']:.1f} | {row['cache_hit_rate']:.0%} |")
        text += ["", "Uncached throughput falls under concurrent load: the local service serializes inference and persists events in SQLite. This is a limitation to address before production scaling. Cached performance is not a substitute for measuring uncached latency.", ""]
    text += [f"Training-pipeline durations (including preprocessing and validation blend selection) were {', '.join(format(r["training_seconds"], ".1f") for r in reports)} seconds. Seed 42 model loading took {reports[0]['model_load_seconds']:.3f} seconds; peak training-process RSS was {reports[0]['environment']['peak_process_rss_bytes']/2**20:.1f} MiB. These are single local observations.", "", "Runtime: macOS 15.7.4 ARM64, Python 3.12.15, PyTorch 2.15.0, FAISS 1.15.1, NumPy 2.5.4. Four configured Torch/BLAS threads; one thread during HNSW construction. The sandbox denied CPU-brand/RAM queries, so the hardware description is incomplete. Raw environment metadata is included in reports; performance is not a cross-machine guarantee.", "", "## Reproduce and inspect", "", "Run the commands in the project README. `benchmarks/scripts/aggregate.py` validates matching runs and exports seed summaries; `benchmarks/scripts/summary.py` rebuilds this report and chart. `benchmarks/scripts/stages.py` measures retrieval, feature construction, and ranking separately. Full reports, per-user metrics, and retrieval CSVs are under `results/run-42`, `results/run-43`, and `results/run-44`.", "", "14 synthetic tests passed, including end-to-end training/report generation, leakage invariance, deterministic training, serialization, cache invalidation, and API identity isolation. A real MLflow run (`run-43`) completed with 16 metrics, 15 parameters, model artifacts, and evaluation artifacts. Optuna's search entry point is implemented but a full tuning study was not run. Docker/CI configurations were authored but not executed on their target platforms.", "", "## Resume wording supported by these measurements", "", f"Built a Python/PyTorch recommendation pipeline on MovieLens 1M with chronological evaluation, FAISS HNSW retrieval, and temporal supervised reranking; improved established-user NDCG@10 by {lift:.1%} over neural ANN retrieval across three seeds using validation-selected blending.", "", "Paired this with a FastAPI profile service, versioned artifacts, MLflow logging, leakage tests, and measured cache/concurrency benchmarks. Avoid claiming production deployment, online business impact, ANN acceleration at scale, or that pure reranking improved quality."]
    Path("benchmarks/RESULTS.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
