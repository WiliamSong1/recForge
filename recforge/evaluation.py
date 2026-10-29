import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .metrics import ranking_metrics, paired_bootstrap

MODELS = ("popularity", "mf", "two_tower", "two_tower_ann", "two_stage")


def evaluate(engine, heldout, models=MODELS):
    """Full training-catalog universe; fixed training history for every heldout user.

    Validation interactions are not added to test-time histories. This is a static
    snapshot experiment, not a rolling online simulation.
    """
    rows, catalog = [], set(engine.eligible)
    excluded_positives, no_relevance = 0, 0
    with threadpool_limits(limits=engine.config["threads"]):
        for u, group in heldout.groupby("u", sort=True):
            original = set(group.loc[group.rating >= engine.config["positive_rating"], "i"])
            relevant = original & catalog - engine.seen[u]
            excluded_positives += len(original - relevant)
            if not relevant:
                no_relevance += 1
                continue
            cohort = "established" if engine.counts[u] >= engine.config["min_history"] else "cold"
            for model in models:
                start = time.perf_counter()
                items, _ = engine.recommend(int(u), model)
                ms = (time.perf_counter() - start) * 1000
                rows.append({"user": int(u), "model": model, "cohort": cohort, **ranking_metrics(items, relevant),
                             "latency_ms": ms, "items": items.tolist()})
    if not rows:
        raise ValueError("No evaluable held-out users")
    frame = pd.DataFrame(rows)
    summaries = []
    for cohort in ("all", "established", "cold"):
        subset = frame if cohort == "all" else frame[frame.cohort == cohort]
        for model, part in subset.groupby("model", sort=False):
            summary = {"cohort": cohort, "model": model, "users": len(part)}
            for key in ("recall@5", "recall@10", "precision@10", "ndcg@10", "hit_rate@10", "mrr@10"):
                summary[key] = float(part[key].mean())
            summary["coverage"] = len({i for items in part["items"] for i in items}) / len(catalog)
            summary.update({f"latency_{p}_ms": float(np.percentile(part.latency_ms, p)) for p in (50, 95, 99)})
            summary["latency_mean_ms"] = float(part.latency_ms.mean())
            summaries.append(summary)
    return frame, summaries, {"excluded_positive_items": excluded_positives, "users_without_eligible_positives": no_relevance}


def rating_metrics(engine, heldout):
    warm = heldout[(heldout.u.map(lambda u: engine.counts[u] >= engine.config["min_history"])) & heldout.i.isin(engine.eligible)]
    if warm.empty:
        return {"observations": 0}
    u, i = warm.u.to_numpy(), warm.i.to_numpy()
    predictions = np.clip(engine.mean + engine.mf_ub[u] + engine.mf_ib[i] + (engine.mf_users[u] * engine.mf_items[i]).sum(1), 1, 5)
    error = predictions - warm.rating.to_numpy()
    return {"observations": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "clipped_to": [1, 5]}


def retrieval_benchmark(engine, users, pools=(20, 100, 300)):
    result = []
    for pool in pools:
        recalls, exact_times, ann_times = [], [], []
        for u in users:
            query, seen = engine.user_vectors[u], engine.seen[u]
            begin = time.perf_counter()
            exact, _ = engine.index.search(query, pool, seen, approximate=False)
            exact_times.append((time.perf_counter() - begin) * 1000)
            begin = time.perf_counter()
            ann, _ = engine.index.search(query, pool, seen, approximate=True)
            ann_times.append((time.perf_counter() - begin) * 1000)
            if len(exact):
                recalls.append(len(set(exact) & set(ann)) / len(exact))
        result.append({"candidate_pool": pool, "queries": len(users), "ann_recall_vs_exact": float(np.mean(recalls)),
                       "exact_mean_ms": float(np.mean(exact_times)), "ann_mean_ms": float(np.mean(ann_times)),
                       "ann_p95_ms": float(np.percentile(ann_times, 95))})
    return result


def environment():
    import torch, faiss, sklearn
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "hardware_note": "CPU brand and physical RAM unavailable in this sandbox; timings are local observations, not portable guarantees.", "torch": torch.__version__, "numpy": np.__version__,
            "faiss": faiss.__version__, "sklearn": sklearn.__version__,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}


def write_report(path: Path, report, frame):
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame(report["quality"]).to_csv(path / "quality.csv", index=False)
    frame.drop(columns="items").to_csv(path / "per_user.csv", index=False)
    pd.DataFrame(report["retrieval"]).to_csv(path / "retrieval.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = path.parent.parent / "plots" / path.name
    plots.mkdir(parents=True, exist_ok=True)
    quality = pd.DataFrame(report["quality"])
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for offset, cohort in enumerate(("established", "cold")):
        data = quality[quality.cohort == cohort].set_index("model").reindex(MODELS)
        if data["ndcg@10"].notna().any():
            ax.bar(np.arange(len(MODELS)) + (offset - 0.5) * .36, data["ndcg@10"], .36, label=cohort)
    ax.set_xticks(np.arange(len(MODELS)), MODELS, rotation=15)
    ax.set_ylabel("NDCG@10")
    ax.set_title(f"MovieLens 1M • global temporal {report['split']} • full training catalog")
    ax.legend()
    fig.savefig(plots / "ndcg.png", dpi=180)
    fig.savefig(plots / "ndcg.svg")
    plt.close(fig)


def comparisons(frame, seed):
    result = {}
    for cohort in ("all", "established", "cold"):
        part = frame if cohort == "all" else frame[frame.cohort == cohort]
        pivot = part.pivot(index="user", columns="model", values="ndcg@10")
        if pivot.empty:
            continue
        for a, b in (("mf", "two_tower"), ("two_tower_ann", "two_stage"), ("popularity", "mf")):
            result[f"{cohort}:{b}-minus-{a}"] = paired_bootstrap(pivot[a], pivot[b], seed)
    return result
