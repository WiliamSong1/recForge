"""Time retrieval, feature construction, and rank prediction separately."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from recforge.engine import Engine
from recforge.evaluation import environment


def measure(artifact):
    engine = Engine.load(artifact)
    users = np.flatnonzero(engine.counts >= engine.config["min_history"])[:500]
    result = []
    with threadpool_limits(limits=engine.config["threads"]):
        for size in (20, 100, 300):
            records = []
            for u in users:
                start = time.perf_counter()
                items, scores = engine.index.search(engine.user_vectors[u], size, engine.seen[u])
                retrieved = time.perf_counter()
                features = engine.features(u, items)
                featurized = time.perf_counter()
                ranking = engine.ranker.predict_proba(features)[:, 1]
                engine.ordered(items, engine.blend(scores, ranking), 10)
                finished = time.perf_counter()
                records.append([(retrieved-start)*1000, (featurized-retrieved)*1000, (finished-featurized)*1000])
            values = np.asarray(records)
            result.append({"candidates": size, "queries": len(users),
                           **{name: {"mean_ms": float(values[:, index].mean()), "p95_ms": float(np.percentile(values[:, index], 95))}
                              for index, name in enumerate(("retrieval", "features", "ranking_and_sorting"))}})
    return {"version": engine.version, "environment": environment(), "stages": result,
            "scope": "Single process, established training users; no HTTP or database; first query included"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/run-42"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/stages.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.artifact), indent=2))
