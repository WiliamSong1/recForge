"""Real HTTP load benchmark. Run against a local Uvicorn process only.

Creates benchmark profiles and events; use a disposable RECFORGE_DATABASE.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import httpx
import numpy as np
from recforge.evaluation import environment


def benchmark(url, requests=100):
    result = []
    with httpx.Client(base_url=url, timeout=30) as client:
        assert client.get("/ready").status_code == 200
        movies = client.get("/movies?limit=3").json()
        profiles = []
        for _ in range(requests):
            response = client.post("/users")
            response.raise_for_status()
            user = response.json()
            headers = {"Authorization": "Bearer " + user["token"]}
            for movie in movies:
                response = client.post(f"/users/{user['user_id']}/ratings", headers=headers,
                                       json={"movie_id": movie["movie_id"], "rating": 5})
                response.raise_for_status()
            profiles.append((user["user_id"], headers))
        for model in ("mf", "two_tower", "two_stage"):
            for concurrency in (1, 4, 8):
                count = 10
                # Advance profile revisions outside the timed region to invalidate all
                # old cache keys while preserving an identical recommendation workload.
                for user, headers in profiles:
                    response = client.post(f"/users/{user}/ratings", headers=headers,
                                           json={"movie_id": movies[0]["movie_id"], "rating": 5})
                    response.raise_for_status()
                for cache_state in ("cold", "warm"):
                    def request(index):
                        user, headers = profiles[index]
                        start = time.perf_counter()
                        response = client.get(f"/users/{user}/recommendations", params={"model": model, "count": count}, headers=headers)
                        response.raise_for_status()
                        payload = response.json()
                        return (time.perf_counter() - start) * 1000, payload["cache_hit"]
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        samples = list(pool.map(request, range(requests)))
                    seconds = time.perf_counter() - start
                    latencies = np.array([s[0] for s in samples])
                    result.append({"model": model, "concurrency": concurrency, "cache": cache_state, "requests": requests, "recommendation_count": count,
                                   "rps": requests / seconds, "mean_ms": float(latencies.mean()),
                                   "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
                                   "p99_ms": float(np.percentile(latencies, 99)), "cache_hit_rate": float(np.mean([s[1] for s in samples]))})
    return {"environment": environment(), "transport": "HTTP loopback", "results": result,
            "scope": "Single Uvicorn process; SQLite event writes included; adapted app profiles, not offline user embeddings"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/serving.json"))
    args = parser.parse_args()
    report = benchmark(args.url, args.requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
