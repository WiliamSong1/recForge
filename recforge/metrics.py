import numpy as np


def ranking_metrics(recommended, relevant, k=10):
    """Binary relevance; precision divides by k even for short result lists."""
    if not relevant:
        raise ValueError("Ranking metrics require a nonempty relevance set")
    top = list(recommended[:k])
    if len(top) != len(set(top)):
        raise ValueError("Recommendations must be unique")
    hit = np.array([i in relevant for i in top], dtype=float)
    dcg = float((hit / np.log2(np.arange(len(hit)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
    positions = np.flatnonzero(hit)
    return {"recall@5": float(hit[:5].sum() / len(relevant)), "recall@10": float(hit.sum() / len(relevant)),
            "precision@10": float(hit.sum() / k), "ndcg@10": dcg / ideal,
            "hit_rate@10": float(hit.any()), "mrr@10": float(1 / (positions[0] + 1)) if len(positions) else 0.0}


def paired_bootstrap(a, b, seed=42, repetitions=2000):
    """Paired user bootstrap for mean(b-a); not a multi-training-seed interval."""
    delta = np.asarray(b) - np.asarray(a)
    if not len(delta):
        raise ValueError("No users to bootstrap")
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"mean_delta": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "users": len(delta)}
