import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def train_ranker(engine, rank_frame, config):
    """Learn on future-of-base labels, using exclusively base-period features.

    Training candidates are ANN hard negatives plus positives from the label window.
    Unobserved items are implicit negatives, not evidence of explicit dislike.
    """
    features, labels = [], []
    eligible = set(engine.eligible)
    for u, group in rank_frame.groupby("u", sort=True):
        if engine.counts[u] < config["min_history"]:
            continue
        positives = set(group.loc[group.rating >= config["positive_rating"], "i"]) & eligible - engine.seen[u]
        if not positives:
            continue
        candidates, _ = engine.index.search(engine.user_vectors[u], config["candidates"], engine.seen[u])
        items = np.array(sorted(set(candidates) | positives), dtype=np.int64)
        features.append(engine.features(u, items))
        labels.extend([int(i in positives) for i in items])
    if not features or len(set(labels)) < 2:
        raise ValueError("Insufficient temporal reranking labels")
    X, y = np.concatenate(features), np.asarray(labels)
    # No random validation split: fixed iterations; outer temporal validation selects blending.
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.06,
                                           l2_regularization=5, early_stopping=False, random_state=config["seed"])
    with threadpool_limits(limits=config["threads"]):
        model.fit(X, y)
    return model, {"examples": len(y), "positives": int(y.sum()), "features": ["tower_affinity", "mf_rating", "genre_affinity", "log_popularity", "smoothed_item_rating", "log_user_count", "smoothed_user_rating"]}
