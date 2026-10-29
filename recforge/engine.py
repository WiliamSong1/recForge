from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import torch
from .data import histories
from .retrieval import RetrievalIndex


class Engine:
    def __init__(self, frame, genres, movies, mf, tower, config):
        self.config = config
        self.genres = genres
        self.movies = movies
        self.n_users = mf.user.num_embeddings
        self.seen = histories(frame, self.n_users)
        self.counts = np.bincount(frame.u, minlength=self.n_users)
        self.eligible = np.sort(frame.i.unique())
        self.popularity = np.bincount(frame.loc[frame.rating >= config["positive_rating"], "i"], minlength=len(movies)).astype(float)
        self.item_count = np.bincount(frame.i, minlength=len(movies)).astype(float)
        self.item_mean = (np.bincount(frame.i, weights=frame.rating, minlength=len(movies)) + 10 * frame.rating.mean()) / (self.item_count + 10)
        self.user_mean = (np.bincount(frame.u, weights=frame.rating, minlength=self.n_users) + 5 * frame.rating.mean()) / (self.counts + 5)
        self.preferences = np.zeros((self.n_users, genres.shape[1]), dtype=np.float32)
        positive = frame[frame.rating >= config["positive_rating"]]
        np.add.at(self.preferences, positive.u.to_numpy(), genres[positive.i.to_numpy()])
        self.preferences /= np.maximum(self.preferences.sum(1, keepdims=True), 1)
        with torch.inference_mode():
            self.user_vectors = tower.users(torch.arange(self.n_users)).numpy()
            self.item_vectors = tower.items(torch.arange(len(movies))).numpy()
            self.mf_users = mf.user.weight.numpy().copy()
            self.mf_items = mf.item.weight.numpy().copy()
            self.mf_ub = mf.ub.weight.numpy().ravel().copy()
            self.mf_ib = mf.ib.weight.numpy().ravel().copy()
            self.mean = float(mf.mean)
        self.ranker = None
        self.alpha = 1.0
        self.version = "unversioned"
        self.index = RetrievalIndex(self.item_vectors, self.eligible, config["threads"])

    def mf_scores(self, u):
        return self.mean + self.mf_ub[u] + self.mf_ib + self.mf_items @ self.mf_users[u]

    def features(self, u, items):
        similarity = self.item_vectors[items] @ self.user_vectors[u]
        mf_scores = self.mean + self.mf_ub[u] + self.mf_ib[items] + self.mf_items[items] @ self.mf_users[u]
        return self.feature_values(items, similarity, mf_scores, self.preferences[u], self.counts[u], self.user_mean[u])

    def feature_values(self, items, similarity, mf_scores, preference, count, user_mean):
        return np.column_stack([similarity, mf_scores, self.genres[items] @ preference,
                                np.log1p(self.popularity[items]), self.item_mean[items],
                                np.full(len(items), np.log1p(count)), np.full(len(items), user_mean)])

    @staticmethod
    def ordered(items, scores, k):
        order = np.lexsort((items, -np.asarray(scores)))[:k]
        return np.asarray(items)[order], np.asarray(scores)[order]

    def recommend(self, u, model, k=10, candidates=None):
        seen = self.seen[u]
        candidates = candidates or self.config["candidates"]
        if self.counts[u] < self.config["min_history"]:
            model = "popularity"
        if model in ("two_tower_ann", "two_stage"):
            items, scores = self.index.search(self.user_vectors[u], max(k, candidates), seen)
            if model == "two_stage" and len(items):
                if self.ranker is None:
                    raise RuntimeError("Ranker is not loaded")
                rank = self.ranker.predict_proba(self.features(u, items))[:, 1]
                # Blend rank positions, avoiding arbitrary incompatible raw score scales.
                scores = self.blend(scores, rank)
            return self.ordered(items, scores, k)
        if model == "popularity":
            scores = self.popularity
        elif model == "mf":
            scores = self.mf_scores(u)
        elif model == "two_tower":
            scores = self.item_vectors @ self.user_vectors[u]
        else:
            raise ValueError(f"Unknown model: {model}")
        items = np.array([i for i in self.eligible if i not in seen], dtype=np.int64)
        return self.ordered(items, scores[items], k)

    def blend(self, retrieval_scores, rank_scores):
        n = len(rank_scores)
        def percentile(values):
            return np.argsort(np.argsort(values, kind="stable"), kind="stable") / max(n - 1, 1)
        return (1 - self.alpha) * percentile(retrieval_scores) + self.alpha * percentile(rank_scores)

    def profile_recommend(self, ratings, model, k, genre=None):
        """Application profiles never address training-user embeddings."""
        ids = {int(mid): i for i, mid in enumerate(self.movies.movie_id)}
        known = [(ids[mid], rating) for mid, rating in ratings.items() if mid in ids]
        seen = {i for i, _ in known}
        items = np.array([i for i in self.eligible if i not in seen and
                          (genre is None or genre in self.movies.iloc[i].genres.split("|"))], dtype=np.int64)
        if not len(items):
            return []
        weights = np.array([max(r - 3, 0) for _, r in known])
        predicted = None
        reason = "Popular among training users"
        if not known or weights.sum() == 0 or model == "popularity":
            scores = self.popularity[items]
        else:
            history = np.array([i for i, _ in known])
            query = (self.item_vectors[history] * weights[:, None]).sum(0)
            query /= max(np.linalg.norm(query), 1e-8)
            preference = (self.genres[history] * weights[:, None]).sum(0)
            preference /= max(preference.sum(), 1)
            design = self.mf_items[history]
            targets = np.array([r for _, r in known]) - self.mean - self.mf_ib[history]
            adapted = np.linalg.solve(design.T @ design + np.eye(design.shape[1]), design.T @ targets)
            predicted = np.clip(self.mean + self.mf_ib[items] + self.mf_items[items] @ adapted, 1, 5)
            scores = self.item_vectors[items] @ query
            reason = "Based on embeddings of movies you rated positively"
            if model == "mf":
                scores = predicted
                reason = "Based on a regularized factor fit to your ratings"
            elif model in ("two_tower_ann", "two_stage"):
                # Filtered full-catalog candidate selection avoids underfilled genre results.
                order = np.lexsort((items, -scores))[:max(k, self.config["candidates"])]
                items, scores, predicted = items[order], scores[order], predicted[order]
                if model == "two_stage":
                    features = self.feature_values(items, scores, predicted, preference, len(known), np.mean([r for _, r in known]))
                    scores = self.blend(scores, self.ranker.predict_proba(features)[:, 1])
                    reason = "Reranked using your profile, movie affinity, and training popularity"
        order = np.lexsort((items, -scores))[:k]
        return [{"movie_id": int(self.movies.iloc[items[j]].movie_id), "title": str(self.movies.iloc[items[j]].title),
                 "score": float(scores[j]), "predicted_rating": float(predicted[j]) if predicted is not None and model == "mf" else None,
                 "reason": reason} for j in order]

    def save(self, path: Path, metadata):
        path.mkdir(parents=True, exist_ok=False)
        array_names = ["genres", "counts", "eligible", "popularity", "item_count", "item_mean", "user_mean", "preferences",
                       "user_vectors", "item_vectors", "mf_users", "mf_items", "mf_ub", "mf_ib"]
        np.savez_compressed(path / "arrays.npz", **{key: getattr(self, key) for key in array_names})
        self.movies.to_json(path / "movies.json", orient="records")
        (path / "seen.json").write_text(json.dumps([sorted(s) for s in self.seen]))
        joblib.dump(self.ranker, path / "ranker.joblib")
        self.version = hashlib.sha256((path / "arrays.npz").read_bytes() + (path / "ranker.joblib").read_bytes()
                                      + json.dumps({"config": self.config, "alpha": self.alpha}, sort_keys=True).encode()).hexdigest()[:16]
        manifest = {"schema_version": 1, "version": self.version, "config": self.config, "alpha": self.alpha,
                    "mean": self.mean, "n_users": self.n_users, "metadata": metadata}
        manifest["files"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir() if f.is_file()}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path: Path):
        """Load only trusted locally generated artifacts (joblib uses pickle)."""
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported artifact schema")
        for name, checksum in manifest["files"].items():
            if Path(name).name != name or hashlib.sha256((path / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self = cls.__new__(cls)
        for key in ("config", "alpha", "mean", "n_users", "version"):
            setattr(self, key, manifest[key])
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            for key in arrays.files:
                setattr(self, key, arrays[key])
        self.movies = pd.read_json(path / "movies.json")
        self.seen = [set(s) for s in json.loads((path / "seen.json").read_text())]
        self.ranker = joblib.load(path / "ranker.joblib")
        self.index = RetrievalIndex(self.item_vectors, self.eligible, self.config["threads"])
        return self
