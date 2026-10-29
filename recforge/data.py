"""Official ingestion, stable mappings, and strictly ordered temporal partitions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-1m.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        # Extract only known files; never arbitrary archive paths.
        for name in ("ratings.dat", "movies.dat", "users.dat", "README"):
            (destination / name).write_bytes(z.read(f"ml-1m/{name}"))
    metadata = {"url": URL, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (destination / "source.json").write_text(json.dumps(metadata, indent=2))
    return destination


def chronological(frame, fraction):
    """A timestamp is never split across partitions, including at tied cutoffs."""
    times = np.sort(frame.timestamp.unique())
    if len(times) < 3:
        raise ValueError("Need at least three distinct timestamps")
    cutoff = frame.timestamp.quantile(fraction, interpolation="lower")
    left, right = frame[frame.timestamp <= cutoff].copy(), frame[frame.timestamp > cutoff].copy()
    if left.empty or right.empty:
        raise ValueError("Temporal partition is empty")
    assert left.timestamp.max() < right.timestamp.min()
    return left, right


@dataclass
class Dataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    genres: np.ndarray
    users: list
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict


def load(path: Path, config: dict) -> Dataset:
    ratings = pd.read_csv(path / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv(path / "movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["movie_id", "title", "genres"])
    if ratings.isna().any().any() or movies.isna().any().any():
        raise ValueError("Missing values in source")
    before = len(ratings)
    for name in ("user_id", "movie_id", "rating", "timestamp"):
        ratings[name] = pd.to_numeric(ratings[name], errors="coerce")
    valid_numbers = np.isfinite(ratings[["user_id", "movie_id", "rating", "timestamp"]]).all(axis=1)
    integer_ids = (ratings[["user_id", "movie_id", "timestamp"]] % 1 == 0).all(axis=1)
    ratings = ratings[valid_numbers & integer_ids].copy()
    for name in ("user_id", "movie_id", "timestamp"):
        ratings[name] = ratings[name].astype(np.int64)
    movies["movie_id"] = pd.to_numeric(movies.movie_id, errors="raise")
    if not ((movies.movie_id > 0) & (movies.movie_id % 1 == 0)).all():
        raise ValueError("Movie IDs must be positive integers")
    if movies.movie_id.duplicated().any():
        raise ValueError("Movie catalog IDs must be unique")
    ratings = ratings[ratings.rating.between(1, 5) & (ratings.timestamp > 0)
                      & (ratings.user_id > 0) & ratings.movie_id.isin(movies.movie_id)]
    ratings = ratings.sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
    # Keep the first observation; selecting a last rating would let the future alter history.
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="first").reset_index(drop=True)
    movies = movies.drop_duplicates("movie_id").sort_values("movie_id").reset_index(drop=True)
    users = sorted(ratings.user_id.unique().tolist())
    ratings["u"] = ratings.user_id.map({v: i for i, v in enumerate(users)}).astype(int)
    ratings["i"] = ratings.movie_id.map({v: i for i, v in enumerate(movies.movie_id)}).astype(int)
    vocab = sorted({g for s in movies.genres for g in s.split("|")})
    genres = np.array([[float(g in s.split("|")) for g in vocab] for s in movies.genres], dtype=np.float32)
    train, rest = chronological(ratings, config["train_fraction"])
    relative = config["validation_fraction"] / (1 - config["train_fraction"])
    validation, test = chronological(rest, relative)
    digest = hashlib.sha256((path / "ratings.dat").read_bytes()).hexdigest()
    metadata = {"ratings_sha256": digest, "removed_records": before - len(ratings),
                "genre_vocabulary": vocab, "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
                "cutoffs": {"train_end": int(train.timestamp.max()), "validation_end": int(validation.timestamp.max())},
                "split_protocol": "global-temporal-v1", "mapping_policy": "sorted IDs; no outcome features"}
    return Dataset(ratings, movies, genres, users, train, validation, test, metadata)


def histories(frame, n_users):
    result = [set() for _ in range(n_users)]
    for u, i in frame[["u", "i"]].itertuples(index=False, name=None):
        result[u].add(i)
    return result
