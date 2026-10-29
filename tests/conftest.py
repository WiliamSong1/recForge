import numpy as np
import pandas as pd
import pytest
from recforge.models import MatrixFactorization, TwoTower
from recforge.engine import Engine
from recforge.ranking import train_ranker
from recforge.training import seed_everything


@pytest.fixture
def config():
    return dict(seed=42, dimension=8, epochs=2, batch_size=32, learning_rate=.01,
                weight_decay=.0001, patience=2, candidates=12, min_history=2,
                positive_rating=4., train_fraction=.8, validation_fraction=.1,
                rank_train_fraction=.8, threads=1, device="cpu")


@pytest.fixture
def fixture_data():
    rng = np.random.default_rng(4)
    rows = []
    for u in range(8):
        for t, i in enumerate(rng.choice(24, 12, replace=False)):
            rows.append((u, int(i), float(4 + (t % 2)), 100 + t * 20 + u))
    frame = pd.DataFrame(rows, columns=["u", "i", "rating", "timestamp"])
    genres = np.eye(4, dtype=np.float32)[np.arange(24) % 4]
    movies = pd.DataFrame({"movie_id": np.arange(24) + 1, "title": [f"Movie {i}" for i in range(24)], "genres": [f"G{i % 4}" for i in range(24)]})
    return frame, genres, movies


@pytest.fixture
def engine(config, fixture_data):
    frame, genres, movies = fixture_data
    seed_everything(42, 1)
    base, rank = frame[frame.timestamp < 240], frame[frame.timestamp >= 240]
    service = Engine(base, genres, movies, MatrixFactorization(8, 24, 8), TwoTower(8, 24, genres, 8), config)
    service.ranker, _ = train_ranker(service, rank, config)
    return service
