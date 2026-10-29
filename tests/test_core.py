import numpy as np
import pandas as pd
import pytest
import torch
from recforge.data import chronological, load
from recforge.metrics import ranking_metrics, paired_bootstrap
from recforge.models import sample_negatives
from recforge.training import train_models
from recforge.engine import Engine
from recforge.api import cache_key


def test_metrics_hand_calculation():
    m = ranking_metrics([3, 2, 1], {1, 2})
    assert m["recall@10"] == 1
    assert m["precision@10"] == .2
    assert m["mrr@10"] == .5
    assert m["ndcg@10"] == pytest.approx((1 / np.log2(3) + .5) / (1 + 1 / np.log2(3)))
    assert ranking_metrics([], {1})["ndcg@10"] == 0
    with pytest.raises(ValueError):
        ranking_metrics([1, 1], {1})
    with pytest.raises(ValueError):
        ranking_metrics([1], set())


def test_bootstrap_paired():
    result = paired_bootstrap([.1, .3, .5], [.2, .4, .6])
    assert result["mean_delta"] == pytest.approx(.1)
    assert result["ci95_low"] == pytest.approx(.1)


def test_split_ties_do_not_cross():
    df = pd.DataFrame({"timestamp": [1, 1, 2, 2, 3, 3, 4]})
    a, b = chronological(df, .5)
    assert a.timestamp.max() < b.timestamp.min()
    assert len(a) + len(b) == len(df)


def test_ingestion(tmp_path, config):
    (tmp_path / "movies.dat").write_text("8::Eight::Comedy\n3::Three::Drama\n")
    lines = [f"{u}::{3 if t % 2 else 8}::4::{t}" for t, u in enumerate(range(10, 40), 1)]
    lines += ["10::3::5::200"]
    (tmp_path / "ratings.dat").write_text("\n".join(lines))
    data = load(tmp_path, config)
    assert data.metadata["removed_records"] == 1
    assert data.ratings.iloc[0].rating == 4
    assert data.movies.movie_id.tolist() == [3, 8]
    assert data.train.timestamp.max() < data.validation.timestamp.min()
    assert data.validation.timestamp.max() < data.test.timestamp.min()


def test_negative_sampling():
    seen = [{0, 1}, {1, 2}]
    users = np.tile([0, 1], 100)
    neg = sample_negatives(users, seen, [0, 1, 2], np.random.default_rng(1))
    assert all(i not in seen[u] for u, i in zip(users, neg))
    with pytest.raises(ValueError):
        sample_negatives([0], [{0, 1}], [0, 1], np.random.default_rng(1))


def test_training_determinism_and_finite(config, fixture_data):
    frame, genres, _ = fixture_data
    a, b, logs = train_models(frame, genres, 8, config)
    c, d, _ = train_models(frame, genres, 8, config)
    assert all(np.isfinite(row["train_loss"]) for row in logs)
    assert torch.equal(a.user.weight, c.user.weight)
    assert torch.equal(b.item_id.weight, d.item_id.weight)
    assert b.items(torch.arange(24)).shape == (24, 8)


def test_filtering_retrieval_and_serialization(engine, tmp_path):
    exact, _ = engine.index.search(engine.user_vectors[0], 5, engine.seen[0], approximate=False)
    ann, _ = engine.index.search(engine.user_vectors[0], 5, engine.seen[0])
    assert set(exact) == set(ann)
    for model in ("popularity", "mf", "two_tower", "two_tower_ann", "two_stage"):
        items, scores = engine.recommend(0, model, 5)
        assert not set(items) & engine.seen[0]
        assert len(items) == len(set(items)) == 5
        assert np.isfinite(scores).all()
    path = tmp_path / "artifact"
    engine.save(path, {})
    loaded = Engine.load(path)
    assert np.array_equal(engine.recommend(0, "two_stage")[0], loaded.recommend(0, "two_stage")[0])
    with (path / "arrays.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        Engine.load(path)


def test_profile_cold_and_adaptation(engine):
    cold = engine.profile_recommend({}, "two_stage", 5)
    assert len(cold) == 5
    warm = engine.profile_recommend({cold[0]["movie_id"]: 5}, "mf", 5)
    assert cold[0]["movie_id"] not in {r["movie_id"] for r in warm}
    assert all(1 <= r["predicted_rating"] <= 5 for r in warm)
    assert engine.profile_recommend({}, "mf", 5, genre="nonexistent") == []


def test_cache_key_isolation():
    key = cache_key("a", 1, "v1", "mf", 10, None)
    for args in [("b", 1, "v1", "mf", 10, None), ("a", 2, "v1", "mf", 10, None),
                 ("a", 1, "v2", "mf", 10, None), ("a", 1, "v1", "mf", 10, "Drama")]:
        assert key != cache_key(*args)
