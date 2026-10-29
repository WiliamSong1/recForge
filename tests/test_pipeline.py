import json
import numpy as np
import pandas as pd
from recforge.data import load
from recforge.pipeline import run
from recforge.engine import Engine
from recforge.evaluation import evaluate


def write_fixture(path, fixture_data):
    frame, _, movies = fixture_data
    path.mkdir()
    (path / "ratings.dat").write_text("\n".join(f"{u+1}::{i+1}::{r}::{t}" for u, i, r, t in frame.itertuples(index=False, name=None)))
    (path / "movies.dat").write_text("\n".join(f"{r.movie_id}::{r.title}::{r.genres}" for r in movies.itertuples()))


def test_full_pipeline(tmp_path, monkeypatch, fixture_data, config):
    # Tiny temporal experiment exercises training -> reranking -> artifacts -> report.
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    write_fixture(path, fixture_data)
    config = dict(config, epochs=1)
    report = run(path, tmp_path / "artifacts" / "smoke", config)
    assert report["split"] == "validation"
    assert report["training"]["ranker"]["feature_cutoff"] < report["training"]["ranker"]["label_start"]
    assert (tmp_path / "benchmarks/results/smoke/report.json").exists()
    assert (tmp_path / "benchmarks/plots/smoke/ndcg.png").exists()
    assert report["model_load_seconds"] > 0
    assert 0 <= report["selected_alpha"] <= 1
    assert all(np.isfinite(row["ndcg@10"]) for row in report["quality"])


def test_future_ratings_do_not_change_training(tmp_path, fixture_data, config):
    path = tmp_path / "data"
    write_fixture(path, fixture_data)
    before = load(path, config)
    lines = (path / "ratings.dat").read_text().splitlines()
    changed = []
    cutoff = before.metadata["cutoffs"]["validation_end"]
    for line in lines:
        values = line.split("::")
        if int(values[3]) > cutoff:
            values[2] = "1"
        changed.append("::".join(values))
    (path / "ratings.dat").write_text("\n".join(changed))
    after = load(path, config)
    pd.testing.assert_frame_equal(before.train, after.train)
    pd.testing.assert_frame_equal(before.validation, after.validation)


def test_cold_cohort_uses_same_fallback(engine):
    engine.counts[0] = 0
    candidates = [i for i in engine.eligible if i not in engine.seen[0]]
    heldout = pd.DataFrame({"u": [0], "i": [candidates[0]], "rating": [5.]})
    frame, summaries, _ = evaluate(engine, heldout)
    assert set(frame.cohort) == {"cold"}
    assert len(set(tuple(row) for row in frame["items"])) == 1
