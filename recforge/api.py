"""Local profile serving: persistent identities, versioned cache, immutable models."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from threadpoolctl import threadpool_limits
from .engine import Engine

ModelName = Literal["popularity", "mf", "two_tower", "two_tower_ann", "two_stage"]


class Rating(BaseModel):
    movie_id: int = Field(gt=0)
    rating: float = Field(ge=1, le=5)


class Feedback(BaseModel):
    event_id: str
    movie_id: int = Field(gt=0)
    liked: bool


def cache_key(user, revision, version, model, count, genre):
    return hashlib.sha256(json.dumps([user, revision, version, model, count, genre], separators=(",", ":")).encode()).hexdigest()


def create_app(artifact=None, database=None):
    artifact = Path(artifact or os.getenv("RECFORGE_ARTIFACT", "artifacts/run-42"))
    database = str(database or os.getenv("RECFORGE_DATABASE", "profiles.db"))
    registry = CollectorRegistry()
    requests = Counter("recforge_recommendations_total", "Recommendation requests", ["model"], registry=registry)
    latency = Histogram("recforge_recommendation_seconds", "Recommendation latency", registry=registry)
    hits = Counter("recforge_cache_hits_total", "Recommendation cache hits", registry=registry)
    cache, cache_lock, inference_lock = {}, threading.Lock(), threading.Lock()

    def connect():
        db = sqlite3.connect(database, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = sqlite3.Row
        return db

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        with connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS ratings(user_id TEXT REFERENCES users(id), movie_id INTEGER, rating REAL, PRIMARY KEY(user_id,movie_id));
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id), payload TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS feedback(event_id TEXT REFERENCES events(id), movie_id INTEGER, liked INTEGER, created REAL);
            ''')
        try:
            app.state.engine = Engine.load(artifact)
        except (OSError, ValueError, KeyError):
            pass
        yield
        cache.clear()

    app = FastAPI(title="RecForge", version="0.1.0", lifespan=lifespan,
                  description="Local profile API. Profile adaptation is separate from offline training-user benchmarks.")

    def engine():
        if app.state.engine is None:
            raise HTTPException(503, "Model artifacts unavailable; configure RECFORGE_ARTIFACT")
        return app.state.engine

    def authenticate(authorization: str = Header(default="")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer profile token required")
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        with connect() as db:
            user = db.execute("SELECT * FROM users WHERE token_hash=?", (digest,)).fetchone()
        if user is None:
            raise HTTPException(401, "Invalid profile token")
        return dict(user)

    def authorize(user, user_id):
        if user["id"] != user_id:
            raise HTTPException(403, "This token does not own the profile")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(model=Depends(engine)):
        return {"status": "ready", "version": model.version}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/movies")
    def movies(q: str = "", limit: int = Query(20, ge=1, le=100), model=Depends(engine)):
        return model.movies[model.movies.title.str.contains(q, case=False, regex=False)].head(limit).to_dict(orient="records")

    @app.get("/movies/{movie_id}")
    def movie(movie_id: int, model=Depends(engine)):
        frame = model.movies[model.movies.movie_id == movie_id]
        if frame.empty:
            raise HTTPException(404, "Unknown movie")
        return frame.iloc[0].to_dict()

    @app.post("/users", status_code=201)
    def new_user():
        user_id, token = str(uuid.uuid4()), secrets.token_urlsafe(32)
        with connect() as db:
            db.execute("INSERT INTO users(id,token_hash) VALUES (?,?)", (user_id, hashlib.sha256(token.encode()).hexdigest()))
        return {"user_id": user_id, "token": token}

    @app.post("/users/{user_id}/ratings")
    def rate(user_id: str, rating: Rating, user=Depends(authenticate), model=Depends(engine)):
        authorize(user, user_id)
        if rating.movie_id not in set(model.movies.movie_id):
            raise HTTPException(404, "Unknown movie")
        with connect() as db:
            db.execute("INSERT INTO ratings VALUES (?,?,?) ON CONFLICT(user_id,movie_id) DO UPDATE SET rating=excluded.rating",
                       (user_id, rating.movie_id, rating.rating))
            db.execute("UPDATE users SET revision=revision+1 WHERE id=?", (user_id,))
        return {"status": "saved"}

    @app.get("/users/{user_id}/recommendations")
    def recommend(user_id: str, model: ModelName = "two_tower", count: int = Query(10, ge=1, le=50),
                  genre: str | None = None, user=Depends(authenticate), service=Depends(engine)):
        authorize(user, user_id)
        begin = time.perf_counter()
        requests.labels(model).inc()
        with connect() as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM users WHERE id=?", (user_id,)).fetchone()[0]
            ratings = {r[0]: r[1] for r in db.execute("SELECT movie_id,rating FROM ratings WHERE user_id=?", (user_id,))}
        key = cache_key(user_id, revision, service.version, model, count, genre)
        with cache_lock:
            entry = cache.get(key)
        cache_hit = entry is not None and time.monotonic() - entry[0] < 60
        if cache_hit:
            items = entry[1]
            hits.inc()
        else:
            with inference_lock, threadpool_limits(limits=service.config["threads"]):
                items = service.profile_recommend(ratings, model, count, genre)
            with cache_lock:
                if len(cache) >= 1024:
                    cache.pop(next(iter(cache)))
                cache[key] = (time.monotonic(), items)
        event_id = str(uuid.uuid4())
        with connect() as db:
            db.execute("INSERT INTO events VALUES (?,?,?,?)", (event_id, user_id, json.dumps({"model": model, "version": service.version, "items": items}), time.time()))
        elapsed = time.perf_counter() - begin
        latency.observe(elapsed)
        return {"user_id": user_id, "event_id": event_id, "model": model, "version": service.version,
                "strategy": "profile_adaptation", "recommendations": items, "cache_hit": cache_hit, "latency_ms": elapsed * 1000}

    @app.post("/recommendations/feedback")
    def feedback(body: Feedback, user=Depends(authenticate)):
        with connect() as db:
            event = db.execute("SELECT payload FROM events WHERE id=? AND user_id=?", (body.event_id, user["id"])).fetchone()
            if event is None:
                raise HTTPException(404, "Unknown recommendation event")
            if body.movie_id not in {r["movie_id"] for r in json.loads(event[0])["items"]}:
                raise HTTPException(422, "Movie was not in this recommendation event")
            db.execute("INSERT INTO feedback VALUES (?,?,?,?)", (body.event_id, body.movie_id, body.liked, time.time()))
        return {"status": "recorded"}

    @app.get("/models")
    def models(service=Depends(engine)):
        return {"version": service.version, "models": ["popularity", "mf", "two_tower", "two_tower_ann", "two_stage"],
                "profile_policy": "Centroid or ridge adaptation; offline metrics do not measure application profiles.",
                "profile_retrieval": "Exact filtered catalog search; offline ANN uses FAISS HNSW.", "rerank_alpha": service.alpha}

    @app.get("/models/active")
    def active(service=Depends(engine)):
        return {"version": service.version, "default": "two_tower", "rerank_alpha": service.alpha}

    return app


app = create_app()
