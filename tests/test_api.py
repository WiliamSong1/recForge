from fastapi.testclient import TestClient
from recforge.api import create_app


def test_profile_lifecycle(engine, tmp_path):
    path = tmp_path / "model"
    engine.save(path, {})
    with TestClient(create_app(path, tmp_path / "profiles.db")) as client:
        assert client.get("/ready").status_code == 200
        user = client.post("/users").json()
        other = client.post("/users").json()
        headers = {"Authorization": "Bearer " + user["token"]}
        url = f"/users/{user['user_id']}/recommendations"
        assert client.get(url).status_code == 401
        assert client.get(f"/users/{other['user_id']}/recommendations", headers=headers).status_code == 403
        first = client.get(url, headers=headers).json()
        assert not first["cache_hit"]
        assert client.get(url, headers=headers).json()["cache_hit"]
        movie = first["recommendations"][0]["movie_id"]
        assert client.post(f"/users/{user['user_id']}/ratings", headers=headers, json={"movie_id": movie, "rating": 5}).status_code == 200
        response = client.get(url, headers=headers).json()
        assert not response["cache_hit"]
        assert movie not in {r["movie_id"] for r in response["recommendations"]}
        assert client.post("/recommendations/feedback", headers=headers, json={"event_id": first["event_id"], "movie_id": movie, "liked": True}).status_code == 200
        assert client.post(f"/users/{user['user_id']}/ratings", headers=headers, json={"movie_id": movie, "rating": 7}).status_code == 422
        assert "recforge_recommendations_total" in client.get("/metrics").text


def test_unready(tmp_path):
    with TestClient(create_app(tmp_path / "missing", tmp_path / "profiles.db")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/movies").status_code == 503
