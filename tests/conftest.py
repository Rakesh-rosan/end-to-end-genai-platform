import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        llm_provider="mock",
        embedding_provider="hashing",
        db_path=tmp_path / "commerce.db",
        cdc_enabled=False,  # tests drive CDC explicitly via poll_once()
        rate_limit_per_minute=1000,
        redis_url=None,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def login(client):
    def _login(username: str) -> dict:
        password = {"alice": "alice123", "bob": "bob123", "agent": "agent123"}[username]
        r = client.post("/auth/token", json={"username": username, "password": password})
        assert r.status_code == 200
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _login


@pytest.fixture
def ask(client, login):
    def _ask(username: str, message: str, session_id: str = "s1") -> dict:
        r = client.post("/v1/chat", json={"message": message, "session_id": session_id}, headers=login(username))
        assert r.status_code == 200, r.text
        return r.json()

    return _ask
