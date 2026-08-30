"""
Authentication and rate limiting.

What is actually being protected is a pair of third-party quotas — Gemini
allows 20 generations/day per model on the free tier — so these tests care
about two questions: can an unauthenticated caller reach an endpoint that
spends quota, and can an authenticated one spend it without limit.
"""

import pytest
from fastapi.testclient import TestClient

from api.config import settings
from api.main import _check_deployment_config


@pytest.fixture
def secured(app, monkeypatch):
    """The app with a key configured, as it would run in a deployment."""
    monkeypatch.setattr(settings, "api_key", "test-secret-key")
    return TestClient(app)


# ---- authentication --------------------------------------------------------

def test_requests_without_a_key_are_rejected(secured):
    for method, path, kwargs in [
        ("post", "/api/chat", {"json": {"session_id": "x", "question": "hi"}}),
        ("get", "/api/documents/x", {}),
        ("post", "/api/sessions/x/reset", {}),
        ("delete", "/api/sessions/x", {}),
    ]:
        response = getattr(secured, method)(path, **kwargs)
        assert response.status_code == 401, f"{method} {path} was not protected"
        assert "API key" in response.json()["detail"]


def test_upload_is_protected(secured, formula_pdf_bytes):
    response = secured.post(
        "/api/documents", files={"file": ("f.pdf", formula_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 401


def test_a_wrong_key_is_rejected(secured):
    response = secured.post(
        "/api/chat",
        json={"session_id": "x", "question": "hi"},
        headers={"X-API-Key": "not-the-key"},
    )
    assert response.status_code == 401


def test_the_right_key_is_accepted(secured):
    """404 rather than 401: authentication passed, the session simply is not real."""
    response = secured.post(
        "/api/chat",
        json={"session_id": "does-not-exist", "question": "hi"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 404


def test_health_stays_open_for_monitoring(secured):
    assert secured.get("/health").status_code == 200


def test_no_key_configured_means_an_open_api(client):
    """The local default: settings.api_key is None, so nothing is required."""
    assert settings.api_key is None
    assert client.get("/api/documents/nope").status_code == 404  # not 401


# ---- rate limiting ---------------------------------------------------------

def test_questions_are_capped_per_client(secured, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_questions", 3)
    headers = {"X-API-Key": "test-secret-key", "X-Client-Id": "rate-test-questions"}

    # Unknown session -> 404, which still passes through the limiter first.
    for _ in range(3):
        r = secured.post("/api/chat", json={"session_id": "x", "question": "hi"}, headers=headers)
        assert r.status_code == 404

    blocked = secured.post("/api/chat", json={"session_id": "x", "question": "hi"}, headers=headers)
    assert blocked.status_code == 429
    assert "Rate limit reached" in blocked.json()["detail"]
    assert blocked.headers["retry-after"]


def test_uploads_have_their_own_budget(secured, monkeypatch):
    """A spent question budget must not block uploads, and vice versa."""
    monkeypatch.setattr(settings, "rate_limit_uploads", 2)
    monkeypatch.setattr(settings, "rate_limit_questions", 1)
    headers = {"X-API-Key": "test-secret-key", "X-Client-Id": "rate-test-separate"}

    secured.post("/api/chat", json={"session_id": "x", "question": "hi"}, headers=headers)
    assert secured.post(
        "/api/chat", json={"session_id": "x", "question": "hi"}, headers=headers
    ).status_code == 429

    # Uploads still work: different counter.
    for _ in range(2):
        r = secured.post(
            "/api/documents", files={"file": ("f.txt", b"x", "text/plain")}, headers=headers
        )
        assert r.status_code == 400  # rejected on type, not on rate
    assert secured.post(
        "/api/documents", files={"file": ("f.txt", b"x", "text/plain")}, headers=headers
    ).status_code == 429


def test_clients_are_limited_independently(secured, monkeypatch):
    """One visitor exhausting their budget must not lock everyone else out."""
    monkeypatch.setattr(settings, "rate_limit_questions", 1)
    key = {"X-API-Key": "test-secret-key"}

    a = {**key, "X-Client-Id": "visitor-a"}
    b = {**key, "X-Client-Id": "visitor-b"}
    secured.post("/api/chat", json={"session_id": "x", "question": "hi"}, headers=a)
    assert secured.post(
        "/api/chat", json={"session_id": "x", "question": "hi"}, headers=a
    ).status_code == 429
    assert secured.post(
        "/api/chat", json={"session_id": "x", "question": "hi"}, headers=b
    ).status_code == 404


def test_a_limit_of_zero_disables_it(secured, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_questions", 0)
    headers = {"X-API-Key": "test-secret-key", "X-Client-Id": "rate-test-disabled"}
    for _ in range(6):
        r = secured.post("/api/chat", json={"session_id": "x", "question": "hi"}, headers=headers)
        assert r.status_code == 404


def test_a_broken_limiter_does_not_take_the_api_down(secured, monkeypatch, app):
    """Failing open is deliberate: the provider quotas are still enforced upstream."""
    def explode(*_args, **_kwargs):
        raise RuntimeError("database gone")

    monkeypatch.setattr(app.state.rag_state.storage, "record_and_count", explode)
    r = secured.post(
        "/api/chat",
        json={"session_id": "x", "question": "hi"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert r.status_code == 404  # served, not 500


# ---- deployment guards -----------------------------------------------------

def test_deployment_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(settings, "api_key", None)
    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        _check_deployment_config()


def test_deployment_refuses_a_wildcard_cors_origin(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setattr(settings, "allowed_origins", ["*"])
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        _check_deployment_config()


def test_a_correctly_configured_deployment_starts(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(settings, "api_key", "k")
    monkeypatch.setattr(settings, "allowed_origins", ["https://app.vercel.app"])
    _check_deployment_config()


def test_local_development_is_unaffected_by_the_guards(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(settings, "api_key", None)
    _check_deployment_config()  # no error
