"""
Tests for DeepGuard FastAPI backend endpoints.
Verifies Section 14 automated tests requirements:
- API: upload invalid extension -> 400
- API: unknown id -> 404
- API: stats keys present (total, analyzed, fake, real, avg_score, errors)
- Health check returns expected booleans
- Valid upload initiates processing and returns 8-character ID
"""

import io
import pytest
from fastapi.testclient import TestClient

from app import app
import config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /api/health returns status booleans."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "ffmpeg" in data
    assert "ffprobe" in data
    assert "exiftool" in data
    assert "ollama" in data
    assert "weights" in data
    assert isinstance(data["ffmpeg"], bool)
    assert isinstance(data["weights"], bool)


def test_stats_endpoint(client):
    """GET /api/stats returns all required summary counters."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"total", "analyzed", "fake", "real", "avg_score", "errors"}
    assert expected_keys.issubset(data.keys())
    assert isinstance(data["total"], int)
    assert isinstance(data["fake"], int)
    assert isinstance(data["real"], int)
    assert isinstance(data["avg_score"], (int, float))


def test_upload_invalid_extension(client):
    """POST /api/upload with unsupported extension returns HTTP 400."""
    fake_file = io.BytesIO(b"fake file content")
    response = client.post(
        "/api/upload",
        files={"file": ("unsupported_document.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Unsupported file extension" in detail


def test_get_unknown_video_id(client):
    """GET /api/videos/{unknown_id} returns HTTP 404."""
    response = client.get("/api/videos/nonexistent_id_999")
    assert response.status_code == 404
    assert "not found" in response.json().get("detail", "").lower()


def test_stream_unknown_video_id(client):
    """GET /api/video/{unknown_id} returns HTTP 404."""
    response = client.get("/api/video/nonexistent_id_999")
    assert response.status_code == 404


def test_upload_valid_video_file(client):
    """POST /api/upload with valid video returns 8-character ID."""
    fake_mp4 = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    response = client.post(
        "/api/upload",
        files={"file": ("sample_test_upload.mp4", fake_mp4, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert len(data["id"]) == 8

    # Verify record appears in /api/videos
    list_res = client.get("/api/videos")
    assert list_res.status_code == 200
    records = list_res.json()
    assert any(r["id"] == data["id"] for r in records)
