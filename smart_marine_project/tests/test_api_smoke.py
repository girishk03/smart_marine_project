import pytest


def test_health_endpoint_when_detector_unavailable():
    from fastapi.testclient import TestClient
    import os

    os.environ["SMART_MARINE_DISABLE_AUTOLOAD"] = "1"

    import api_server

    api_server.detector = None

    client = TestClient(api_server.app)

    r = client.get("/health")
    assert r.status_code == 200

    payload = r.json()
    assert payload["detector_loaded"] is False
    assert payload["status"] == "unhealthy"


def test_detect_endpoint_returns_503_without_detector():
    from fastapi.testclient import TestClient
    import os

    os.environ["SMART_MARINE_DISABLE_AUTOLOAD"] = "1"

    import api_server
    api_server.detector = None

    client = TestClient(api_server.app)

    r = client.post(
        "/detect",
        files={"file": ("x.jpg", b"notanimage", "image/jpeg")},
        data={"confidence": "0.3", "line_thickness": "2"},
    )

    assert r.status_code == 503
