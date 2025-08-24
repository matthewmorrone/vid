import time
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import api


def test_health_and_videos(tmp_path):
    with TestClient(api.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        (tmp_path / "v.mp4").write_bytes(b"00")
        r = client.get("/videos", params={"directory": str(tmp_path)})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert any(Path(p).name == "v.mp4" for p in body["videos"])


def test_job_meta_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("FFPROBE_DISABLE", "1")
    (tmp_path / "v.mp4").write_bytes(b"00")
    with TestClient(api.app) as client:
        payload = {
            "task": "meta",
            "directory": str(tmp_path),
            "recursive": False,
            "force": True,
            "params": {},
        }
        r = client.post("/jobs", json=payload)
        assert r.status_code == 200
        job_id = r.json()["id"]
        for _ in range(20):
            r2 = client.get(f"/jobs/{job_id}")
            if r2.json()["status"] == "done":
                break
            time.sleep(0.1)
        else:
            raise AssertionError("job did not finish")
        report = client.get("/report", params={"directory": str(tmp_path)})
        assert report.status_code == 200
        data = report.json()
        assert data["counts"]["metadata"] == 1
    assert (tmp_path / "v.mp4.ffprobe.json").exists()
