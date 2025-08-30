import time
from pathlib import Path

from fastapi.testclient import TestClient

import api


def test_health_and_videos(tmp_path):
    with TestClient(api.app) as client:
        # Health endpoint test
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True

        # /videos with one video file
        (tmp_path / "v.mp4").write_bytes(b"00")
        r = client.get("/videos", params={"directory": str(tmp_path)})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert any(v["name"] == "v.mp4" for v in body["videos"])
        for v in body["videos"]:
            assert {"path", "name", "size"} <= v.keys()

        # /videos with empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        r = client.get("/videos", params={"directory": str(empty_dir)})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["videos"] == []

        # /videos with invalid/non-existent directory
        invalid_dir = tmp_path / "does_not_exist"
        r = client.get("/videos", params={"directory": str(invalid_dir)})
        # Expecting 400 or 404 depending on API implementation, adjust as needed
        assert r.status_code in (400, 404)


def test_video_detail_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("FFPROBE_DISABLE", "1")
    vid = tmp_path / "a.mp4"
    vid.write_bytes(b"00")
    with TestClient(api.app) as client:
        r = client.get(f"/videos/{vid.name}", params={"directory": str(tmp_path)})
        assert r.status_code == 200
        data = r.json()
        assert "artifacts" in data
        tinfo = data["artifacts"]["thumbs"]
        assert {"url", "exists"} <= tinfo.keys()
        assert tinfo["url"].endswith(".jpg")
        r_list = client.get("/videos", params={"directory": str(tmp_path), "detail": 1})
        assert r_list.status_code == 200
        vdata = r_list.json()["videos"][0]
        assert "artifacts" in vdata
        ltinfo = vdata["artifacts"]["thumbs"]
        assert {"url", "exists"} <= ltinfo.keys()
        assert ltinfo["url"].endswith(".jpg")


def test_job_metadata_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("FFPROBE_DISABLE", "1")
    (tmp_path / "v.mp4").write_bytes(b"00")
    with TestClient(api.app) as client:
        payload = {
            "task": "metadata",
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
    assert (tmp_path / ".artifacts" / "v.ffprobe.json").exists()


def test_api_dupes_and_embed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("FFPROBE_DISABLE", "1")
    (tmp_path / "x.mp4").write_bytes(b"00")
    (tmp_path / "y.mp4").write_bytes(b"00")
    with TestClient(api.app) as client:
        # phash job
        job = client.post("/jobs", json={"task": "phash", "directory": str(tmp_path), "params": {"frames": 1}}).json()
        for _ in range(30):
            j = client.get(f"/jobs/{job['id']}").json()
            if j["status"] == "done":
                break
            time.sleep(0.1)
        # dupes endpoint
        r_dup = client.get("/phash/duplicates", params={"directory": str(tmp_path), "threshold": 0.5})
        assert r_dup.status_code == 200
        assert "pairs" in r_dup.json()
        # embed job
        job2 = client.post("/jobs", json={"task": "embed", "directory": str(tmp_path), "params": {"sample_rate": 1.0}}).json()
        for _ in range(30):
            j2 = client.get(f"/jobs/{job2['id']}").json()
            if j2["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert j2["status"] == "done"
        assert j2["result"] == 0


def test_api_invalid_task():
    with TestClient(api.app) as client:
        r = client.post("/jobs", json={"task": "not-a-task", "directory": "."})
        assert r.status_code == 400


def test_api_tags_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("FFPROBE_DISABLE", "1")
    (tmp_path / "a.mp4").write_bytes(b"00")
    with TestClient(api.app) as client:
        # Initially empty
        r0 = client.get(f"/videos/a.mp4/tags", params={"directory": str(tmp_path)})
        assert r0.status_code == 200
        assert r0.json()["tags"] == []
        # Add tags via PATCH
        r1 = client.patch(f"/videos/a.mp4/tags", params={"directory": str(tmp_path)}, json={"add": ["x", "y"]})
        assert r1.status_code == 200
        assert set(r1.json()["tags"]) == {"x", "y"}
        # Remove one and add another
        r2 = client.patch(f"/videos/a.mp4/tags", params={"directory": str(tmp_path)}, json={"remove": ["x"], "add": ["z"]})
        assert r2.status_code == 200
        assert set(r2.json()["tags"]) == {"y", "z"}
        # Summary
        rs = client.get("/tags/summary", params={"directory": str(tmp_path)})
        assert rs.status_code == 200
        summ = rs.json()
        assert summ["tags"].get("y") == 1
        assert summ["tags"].get("z") == 1
        # Filtering (match all)
        rf = client.get("/videos", params={"directory": str(tmp_path), "tags": "y,z"})
        assert rf.status_code == 200
        assert rf.json()["count"] == 1
        # Filtering (match any)
        rfa = client.get("/videos", params={"directory": str(tmp_path), "tags": "y,foo", "match_any": 1})
        assert rfa.status_code == 200
        assert rfa.json()["count"] == 1
        # ETag conditional
        first = client.get("/videos", params={"directory": str(tmp_path)})
        etag = first.json()["etag"]
        not_mod = client.get("/videos", params={"directory": str(tmp_path)}, headers={"If-None-Match": etag})
        assert not_mod.status_code == 304


    def test_api_cancel_metadata_job(tmp_path, monkeypatch):
        monkeypatch.setenv("FFPROBE_DISABLE", "1")
        # Create many stub videos to keep job busy long enough to cancel
        for i in range(80):
            (tmp_path / f"v{i}.mp4").write_bytes(b"00")
        from fastapi.testclient import TestClient
        import api
        with TestClient(api.app) as client:
            job = client.post("/jobs", json={"task": "metadata", "directory": str(tmp_path), "params": {"force": True}}).json()
            job_id = job["id"]
            # Issue cancel quickly
            cancel_resp = client.delete(f"/jobs/{job_id}")
            assert cancel_resp.status_code == 200
            # Wait for final state
            final_status = None
            for _ in range(40):
                st = client.get(f"/jobs/{job_id}").json()
                if st["status"] in ("canceled", "done", "error"):
                    final_status = st["status"]
                    break
                time.sleep(0.1)
            assert final_status == "canceled", f"expected canceled got {final_status}"
