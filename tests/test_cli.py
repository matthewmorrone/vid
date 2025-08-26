import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
import api

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(args, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "index.py", *args], cwd=REPO_ROOT, capture_output=True, text=True, env=e)


def test_list_no_dir(tmp_path: Path):
    proc = run_cli(["list", str(tmp_path / "does_not_exist")])
    assert proc.returncode == 2


def test_list_and_json(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"00")
    (tmp_path / "b.txt").write_text("ignore")
    proc = run_cli(["list", str(tmp_path)])
    assert proc.returncode == 0
    assert "a.mp4" in proc.stdout
    assert "b.txt" not in proc.stdout
    proc_json = run_cli(["list", str(tmp_path), "--json"])
    data = json.loads(proc_json.stdout)
    assert any(d["name"] == "a.mp4" for d in data)


def test_artifact_commands_stub_mode(tmp_path: Path):
    # Create sample mp4 (stub content)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"00")
    env = {"FFPROBE_DISABLE": "1"}

    # meta
    r_meta = run_cli(["meta", str(tmp_path)], env)
    assert r_meta.returncode == 0, r_meta.stderr
    assert (tmp_path / ".artifacts" / "sample.ffprobe.json").exists()

    # thumb
    r_thumb = run_cli(["thumb", str(tmp_path)], env)
    assert r_thumb.returncode == 0, r_thumb.stderr
    assert (tmp_path / ".artifacts" / "sample.jpg").exists()

    # sprites
    r_sprites = run_cli(["sprites", str(tmp_path)], env)
    assert r_sprites.returncode == 0, r_sprites.stderr
    assert (tmp_path / ".artifacts" / "sample.sprites.jpg").exists()
    assert (tmp_path / ".artifacts" / "sample.sprites.json").exists()

    # previews
    r_prev = run_cli(["previews", str(tmp_path)], env)
    assert r_prev.returncode == 0, r_prev.stderr
    assert (tmp_path / ".artifacts" / "sample.previews.json").exists()

    # subs (default vtt)
    r_subs = run_cli(["subs", str(tmp_path)], env)
    assert r_subs.returncode == 0, r_subs.stderr
    assert (tmp_path / ".artifacts" / "sample.vtt").exists()

    # phash
    r_phash = run_cli(["phash", str(tmp_path), "--frames", "1", "--output-format", "json"], env)
    assert r_phash.returncode == 0, r_phash.stderr
    phash_file = tmp_path / ".artifacts" / "sample.phash.json"
    assert phash_file.exists()
    phash_data = json.loads(phash_file.read_text())
    assert "phash" in phash_data

    # heatmap
    r_heat = run_cli(["heatmap", str(tmp_path), "--output-format", "json"], env)
    assert r_heat.returncode == 0, r_heat.stderr
    assert (tmp_path / ".artifacts" / "sample.heatmap.json").exists()

    # scenes
    r_scenes = run_cli(["scenes", str(tmp_path), "--output-format", "json"], env)
    assert r_scenes.returncode == 0, r_scenes.stderr
    assert (tmp_path / ".artifacts" / "sample.scenes.json").exists()

    # faces
    r_faces = run_cli(["faces", str(tmp_path), "--output-format", "json"], env)
    assert r_faces.returncode == 0, r_faces.stderr
    assert (tmp_path / ".artifacts" / "sample.faces.json").exists()

    # report
    r_report = run_cli(["report", str(tmp_path), "--output-format", "json"], env)
    assert r_report.returncode == 0, r_report.stderr
    report_json = json.loads(r_report.stdout)
    # All artifacts should have count 1
    for key in ["metadata","thumb","sprites","previews","subs","phash","heatmap","scenes","faces"]:
        assert report_json["counts"][key] == 1, f"{key} missing in report"
        assert report_json["coverage"][key] == 1.0


def test_actor_build_and_match_stub(tmp_path: Path):
    people = tmp_path / "people"
    actor_dir = people / "A"
    actor_dir.mkdir(parents=True)
    (actor_dir / "a.jpg").write_bytes(b"00")
    env = {"DEEPFACE_STUB": "1"}
    emb = tmp_path / "gallery.npy"
    lab = tmp_path / "labels.json"
    r_build = run_cli(["actor-build", "--people-dir", str(people), "--embeddings", str(emb), "--labels", str(lab)], env)
    assert r_build.returncode == 0, r_build.stderr
    video = tmp_path / "v.mp4"
    video.write_bytes(b"00")
    r_match = run_cli(["actor-match", "--video", str(video), "--embeddings", str(emb), "--labels", str(lab)], env)
    assert r_match.returncode == 0, r_match.stderr
    faces_file = tmp_path / ".artifacts" / "v.faces.json"
    assert faces_file.exists()
    data = json.loads(faces_file.read_text())
    assert data["detections"][0]["accepted_label"] == "A"
    client = TestClient(api.app)
    resp = client.get(f"/videos/{video.name}/faces", params={"directory": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["detections"][0]["accepted_label"] == "A"
def test_meta_parallel_stub_mode(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"00")
    (tmp_path / "b.mp4").write_bytes(b"0000")
    env = {"FFPROBE_DISABLE": "1"}
    proc = run_cli(["meta", str(tmp_path), "--workers", "2"], env)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / ".artifacts" / "a.ffprobe.json").exists()
    assert (tmp_path / ".artifacts" / "b.ffprobe.json").exists()


def test_list_sort_show_size(tmp_path: Path):
    big = tmp_path / "big.mp4"
    small = tmp_path / "small.mp4"
    big.write_bytes(b"0" * 10)
    small.write_bytes(b"0" * 5)
    proc = run_cli(["list", str(tmp_path), "--show-size", "--sort", "size"])
    assert proc.returncode == 0
    lines = [l for l in proc.stdout.splitlines() if l.endswith(".mp4")]
    assert lines[0].startswith("big.mp4")
    assert "10.0B" in lines[0]
    assert lines[1].startswith("small.mp4")
    assert "5.0B" in lines[1]

