import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


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
    assert (tmp_path / "sample.mp4.ffprobe.json").exists()

    # thumb
    r_thumb = run_cli(["thumb", str(tmp_path)], env)
    assert r_thumb.returncode == 0, r_thumb.stderr
    assert (tmp_path / "sample.mp4.jpg").exists()

    # sprites
    r_sprites = run_cli(["sprites", str(tmp_path)], env)
    assert r_sprites.returncode == 0, r_sprites.stderr
    assert (tmp_path / "sample.mp4.sprites.jpg").exists()
    assert (tmp_path / "sample.mp4.sprites.json").exists()

    # previews
    r_prev = run_cli(["previews", str(tmp_path)], env)
    assert r_prev.returncode == 0, r_prev.stderr
    assert (tmp_path / "sample.mp4.previews.json").exists()

    # subs (default vtt)
    r_subs = run_cli(["subs", str(tmp_path)], env)
    assert r_subs.returncode == 0, r_subs.stderr
    assert (tmp_path / "sample.mp4.vtt").exists()

    # phash
    r_phash = run_cli(["phash", str(tmp_path), "--frames", "1", "--output-format", "json"], env)
    assert r_phash.returncode == 0, r_phash.stderr
    phash_file = tmp_path / "sample.mp4.phash.json"
    assert phash_file.exists()
    phash_data = json.loads(phash_file.read_text())
    assert "phash" in phash_data

    # heatmap
    r_heat = run_cli(["heatmap", str(tmp_path), "--output-format", "json"], env)
    assert r_heat.returncode == 0, r_heat.stderr
    assert (tmp_path / "sample.mp4.heatmap.json").exists()

    # scenes
    r_scenes = run_cli(["scenes", str(tmp_path), "--output-format", "json"], env)
    assert r_scenes.returncode == 0, r_scenes.stderr
    assert (tmp_path / "sample.mp4.scenes.json").exists()

    # report
    r_report = run_cli(["report", str(tmp_path), "--output-format", "json"], env)
    assert r_report.returncode == 0, r_report.stderr
    report_json = json.loads(r_report.stdout)
    # All artifacts should have count 1
    for key in ["metadata","thumb","sprites","previews","subs","phash","heatmap","scenes"]:
        assert report_json["counts"][key] == 1, f"{key} missing in report"
        assert report_json["coverage"][key] == 1.0

