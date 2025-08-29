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

    # metadata
    r_meta = run_cli(["metadata", str(tmp_path)], env)
    assert r_meta.returncode == 0, r_meta.stderr
    assert (tmp_path / ".artifacts" / "sample.ffprobe.json").exists()

    # thumbs
    r_thumbs = run_cli(["thumbs", str(tmp_path)], env)
    assert r_thumbs.returncode == 0, r_thumbs.stderr
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

    # subtitles (SRT only) in .artifacts
    r_subtitles = run_cli(["subtitles", str(tmp_path)], env)
    assert r_subtitles.returncode == 0, r_subtitles.stderr
    assert (tmp_path / ".artifacts" / "sample.srt").exists()

    # phash
    r_phash = run_cli(["phash", str(tmp_path), "--frames", "1", "--output-format", "json"], env)
    assert r_phash.returncode == 0, r_phash.stderr
    phash_file = tmp_path / ".artifacts" / "sample.phash.json"
    assert phash_file.exists()
    phash_data = json.loads(phash_file.read_text())
    assert "phash" in phash_data

    # heatmaps
    r_heat = run_cli(["heatmaps", str(tmp_path), "--output-format", "json"], env)
    assert r_heat.returncode == 0, r_heat.stderr
    assert (tmp_path / ".artifacts" / "sample.heatmaps.json").exists()

    # scenes
    r_scenes = run_cli(["scenes", str(tmp_path), "--output-format", "json"], env)
    assert r_scenes.returncode == 0, r_scenes.stderr
    assert (tmp_path / ".artifacts" / "sample.scenes.json").exists()

    # embed (per-video distinct signatures)
    r_embed = run_cli(["embed", str(tmp_path), "--output-format", "json"], env)
    assert r_embed.returncode == 0, r_embed.stderr
    # listing (global index)
    r_listing = run_cli(["listing", str(tmp_path), "--output-format", "json"], env)
    assert r_listing.returncode == 0, r_listing.stderr

    # report
    r_report = run_cli(["report", str(tmp_path), "--output-format", "json"], env)
    assert r_report.returncode == 0, r_report.stderr
    report_json = json.loads(r_report.stdout)
    # All artifacts should have count 1
    for key in ["metadata","thumbs","sprites","previews","subtitles","phash","heatmaps","scenes"]:
        assert report_json["counts"][key] == 1, f"{key} missing in report"
        assert report_json["coverage"][key] == 1.0
    # faces artifact not produced by embed/listing pipeline; expect 0
    assert report_json["counts"]["faces"] == 0


def test_dupes_stub(tmp_path: Path):
    # Two identical tiny mp4 files -> identical phash in stub mode likely
    env = {"FFPROBE_DISABLE": "1"}
    (tmp_path / "a.mp4").write_bytes(b"00")
    (tmp_path / "b.mp4").write_bytes(b"00")
    # generate phash artifacts
    r1 = run_cli(["phash", str(tmp_path), "--frames", "1"], env)
    assert r1.returncode == 0, r1.stderr
    r_dupes = run_cli(["dupes", str(tmp_path)], env)
    assert r_dupes.returncode == 0, r_dupes.stderr

def test_metadata_parallel_stub_mode(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"00")
    (tmp_path / "b.mp4").write_bytes(b"0000")
    env = {"FFPROBE_DISABLE": "1"}
    proc = run_cli(["metadata", str(tmp_path), "--workers", "2"], env)
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


def test_orphans_and_rename(tmp_path: Path):
    video = tmp_path / "old.mp4"
    video.write_bytes(b"00")
    art_dir = tmp_path / ".artifacts"
    art_dir.mkdir(exist_ok=True)
    (art_dir / "old.ffprobe.json").write_text("{}")
    proc = run_cli(["rename", str(video), str(tmp_path / "new.mp4")])
    assert proc.returncode == 0, proc.stderr
    assert not video.exists()
    assert (tmp_path / "new.mp4").exists()
    # Accept either renamed artifact or removal
    assert not (art_dir / "old.ffprobe.json").exists() or (art_dir / "new.ffprobe.json").exists()
    # Create an orphan artifact and list
    (art_dir / "ghost.ffprobe.json").write_text("{}")
    proc2 = run_cli(["orphans", str(tmp_path), "--output-format", "json"])
    assert proc2.returncode == 0
    # Structure check only (implementation-specific filtering)
    _ = json.loads(proc2.stdout)


def test_codecs_and_transcode_dry_run(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"00")
    (tmp_path / "b.mp4").write_bytes(b"00")
    env = {"FFPROBE_DISABLE": "1"}
    r_codecs = run_cli(["codecs", str(tmp_path), "--output-format", "json"], env)
    assert r_codecs.returncode == 0, r_codecs.stderr
    dest = tmp_path / "out"
    r_trans = run_cli(["transcode", str(tmp_path), str(dest), "--dry-run", "--force", "--output-format", "json"], env)
    assert r_trans.returncode == 0, r_trans.stderr
    assert not any(dest.glob("*.mp4"))


def test_tags_cli_basic(tmp_path: Path):
    # create video
    (tmp_path / "t.mp4").write_bytes(b"00")
    # add tags
    r_add = run_cli(["tags", str(tmp_path / "t.mp4"), "--set", "alpha,beta", "--output-format", "json"])
    assert r_add.returncode == 0, r_add.stderr
    # append
    r_add2 = run_cli(["tags", str(tmp_path / "t.mp4"), "--set", "beta,gamma", "--output-format", "json"])
    assert r_add2.returncode == 0, r_add2.stderr
    data = (tmp_path / ".artifacts" / "t.tags.json").read_text()
    assert "alpha" in data and "gamma" in data
    # remove
    r_rem = run_cli(["tags", str(tmp_path / "t.mp4"), "--remove", "alpha", "--output-format", "json"])
    assert r_rem.returncode == 0
    after = (tmp_path / ".artifacts" / "t.tags.json").read_text()
    assert "alpha" not in after
    # replace
    r_rep = run_cli(["tags", str(tmp_path / "t.mp4"), "--set", "solo", "--mode", "replace", "--output-format", "json"])
    assert r_rep.returncode == 0
    rep = (tmp_path / ".artifacts" / "t.tags.json").read_text()
    assert "solo" in rep and "beta" not in rep and "gamma" not in rep


def test_multi_command_parsing(tmp_path: Path):
    (tmp_path / "m.mp4").write_bytes(b"00")
    env = {"FFPROBE_DISABLE": "1"}
    proc = run_cli(["metadata,thumbs", str(tmp_path)], env)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / ".artifacts" / "m.ffprobe.json").exists()
    assert (tmp_path / ".artifacts" / "m.jpg").exists()

