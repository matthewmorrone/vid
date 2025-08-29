#!/usr/bin/env python3
"""Unified video utility script.

Functions:
  list   - List .mp4 files with optional recursion / JSON / sizes.
  metadata   - Generate ffprobe metadata JSON files for .mp4s.
           (Skips existing unless --force.)
  

The 'queue' command is an ephemeral in-memory scheduler to avoid CPU thrash on
low-power devices. It discovers files (like metadata) then processes them with a
thread pool (default 1 worker).

Metadata Output:
  For each video foo.mp4 we write foo.mp4.ffprobe.json

ffprobe Invocation:
  ffprobe -v quiet -print_format json -show_format -show_streams <file>

Testing / Offline Mode:
  Set FFPROBE_DISABLE=1 to skip invoking ffprobe and produce a stub JSON.

Exit Codes:
  0 success
  2 directory not found
  3 ffprobe missing or failed (non-fatal per-file; command exits 0 unless a
    hard error before processing begins)

"""
from __future__ import annotations
import argparse
import math
import json
import os
import sys
import threading
import time
import subprocess
import random
from pathlib import Path
from typing import List, Dict, Any, Iterable
from types import SimpleNamespace
from queue import Queue, Empty
try:  # Optional dependency; many commands work without OpenCV
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback when opencv-python not installed
    cv2 = None
import numpy as np
import warnings

# Focused deprecation suppression: limit to known noisy modules (pkg_resources / deprecated distutils in deps)
with warnings.catch_warnings():  # pragma: no cover
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"distutils")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# File discovery & listing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Merged helpers (from faces.py & find_misnamed_assets.py)
# ---------------------------------------------------------------------------

def cluster_faces(
    videos: Iterable[Path],
    frame_rate: float = 1.0,
    eps: float = 0.5,
    min_samples: int = 2,
):
    """Detect faces in videos and cluster them by similarity.

    Returns dict[label] -> list occurrences. If heavy dependencies are missing,
    raises a RuntimeError so caller can decide how to degrade.
    """
    try:
        import cv2  # type: ignore
        import face_recognition  # type: ignore
        from sklearn.cluster import DBSCAN  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("face clustering requires cv2, face_recognition and scikit-learn") from exc
    embeddings: list[np.ndarray] = []
    occurrences: list[dict] = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps / frame_rate)), 1)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb = frame[:, :, ::-1]
                boxes = face_recognition.face_locations(rgb)
                if boxes:
                    encs = face_recognition.face_encodings(rgb, boxes)
                    ts = frame_idx / fps
                    for (top, right, bottom, left), enc in zip(boxes, encs):
                        embeddings.append(enc)
                        occurrences.append({
                            "video": Path(video).name,
                            "timestamp": ts,
                            "bbox": {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)},
                        })
            frame_idx += 1
        cap.release()
    if not embeddings:
        return {}
    data = np.vstack(embeddings).astype("float32")
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = data / norms
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(normalized)
    clusters: dict[int, list[dict]] = {}
    for lab, occ in zip(labels, occurrences):
        clusters.setdefault(int(lab), []).append(occ)
    return clusters


def extract_distinct_face_signatures(
    video: Path,
    sample_rate: float = 1.0,
    min_face_area: int = 0,
    blur_threshold: float = 0.0,
    max_gap: float = 0.5,
    min_track_frames: int = 2,
    match_distance: float = 0.6,
    cluster_eps: float = 0.45,
    cluster_min_samples: int = 1,
) -> list[dict]:
    """Return representative face signature tracks for a single video.

    Simplified lightweight implementation: detects faces on sampled frames,
    obtains embeddings (if heavy deps present) and clusters them with DBSCAN.
    In test/stub mode (FFPROBE_DISABLE) or missing deps returns empty list.
    """
    if os.environ.get("FFPROBE_DISABLE"):
        return []
    try:
        import cv2  # type: ignore
        import face_recognition  # type: ignore
        from sklearn.cluster import DBSCAN  # type: ignore
    except Exception:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / max(sample_rate, 0.1))), 1)
    frame_idx = 0
    embeddings: list[np.ndarray] = []
    times: list[float] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = frame[:, :, ::-1]
            boxes = face_recognition.face_locations(rgb)
            if boxes:
                encs = face_recognition.face_encodings(rgb, boxes)
                ts = frame_idx / fps
                for enc in encs:
                    embeddings.append(enc)
                    times.append(ts)
        frame_idx += 1
    cap.release()
    if not embeddings:
        return []
    arr = np.vstack(embeddings).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    try:
        labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric="euclidean").fit_predict(arr)
    except Exception:
        labels = np.arange(len(arr))
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    results: list[dict] = []
    for lab, idxs in groups.items():
        seg = arr[idxs]
        rep = seg.mean(axis=0)
        rep = rep / (np.linalg.norm(rep) or 1.0)
        start_t = min(times[i] for i in idxs)
        end_t = max(times[i] for i in idxs)
        frames = len(idxs)
        results.append({
            "id": f"p{lab}" if lab >= 0 else f"n{abs(lab)}",
            "start_t": start_t,
            "end_t": end_t,
            "frames": frames,
            "embedding": [float(x) for x in rep.tolist()],
        })
    return results


def build_face_index(
    root: Path,
    recursive: bool = False,
    sample_rate: float = 1.0,
    **kwargs,
) -> dict:
    """Iterate videos under root and build a deduplicated face index.

    Returns {"people": [{"id", "embedding", "occurrences": [{video,start_t,end_t,frames}]}], "videos": count}.
    Uses extract_distinct_face_signatures per video then clusters globally.
    """
    videos = []
    if recursive:
        for p in root.rglob("*.mp4"):
            videos.append(p)
    else:
        videos = [p for p in root.glob("*.mp4") if p.is_file()]
    all_entries: list[dict] = []
    for v in videos:
        sigs = extract_distinct_face_signatures(v, sample_rate=sample_rate, **kwargs)
        for s in sigs:
            s2 = dict(s)
            s2["video"] = v.name
            all_entries.append(s2)
    if not all_entries:
        return {"people": [], "videos": len(videos)}
    # Global clustering of embeddings to dedupe across videos
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        embs = np.vstack([np.array(e["embedding"], dtype="float32") for e in all_entries])
        labels = DBSCAN(eps=kwargs.get("cluster_eps", 0.45), min_samples=kwargs.get("cluster_min_samples", 1), metric="euclidean").fit_predict(embs)
        grouped: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            grouped.setdefault(int(lab), []).append(idx)
        people = []
        for lab, idxs in grouped.items():
            arr = embs[idxs]
            rep = arr.mean(axis=0)
            rep = rep / (np.linalg.norm(rep) or 1.0)
            occs = []
            for i in idxs:
                e = all_entries[i]
                occs.append({
                    "video": e["video"],
                    "start_t": e["start_t"],
                    "end_t": e["end_t"],
                    "frames": e["frames"],
                })
            people.append({
                "id": f"g{lab}" if lab >= 0 else f"gneg{abs(lab)}",
                "embedding": [float(x) for x in rep.tolist()],
                "occurrences": occs,
            })
        return {"people": people, "videos": len(videos)}
    except Exception:
        # Fallback: no global clustering
        people = []
        for idx, e in enumerate(all_entries):
            people.append({
                "id": f"u{idx}",
                "embedding": e["embedding"],
                "occurrences": [{"video": e["video"], "start_t": e["start_t"], "end_t": e["end_t"], "frames": e["frames"]}],
            })
        return {"people": people, "videos": len(videos)}

fMEDIA_EXTS = {".mp4"}

ARTIFACTS_DIR = ".artifacts"

def artifact_dir(media_path: Path) -> Path:
    d = media_path.parent / ARTIFACTS_DIR
    d.mkdir(exist_ok=True, parents=True)
    return d

def subtitles_path_candidates(media_path: Path):
    # Store SRT in artifact directory as <stem>.srt (e.g. sample.srt for sample.mp4)
    d = artifact_dir(media_path)
    return [d / f"{media_path.stem}.srt"]

def find_subtitles(media_path: Path):
    for p in subtitles_path_candidates(media_path):
        if p.exists():
            return p
    return None

def rename_with_artifacts(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    src_dir = src.parent / ARTIFACTS_DIR
    if not src_dir.exists():
        return
    dst_dir = dst.parent / ARTIFACTS_DIR
    dst_dir.mkdir(exist_ok=True)
    for p in src_dir.iterdir():
        if p.name.startswith(src.stem):
            new_name = dst.stem + p.name[len(src.stem):]
            os.replace(p, dst_dir / new_name)

def find_videos(root: Path, recursive: bool = False, exts: set[str] | None = None) -> List[Path]:
    exts = exts or {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    iterator = root.rglob("*") if recursive else root.iterdir()
    vids = []
    for p in iterator:
        # Exclude files that are immediate children of ARTIFACTS_DIR
        if (
            p.is_file()
            and p.suffix.lower() in exts
            and p.parent != ARTIFACTS_DIR
        ):
            vids.append(p)
    vids.sort()
    return vids

def find_mp4s(root: Path, recursive: bool = False) -> List[Path]:
    return find_videos(root, recursive, {".mp4"})


def human_size(num: float) -> str:
    if num < 0:
        return "?"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def build_record(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        size = -1
        mtime = 0
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": size,
        "mtime": mtime,
    }

# ---------------------------------------------------------------------------
# ffprobe metadata generation
# ---------------------------------------------------------------------------

def metadata_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.ffprobe.json"

def thumbs_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.jpg"


def ffprobe_available() -> bool:
    if os.environ.get("FFPROBE_DISABLE"):
        return True  # pretend ok in stub mode
    from shutil import which
    return which("ffprobe") is not None


def run_ffprobe(video: Path) -> Dict[str, Any]:
    if os.environ.get("FFPROBE_DISABLE"):
        # Stub result for testing / environments without ffprobe
        return {
            "stub": True,
            "path": str(video),
            "size": video.stat().st_size if video.exists() else None,
            "note": "FFPROBE_DISABLE set; real ffprobe skipped",
        }
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed: {video}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid ffprobe JSON for {video}: {e}") from e


def generate_metadata(videos: Iterable[Path], force: bool = False, workers: int = 1) -> Dict[str, Any]:
    videos = list(videos)
    total = len(videos)
    done = 0  # number of videos examined (generated or skipped)
    generated = 0  # number of metadata JSON files actually (re)written this run
    errors: list[str] = []

    # Worker design: queue file paths, each worker writes file.
    q: Queue[Path] = Queue()
    # Pre-scan to count existing artifacts to provide summary line
    existing = 0
    for v in videos:
        if (not force) and metadata_path(v).exists():
            existing += 1
        q.put(v)

    remaining = total - existing if not force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"metadata: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force to regenerate).", file=sys.stderr)
    else:
        print(f"metadata: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)

    lock = threading.Lock()

    def worker():
        nonlocal done
        nonlocal generated
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            try:
                out_path = metadata_path(v)
                wrote = False
                if out_path.exists() and not force:
                    # skip
                    pass
                else:
                    data = run_ffprobe(v)
                    out_path.write_text(json.dumps(data, indent=2))
                    wrote = True
                if wrote:
                    with lock:
                        generated += 1
                        # Show cumulative covered (existing + generated) over total
                        covered = existing + generated if not force else generated
                        if covered > total:
                            covered = total
                        print(f"metadata gen {covered}/{total} {v.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{v}: {e}")
            finally:
                with lock:
                    done += 1
                q.task_done()

    worker_count = max(1, min(workers, 4))  # cap to protect low-power devices
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    # Simple progress loop
    # Only show a rolling progress line if work remains
    if remaining > 0:
        while any(t.is_alive() for t in threads):
            with lock:
                covered = existing + generated if not force else generated
                if covered > total:
                    covered = total
                print(f"metadata scan {covered}/{total} (generated {generated})\r", end="", file=sys.stderr)
            time.sleep(0.25)
    for t in threads:
        t.join()
    if remaining > 0:
        covered = existing + generated if not force else generated
        if covered > total:
            covered = total
        print(f"metadata scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if not errors:
        print(f"metadata: generated {generated} file(s), skipped {total - generated} (existing)", file=sys.stderr)
    return {"total": total, "errors": errors}

 

# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def ffmpeg_available() -> bool:
    if os.environ.get("FFPROBE_DISABLE"):  # allow same toggle to stub
        return True
    from shutil import which
    return which("ffmpeg") is not None


def parse_time_spec(time_spec: str, duration: float | None) -> float:
    if time_spec == "middle":
        return (duration / 2.0) if duration and duration > 0 else 1.0
    if time_spec.endswith("%"):
        pct = float(time_spec[:-1]) / 100.0
        if duration and duration > 0:
            return max(0.0, min(duration - 0.5, duration * pct))
        return 1.0
    try:
        return float(time_spec)
    except ValueError:
        return 1.0


def extract_duration(metadata_json: Dict[str, Any]) -> float | None:
    # ffprobe format.duration is a string
    fmt = metadata_json.get("format") if isinstance(metadata_json, dict) else None
    if isinstance(fmt, dict):
        dur = fmt.get("duration")
        try:
            if dur is not None:
                return float(dur)
            else:
                return None
        except (TypeError, ValueError):
            return None
    return None


def generate_thumbnail(video: Path, force: bool, time_spec: str, quality: int) -> bool:
    """Generate thumbnail; return True if (re)generated, False if skipped."""
    out = thumbs_path(video)
    if out.exists() and not force:
        return False
    duration = None
    metadata_file = metadata_path(video)
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text())
            duration = extract_duration(metadata)
        except Exception:  # noqa: BLE001
            duration = None
    t = parse_time_spec(time_spec, duration)
    if os.environ.get("FFPROBE_DISABLE"):
        # stub mode writes a placeholder
        out.write_text(f"stub thumbnail for {video.name} at {t}s")
        return True
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not available")
    # Use -ss (seek) then -frames:v 1 to capture a single frame
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(t),
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", str(quality),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg thumbnail failed")
    return True


def cmd_thumbs(ns) -> int:
    """Generate a single thumbnail per video (time chosen by --time) with skip-aware progress."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    total = len(videos)
    # Pre-scan existing
    existing = 0
    q: Queue[Path] = Queue()
    for v in videos:
        if (not ns.force) and thumbs_path(v).exists():
            existing += 1
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"thumbs: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force to regenerate).", file=sys.stderr)
        print(f"Thumbs generated for 0 file(s)")
        return 0
    print(f"thumbs: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    errors: list[str] = []
    processed = 0  # videos pulled from queue
    generated = 0  # new thumbnails written
    lock = threading.Lock()

    def worker():
        nonlocal processed, generated
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            try:
                created = generate_thumbnail(v, ns.force, ns.time_spec, ns.quality)
                if created:
                    with lock:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total:
                            covered = total
                        print(f"thumbs gen {covered}/{total} {v.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{v}: {e}")
            finally:
                with lock:
                    processed += 1
                q.task_done()

    worker_count = max(1, min(ns.workers, 4))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        with lock:
            covered = existing + generated if not ns.force else generated
            if covered > total:
                covered = total
            print(f"thumbs scan {covered}/{total} (generated {generated})\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total:
            covered = total
    print(f"thumbs scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if errors:
        print("Errors (some thumbs missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Thumbs generated for {generated} file(s)")
    return 0

# ---------------------------------------------------------------------------
# Sprite sheet generation
# ---------------------------------------------------------------------------

def sprite_sheet_paths(video: Path) -> tuple[Path, Path]:
    base = artifact_dir(video) / f"{video.stem}.sprites"
    return Path(str(base) + ".jpg"), Path(str(base) + ".json")

def build_sprite_ffmpeg_cmd(video: Path, tmp_pattern: Path, interval: float, width: int, cols: int, rows: int, quality: int) -> list[str]:
    # Use fps filter as 1/interval to sample frames. Scale width, preserve aspect, then tile.
    # Limit frames to cols*rows using -frames:v
    max_frames = cols * rows
    vf = f"fps=1/{interval},scale={width}:-1:force_original_aspect_ratio=decrease"  # scaling
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", vf,
        "-frames:v", str(max_frames),
        "-q:v", str(quality),
        str(tmp_pattern),
    ]


def assemble_tile_cmd(tmp_pattern: Path, sheet_out: Path, cols: int, rows: int, quality: int) -> list[str]:
    tile_filter = f"tile={cols}x{rows}"  # will pad with black if insufficient frames
    return [
        "ffmpeg", "-y",
        "-pattern_type", "glob",
        "-i", str(tmp_pattern),
        "-vf", tile_filter,
        "-q:v", str(quality),
        str(sheet_out),
    ]


def generate_sprite_sheet(video: Path, interval: float, width: int, cols: int, rows: int, quality: int, force: bool, max_frames: int) -> tuple[bool, str | None]:
    sheet_path, json_path = sprite_sheet_paths(video)
    if sheet_path.exists() and json_path.exists() and not force:
        return True, None
    if os.environ.get("FFPROBE_DISABLE"):
        # stub: create simple json + empty file
        sheet_path.write_text("stub")
        json_path.write_text(json.dumps({"stub": True, "interval": interval}))
        return True, None
    if not ffmpeg_available():
        return False, "ffmpeg not available"
    # Temporary frames dir
    frames_dir = video.parent / f".{video.name}.frames_tmp"
    frames_dir.mkdir(exist_ok=True)
    pattern = frames_dir / "*.jpg"
    # Extract frames
    cmd_extract = build_sprite_ffmpeg_cmd(video, frames_dir / "%04d.jpg", interval, width, cols, rows, quality)
    proc = subprocess.run(cmd_extract, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "ffmpeg extract failed"
    # Optionally cap frames if user provided smaller max
    if max_frames > 0 and max_frames < cols * rows:
        # We will re-tile only first max_frames frames by renaming extras
        frames = sorted(frames_dir.glob("*.jpg"))
        for f in frames[max_frames:]:
            f.unlink(missing_ok=True)
        cols_rows = min(cols * rows, max_frames)
    else:
        cols_rows = cols * rows
    # Build sheet
    tile_cmd = [
        "ffmpeg", "-y",
        "-pattern_type", "glob",
        "-i", str(pattern),
        "-vf", f"tile={cols}x{rows}",
        "-q:v", str(quality),
        str(sheet_path),
    ]
    proc2 = subprocess.run(tile_cmd, capture_output=True, text=True)
    if proc2.returncode != 0:
        return False, proc2.stderr.strip() or "ffmpeg tile failed"
    # Get tile size from first frame (approx width/height per frame after scaling)
    first = next(iter(frames_dir.glob("*.jpg")), None)
    tile_w = width
    tile_h = None
    if first and ffprobe_available() and not os.environ.get("FFPROBE_DISABLE"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(first)], capture_output=True, text=True)
        if probe.returncode == 0:
            try:
                info = json.loads(probe.stdout)
                if info.get("streams"):
                    s0 = info["streams"][0]
                    tile_w = s0.get("width", tile_w)
                    tile_h = s0.get("height", tile_h)
            except Exception:
                pass
    # Count actual frames
    actual_frames = len(list(frames_dir.glob("*.jpg")))
    # Write JSON index
    json_path.write_text(json.dumps({
        "video": str(video.name),
        "sheet": sheet_path.name,
        "interval": interval,
        "cols": cols,
        "rows": rows,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "frames": actual_frames,
        "max_frames": cols * rows,
    }, indent=2))
    # Clean up frames directory
    for f in frames_dir.glob("*.jpg"):
        f.unlink(missing_ok=True)
    frames_dir.rmdir()
    return True, None


def cmd_sprites(ns) -> int:
    """Create sprite sheet + JSON index (grid of frames) for each video (skip-aware)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    interval = max(0.1, ns.interval)
    cols = max(1, ns.cols)
    rows = max(1, ns.rows)
    total = len(videos)
    existing = 0
    to_process: list[Path] = []
    if ns.force:
        to_process = videos[:]
    else:
        for v in videos:
            s, j = sprite_sheet_paths(v)
            if s.exists() and j.exists():
                existing += 1
            else:
                to_process.append(v)
    remaining = len(to_process)
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"sprites: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        print("Sprite sheets generated for 0 file(s)")
        return 0
    print(f"sprites: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    generated = 0
    processed = 0
    errors: list[str] = []
    for v in to_process:
        ok, err = generate_sprite_sheet(v, interval, ns.width, cols, rows, ns.quality, True, ns.max_frames)
        processed += 1
        if not ok and err:
            errors.append(f"{v}: {err}")
        else:
            generated += 1
            covered = existing + generated if not ns.force else generated
            if covered > total:
                covered = total
            print(f"sprites gen {covered}/{total} {v.name}", file=sys.stderr)
        print(f"sprites scan {existing + generated}/{total} (generated {generated})\r", end="", file=sys.stderr)
    print(f"sprites scan {existing + generated}/{total} (generated {generated})", file=sys.stderr)
    if errors:
        print("Errors (some sprite sheets missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Sprite sheets generated for {generated} file(s)")
    return 0 if not errors else 1

# ---------------------------------------------------------------------------
# Hover preview short clips (decile segments)
# ---------------------------------------------------------------------------

def preview_output_dir(video: Path) -> Path:
    d = artifact_dir(video) / f"{video.stem}.previews"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preview_index_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.previews.json"


def extract_duration_from_file(video: Path) -> float | None:
    if os.environ.get("FFPROBE_DISABLE"):
        return 100.0  # stub default
    if not ffprobe_available():
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def build_preview_cmd(video: Path, start: float, duration: float, width: int, out_file: Path, fmt: str, crf: int, bitrate: str) -> list[str]:
    vfilter = f"scale={width}:-1:force_original_aspect_ratio=decrease"
    base = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}"]
    if fmt == "mp4":
        return base + [
            "-vf", vfilter,
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(out_file),
        ]
    # webm (vp9)
    return base + [
        "-vf", vfilter,
        "-an",
        "-c:v", "libvpx-vp9",
        "-b:v", bitrate,
        "-crf", str(crf),
        str(out_file),
    ]


def cmd_previews(ns) -> int:
    """Generate short hover preview clips (segmented) and an index JSON (skip-aware)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    if os.environ.get("FFPROBE_DISABLE"):
        print("FFPROBE_DISABLE set - using stub durations", file=sys.stderr)
    elif not ffprobe_available():
        print("Warning: ffprobe not available; durations may be missing", file=sys.stderr)
    worker_count = max(1, min(ns.workers, 4))
    total = len(videos)
    existing = 0
    q: Queue[Path] = Queue()
    for v in videos:
        if (not ns.force):
            idx = preview_index_path(v)
            if idx.exists():
                try:
                    data = json.loads(idx.read_text())
                    segs_meta = data.get("segments", [])
                    clip_dir = preview_output_dir(v)
                    # If any referenced clip missing, treat as incomplete => regenerate
                    all_present = True
                    for s in segs_meta:
                        fname = s.get("file")
                        if fname and not (clip_dir / fname).exists():
                            all_present = False
                            break
                    if all_present:
                        existing += 1
                except Exception:
                    pass  # corrupted index -> regenerate
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"previews: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        print("Hover previews generated for 0 file(s)")
        return 0
    print(f"previews: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    processed = 0
    generated = 0
    lock = threading.Lock()
    errors: list[str] = []

    def worker():
        nonlocal processed, generated
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                duration = extract_duration_from_file(vid) or 0
                # Decile segments: by default segments=9 => 10%,20%,...,90%
                segs = max(1, ns.segments)
                points = [((i + 1) / 10.0) * duration for i in range(segs)] if duration > 0 else [i * ns.duration for i in range(segs)]
                out_dir = preview_output_dir(vid)
                out_dir.mkdir(exist_ok=True)
                clip_ext = ".mp4" if ns.format == "mp4" else ".webm"
                index_entries = []
                new_segments = False
                for idx, start in enumerate(points, start=1):
                    out_file = out_dir / f"seg_{idx:02d}{clip_ext}"
                    if out_file.exists() and not ns.force:
                        index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                        continue
                    if os.environ.get("FFPROBE_DISABLE"):
                        out_file.write_text(f"stub preview {vid.name} t={start:.2f}")
                        index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                        new_segments = True
                        continue
                    if not ffmpeg_available():
                        raise RuntimeError("ffmpeg not available")
                    cmd = build_preview_cmd(vid, start, ns.duration, ns.width, out_file, ns.format, ns.crf, ns.bitrate)
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr.strip() or "ffmpeg preview failed")
                    index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                    new_segments = True
                wrote_any = False
                if not ns.no_index:
                    idx_path = preview_index_path(vid)
                    if (not idx_path.exists()) or ns.force:
                        idx_path.write_text(json.dumps({
                            "video": vid.name,
                            "format": ns.format,
                            "segments": index_entries,
                            "segment_duration": ns.duration,
                            "generated_at": time.time(),
                        }, indent=2))
                        wrote_any = True
                if wrote_any or (ns.no_index and new_segments):
                    with lock:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total: covered = total
                        print(f"previews gen {covered}/{total} {vid.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid}: {e}")
            finally:
                with lock:
                    processed += 1
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        with lock:
            covered = existing + generated if not ns.force else generated
            if covered > total: covered = total
            print(f"previews scan {covered}/{total} (generated {generated})\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total: covered = total
    print(f"previews scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if errors:
        print("Errors (some previews missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Hover previews generated for {generated} file(s)")
    if ns.output_format == "json":
        results = []
        for v in videos:
            idx = preview_index_path(v)
            seg_count = 0
            has_index = idx.exists()
            if has_index:
                try:
                    data = json.loads(idx.read_text())
                    seg_count = len(data.get("segments", []))
                except Exception:
                    has_index = False
            results.append({"video": v.name, "has_index": has_index, "segments": seg_count})
        print(json.dumps({"results": results}, indent=2))
    return 0

# ---------------------------------------------------------------------------
# Subtitle generation (Whisper family)
# ---------------------------------------------------------------------------

def detect_backend(preference: str) -> str:
    if preference != "auto":
        return preference
    # Try faster-whisper first (performance), then whisper, then whisper.cpp
    try:
        import importlib  # noqa: F401
        __import__("faster_whisper")
        return "faster-whisper"
    except Exception:
        pass
    try:
        __import__("whisper")
        return "whisper"
    except Exception:
        pass
    return "whisper.cpp"  # assume external binary


def format_segments(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, s in enumerate(segments, start=1):
        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = t % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')
        lines.append(str(i))
        lines.append(f"{ts(s['start'])} --> {ts(s['end'])}")
        lines.append(s['text'].strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_whisper_backend(video: Path, backend: str, model_name: str, language: str | None, translate: bool, cpp_bin: str | None, cpp_model: str | None, compute_type: str | None = None):
    # Returns list of segments: {start,end,text}
    # compute_type: for faster-whisper (auto|int8|int8_float16|int16|float16|float32). We'll attempt fallback if unsupported.
    if os.environ.get("FFPROBE_DISABLE"):
        return [{"start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"Stub segment {i+1}"} for i in range(3)]
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel  # type: ignore
        # Choose initial compute type: explicit flag > translate heuristic > default float16
        initial_type = compute_type or ("int8" if translate else "float16")
        tried: list[str] = []
        fallback_chain = [initial_type]
        # If user forced a type we still may fallback if it errors.
        if initial_type != "int8":
            fallback_chain.append("int8")
        if initial_type not in ("float32", "int16"):
            fallback_chain.append("float32")  # last resort correctness over speed
        last_err: Exception | None = None
        for ct in fallback_chain:
            try:
                tried.append(ct)
                model = WhisperModel(model_name, compute_type=ct)
                seg_iter, _info = model.transcribe(str(video), language=language, task="translate" if translate else "transcribe")
                segments = []
                for s in seg_iter:
                    segments.append({"start": s.start, "end": s.end, "text": s.text})
                if ct != initial_type:
                    print(f"[subtitles] compute_type fallback: {initial_type} -> {ct}", file=sys.stderr)
                return segments
            except Exception as e:  # noqa: BLE001
                last_err = e
                # Try next fallback
                continue
        # All attempts failed
        raise RuntimeError(f"faster-whisper failed (tried {tried}): {last_err}")
    if backend == "whisper":
        import whisper  # type: ignore
        model = whisper.load_model(model_name)
        result = model.transcribe(str(video), language=language, task="translate" if translate else "transcribe")
        segs = []
        for s in result.get("segments", []):
            if isinstance(s, dict):
                segs.append({"start": s.get("start"), "end": s.get("end"), "text": s.get("text")})
        return segs
    if backend == "whisper.cpp":
        # Need external binary invocation; minimal implementation.
        if not cpp_bin or not Path(cpp_bin).exists():
            raise RuntimeError("whisper.cpp binary not found (provide --whisper-cpp-bin)")
        if not cpp_model or not Path(cpp_model).exists():
            raise RuntimeError("whisper.cpp model not found (provide --whisper-cpp-model)")
        out_json = artifact_dir(video) / f"{video.stem}.{backend}.json"
        cmd = [
            cpp_bin,
            "-m", cpp_model,
            "-f", str(video),
            "-otxt",
            "-oj",  # json output
        ]
        if language:
            cmd += ["-l", language]
        if translate:
            cmd += ["-tr"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "whisper.cpp failed")
        # whisper.cpp with -oj writes JSON file alongside or to stdout? We'll parse stdout if JSON-looking else fallback.
        try:
            data = json.loads(proc.stdout)
            segs = []
            for s in data.get("transcription", {}).get("segments", []):
                segs.append({"start": s.get("t0", 0)/1000.0, "end": s.get("t1", 0)/1000.0, "text": s.get("text", "")})
            return segs
        except Exception:
            if out_json.exists():
                data = json.loads(out_json.read_text())
                segs = []
                for s in data.get("transcription", {}).get("segments", []):
                    segs.append({"start": s.get("t0", 0)/1000.0, "end": s.get("t1", 0)/1000.0, "text": s.get("text", "")})
                return segs
            raise RuntimeError("Failed to parse whisper.cpp output")
    raise RuntimeError(f"Unknown backend {backend}")


def cmd_subtitles(ns) -> int:
    """Generate SRT subtitles only (skip-aware; counts existing)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    backend = detect_backend(ns.backend)
    worker_count = max(1, min(ns.workers, 2))
    out_dir_base = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else None
    total = len(videos)
    existing = 0
    q: Queue[Path] = Queue()
    for v in videos:
        out_dir = out_dir_base or artifact_dir(v)
        out_file = out_dir / f"{v.stem}.srt"
        if (not ns.force) and out_file.exists():
            existing += 1
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"subtitles: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        print("Subtitles generated for 0 file(s)")
        return 0
    print(f"subtitles: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    processed = 0
    generated = 0
    errors: list[str] = []
    lock = threading.Lock()

    def worker():
        nonlocal processed, generated
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                out_dir = out_dir_base or artifact_dir(vid)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{vid.stem}.srt"
                if out_file.exists() and not ns.force:
                    wrote = False
                else:
                    segments = run_whisper_backend(vid, backend, ns.model, ns.language, ns.translate, ns.whisper_cpp_bin, ns.whisper_cpp_model, getattr(ns, "compute_type", None))
                    srt_text = format_segments(segments)
                    out_file.write_text(srt_text)
                    wrote = True
                if wrote:
                    with lock:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total: covered = total
                        print(f"subtitles gen {covered}/{total} {vid.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid}: {e}")
            finally:
                with lock:
                    processed += 1
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        with lock:
            covered = existing + generated if not ns.force else generated
            if covered > total: covered = total
            print(f"subtitles scan {covered}/{total} (generated {generated}) backend={backend}\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total: covered = total
    print(f"subtitles scan {covered}/{total} (generated {generated}) backend={backend}", file=sys.stderr)
    if errors:
        print("Errors (some subtitles missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Subtitles generated for {generated} file(s)")
    return 0

# ---------------------------------------------------------------------------
# Ephemeral multi-task batch scheduler
# ---------------------------------------------------------------------------

class BatchJob:
    __slots__ = ("video", "task")
    def __init__(self, video: Path, task: str):
        self.video = video
        self.task = task  # metadata, thumbs, sprites, previews, subtitles


def artifact_exists(video: Path, task: str) -> bool:
    if task == "metadata":
        return metadata_path(video).exists()
    if task == "thumbs":
        return thumbs_path(video).exists()
    if task == "sprites":
        s, j = sprite_sheet_paths(video)
        return s.exists() and j.exists()
    if task == "previews":
        return preview_index_path(video).exists() or preview_output_dir(video).exists()
    if task == "subtitles":
        return (artifact_dir(video) / f"{video.stem}.srt").exists()
    if task == "phash":
        return phash_path(video).exists()
    if task == "heatmaps":
        return heatmaps_json_path(video).exists()
    if task == "scenes":
        return scenes_json_path(video).exists()
    if task == "faces":
        return faces_json_path(video).exists()
    return False


def run_task(video: Path, task: str, ns) -> tuple[bool, str | None]:
    try:
        if task == "metadata":
            if not meta_single(video, force=ns.force):
                return True, None
        elif task == "thumbs":
            generate_thumbnail(video, ns.force, "middle", 2)
        elif task == "sprites":
            ok, err = generate_sprite_sheet(video, ns.sprites_interval, ns.sprites_width, ns.sprites_cols, ns.sprites_rows, 4, ns.force, 0)
            if not ok:
                return False, err
        elif task == "previews":
            # simplified single-video preview invocation
            class _Tmp:  # minimal namespace simulation
                duration = ns.preview_duration
                segments = ns.preview_segments
                width = ns.preview_width
                format = "webm"
                crf = 30
                bitrate = "300k"
                force = ns.force
                no_index = False
            # reuse code by faking loop (direct call logic lifted from cmd_previews)
            duration = extract_duration_from_file(video) or 0
            segs = max(1, _Tmp.segments)
            points = [((i + 1) / 10.0) * duration for i in range(segs)] if duration > 0 else [i * _Tmp.duration for i in range(segs)]
            out_dir = preview_output_dir(video)
            out_dir.mkdir(exist_ok=True)
            clip_ext = ".webm"
            index_entries = []
            for idx, start in enumerate(points, start=1):
                out_file = out_dir / f"seg_{idx:02d}{clip_ext}"
                if out_file.exists() and not _Tmp.force:
                    index_entries.append({"start": start, "file": out_file.name, "duration": _Tmp.duration})
                    continue
                if os.environ.get("FFPROBE_DISABLE"):
                    out_file.write_text(f"stub preview {video.name} t={start:.2f}")
                    index_entries.append({"start": start, "file": out_file.name, "duration": _Tmp.duration})
                    continue
                if not ffmpeg_available():
                    raise RuntimeError("ffmpeg not available")
                cmd = build_preview_cmd(video, start, _Tmp.duration, _Tmp.width, out_file, _Tmp.format, _Tmp.crf, _Tmp.bitrate)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "ffmpeg preview failed")
                index_entries.append({"start": start, "file": out_file.name, "duration": _Tmp.duration})
            preview_index_path(video).write_text(json.dumps({
                "video": video.name,
                "format": _Tmp.format,
                "segments": index_entries,
                "segment_duration": _Tmp.duration,
                "generated_at": time.time(),
            }, indent=2))
        elif task == "subtitles":
            backend = detect_backend(ns.subtitles_backend)
            segments = run_whisper_backend(video, backend, ns.subtitles_model, ns.subtitles_language, ns.subtitles_translate, None, None, getattr(ns, "compute_type", None))
            text = format_segments(segments)
            artifact_dir(video).mkdir(exist_ok=True)
            (artifact_dir(video) / f"{video.stem}.srt").write_text(text)
        elif task == "phash":
            # Use 5 evenly spaced frames by default in batch mode for more robust whole-video signature (algo defaults)
            compute_phash_video(video, frame_time_spec="middle", frames=5, force=ns.force, algo="ahash", combine="xor")
        elif task == "scenes":
            # Default: detect scenes only (no thumbs/clips) for speed
            generate_scene_artifacts(video, threshold=0.4, limit=0, gen_thumbs=False, gen_clips=False, thumbs_width=320, clip_duration=2.0, force=ns.force)
        else:
            return False, f"Unknown task {task}"
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def meta_single(video: Path, force: bool) -> bool:
    out = metadata_path(video)
    if out.exists() and not force:
        return False
    try:
        data = run_ffprobe(video)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"ffprobe failed: {e}")
    out.write_text(json.dumps(data, indent=2))
    return True


def cmd_batch(ns) -> int:
    """Run a batch of artifact generators (metadata, thumbs, sprites, previews, subtitles) together."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    requested = [t.strip() for t in ns.tasks.split(',') if t.strip()]
    valid = {"metadata", "thumbs", "sprites", "previews", "subtitles", "phash", "scenes"}
    for t in requested:
        if t not in valid:
            print(f"Unknown task in --tasks: {t}", file=sys.stderr)
            return 2
    per_type_caps = {
        "metadata": max(1, ns.max_meta),
        "thumbs": max(1, ns.max_thumbs),
        "sprites": max(1, ns.max_sprites),
        "previews": max(1, ns.max_previews),
        "subtitles": max(1, ns.max_subtitles),
        "phash": max(1, ns.max_phash),
        "scenes": max(1, ns.max_scenes),
    }

    # Build job list in stage order: stage-by-stage to respect dependencies (e.g., sprites prefer metadata but not strictly required).
    jobs: list[BatchJob] = []
    for task in requested:
        for v in videos:
            if not ns.force and artifact_exists(v, task):
                continue
            jobs.append(BatchJob(v, task))

    if not jobs:
        print("Nothing to do (artifacts exist). Use --force to regenerate.")
        return 0

    lock = threading.Lock()
    running_counts: Dict[str, int] = {k: 0 for k in per_type_caps}
    pending = jobs[:]  # simple list acts as queue
    completed = 0
    failed: list[str] = []
    total = len(pending)
    stop = False

    def worker_loop():
        nonlocal completed, stop
        while True:
            with lock:
                if stop:
                    return
                # pick a job whose type has capacity
                pick_index = None
                for i, j in enumerate(pending):
                    cap = per_type_caps[j.task]
                    if running_counts[j.task] < cap:
                        pick_index = i
                        running_counts[j.task] += 1
                        job = j
                        break
                if pick_index is None:
                    # no job fits now; check if done
                    if not pending:
                        return
                    # brief release
                    pass
                else:
                    pending.pop(pick_index)
            if pick_index is None:
                time.sleep(0.05)
                continue
            ok, err = run_task(job.video, job.task, ns)
            with lock:
                running_counts[job.task] -= 1
                completed += 1
                if not ok and err:
                    failed.append(f"{job.task}:{job.video}: {err}")
            # loop continues until pending drained

    # spawn limited thread pool sized to sum of caps but not too large
    pool_size = min(sum(per_type_caps.values()), 16)
    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(pool_size)]
    for t in threads:
        t.start()
    last_report = 0
    try:
        while any(t.is_alive() for t in threads):
            with lock:
                comp = completed
                rem = total - comp
            if comp != last_report:
                with lock:
                    caps_state = ' '.join(f"{k}:{running_counts[k]}/{per_type_caps[k]}" for k in per_type_caps)
                print(f"Batch {comp}/{total} rem={rem} [{caps_state}]\r", end="", file=sys.stderr)
                last_report = comp
            time.sleep(0.2)
    finally:
        for t in threads:
            t.join()
    print(f"Batch {completed}/{total}", file=sys.stderr)
    if failed:
        print("Failures:", file=sys.stderr)
        for f in failed:
            print("  " + f, file=sys.stderr)
    print(f"Batch done: {completed - len(failed)} success, {len(failed)} failed")
    return 0 if not failed else 1

# ---------------------------------------------------------------------------
# Perceptual hash (pHash) generation
# ---------------------------------------------------------------------------

def phash_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.phash.json"


def compute_frame_hash_ahash(image_bytes: bytes) -> tuple[int, int]:
    """Compute 32x32 aHash -> (bits,width). Width = 1024 bits."""
    try:
        from PIL import Image  # type: ignore
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L").resize((32, 32))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = 0
        for i, p in enumerate(pixels):
            if p >= avg:
                bits |= (1 << i)
        return bits, 1024
    except Exception:
        # Fallback: degrade to 64-bit hash from raw bytes
        return (sum(image_bytes) & ((1 << 64) - 1)), 64


def _dct2(matrix: list[float], size: int) -> list[float]:
    """Naive 2D DCT (slow but OK for 32x32). Returns flat list length size*size."""
    # Implement a very small DCT to avoid numpy dependency; performance acceptable for handful of frames.
    import math
    N = size
    out = [0.0] * (N * N)
    for u in range(N):
        for v in range(N):
            s = 0.0
            for x in range(N):
                for y in range(N):
                    s += (
                        matrix[x * N + y]
                        * math.cos(((2 * x + 1) * u * math.pi) / (2 * N))
                        * math.cos(((2 * y + 1) * v * math.pi) / (2 * N))
                    )
            cu = math.sqrt(1 / N) if u == 0 else math.sqrt(2 / N)
            cv = math.sqrt(1 / N) if v == 0 else math.sqrt(2 / N)
            out[u * N + v] = cu * cv * s
    return out


def compute_frame_hash_dct(image_bytes: bytes) -> tuple[int, int]:
    """Classic pHash style: 32x32 -> DCT -> top-left 8x8 (excluding DC) using median threshold -> 64 bits."""
    try:
        from PIL import Image  # type: ignore
        import io, statistics
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L").resize((32, 32))
        pixels = list(img.getdata())  # 1024 values
        dct = _dct2(pixels, 32)
        # Extract top-left 8x8 coefficients (excluding [0,0])
        coeffs = []
        for u in range(8):
            for v in range(8):
                if u == 0 and v == 0:
                    continue
                coeffs.append(dct[u * 32 + v])
        median = statistics.median(coeffs)
        bits = 0
        for i, c in enumerate(coeffs):  # 63 coefficients
            if c >= median:
                bits |= (1 << i)
        # We currently have 63 bits; pad to 64 by leaving highest bit 0.
        return bits, 64
    except Exception:
        return compute_frame_hash_ahash(image_bytes)


def hash_to_hex(bits: int, width: int = 1024) -> str:
    # width = number of bits (32*32=1024). Convert to hex padded.
    hex_len = width // 4
    return f"{bits:0{hex_len}x}"[-hex_len:]


def extract_frame(video: Path, timestamp: float) -> bytes:
    if os.environ.get("FFPROBE_DISABLE"):
        return b"stub" + str(timestamp).encode()
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not available for frame extraction")
    # Use ffmpeg to output a single JPEG to pipe
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg frame extract failed")
    return proc.stdout


def parse_frame_time_spec(spec: str, duration: float | None) -> float:
    if spec == "middle":
        return (duration / 2.0) if duration and duration > 0 else 1.0
    if spec.endswith('%'):
        try:
            pct = float(spec[:-1]) / 100.0
            if duration and duration > 0:
                return max(0.0, min(duration - 0.5, duration * pct))
        except ValueError:
            pass
        return 1.0
    try:
        return float(spec)
    except ValueError:
        return 1.0


def compute_phash_video(video: Path, frame_time_spec: str, frames: int, force: bool, algo: str = "ahash", combine: str = "xor") -> str:
    out = phash_path(video)
    if out.exists() and not force:
        return json.loads(out.read_text())["phash"]
    duration = extract_duration_from_file(video) or 0
    # frame positions: if frames == 1 => single time. else evenly spaced across [0,duration]
    positions = []
    if frames <= 1:
        positions = [parse_frame_time_spec(frame_time_spec, duration)]
    else:
        for i in range(frames):
            positions.append((i + 0.5) * duration / frames)
    # Collect per-frame bits
    per_bits: list[int] = []
    bit_width = 0
    for pos in positions:
        img = extract_frame(video, pos)
        if algo == "dct":
            bits, width = compute_frame_hash_dct(img)
        else:
            bits, width = compute_frame_hash_ahash(img)
        bit_width = width
        per_bits.append(bits)

    if combine == "xor" or frames == 1:
        combined = 0
        for b in per_bits:
            combined ^= b
    else:
        # majority / avg (treated the same at bit level without raw coefficients)
        counts = [0] * bit_width
        for b in per_bits:
            for i in range(bit_width):
                if b & (1 << i):
                    counts[i] += 1
        threshold = frames / 2.0
        combined = 0
        for i, c in enumerate(counts):
            if c > threshold:
                combined |= (1 << i)
    algo_name = ("dct8x8" if algo == "dct" else "ahash32x32")
    hex_hash = hash_to_hex(combined, width=bit_width)
    out.write_text(json.dumps({
        "video": video.name,
        "phash": hex_hash,
        "algorithm": f"{algo_name}-{combine}" if frames > 1 else algo_name,
        "frames": frames,
        "positions": positions,
        "generated_at": time.time(),
    }, indent=2))
    return hex_hash


def find_phash_duplicates(root: Path, recursive: bool = False, threshold: float = 0.90, limit: int = 0) -> dict:
    """Scan existing *.phash.json artifacts and report likely duplicate pairs.

    Similarity metric primary: 1 - (hamming_distance / bits).
    Crossover (secondary metric): Jaccard index of set bits (intersection/union) to
    express apparent degree of shared visual features when using multi-frame XOR.

    Args:
        root: Directory to scan for videos / artifacts.
        recursive: Recurse into subdirectories.
        threshold: Minimum similarity (0..1) to consider a pair a duplicate candidate.
        limit: Optional max number of pairs to return (0 = no limit).

    Returns:
        dict with summary statistics, duplicate pairs, and grouped clusters.
    """
    root = root.expanduser().resolve()
    videos = find_mp4s(root, recursive)
    # Collect phashes (skip videos missing artifact)
    entries: list[dict] = []
    for v in videos:
        p = phash_path(v)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            h_hex = data.get("phash")
            if not isinstance(h_hex, str):
                continue
            bits_len = len(h_hex) * 4
            h_int = int(h_hex, 16)
            ones = h_int.bit_count()
            entries.append({
                "video": v.name,
                "path": str(v),
                "hex": h_hex,
                "int": h_int,
                "bits": bits_len,
                "ones": ones,
                "algorithm": data.get("algorithm"),
            })
        except Exception:
            continue
    n = len(entries)
    pairs: list[dict] = []
    # Group by bit width (only compare within same size / algorithm family)
    by_bits: dict[int, list[int]] = {}
    for idx, e in enumerate(entries):
        by_bits.setdefault(e["bits"], []).append(idx)
    for bits, idxs in by_bits.items():
        m = len(idxs)
        for i_pos in range(m):
            i = idxs[i_pos]
            ei = entries[i]
            for j_pos in range(i_pos + 1, m):
                j = idxs[j_pos]
                ej = entries[j]
                # (Optional) skip compare if algorithms differ producing differing bit distributions
                if ei.get("algorithm") != ej.get("algorithm"):
                    continue
                xor = ei["int"] ^ ej["int"]
                dist = xor.bit_count()
                sim = 1.0 - (dist / bits) if bits else 0.0
                if sim < threshold:
                    continue
                # crossover / Jaccard of set bits
                inter = (ei["int"] & ej["int"]).bit_count()
                union = ei["ones"] + ej["ones"] - inter
                jaccard = inter / union if union else 1.0
                pairs.append({
                    "a": ei["video"],
                    "b": ej["video"],
                    "similarity": round(sim, 6),
                    "distance": dist,
                    "bits": bits,
                    "crossover": round(jaccard, 6),
                    "overlap_bits": inter,
                    "ones_a": ei["ones"],
                    "ones_b": ej["ones"],
                    "algorithm": ei.get("algorithm"),
                })
    # Sort by descending similarity then ascending distance
    pairs.sort(key=lambda r: (-r["similarity"], r["distance"]))
    if limit > 0:
        pairs = pairs[:limit]
    # Build clusters via union-find
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pinfo in pairs:
        union(pinfo["a"], pinfo["b"])
    clusters: dict[str, list[str]] = {}
    for e in entries:
        v = e["video"]
        r = find(v)
        clusters.setdefault(r, []).append(v)
    # Filter clusters with >1 member
    group_list: list[dict] = []
    for rep, vids in clusters.items():
        if len(vids) < 2:
            continue
        # Compute average pairwise similarity within cluster (only among returned pairs)
        sims: list[float] = []
        vids_set = set(vids)
        for pinfo in pairs:
            if pinfo["a"] in vids_set and pinfo["b"] in vids_set:
                sims.append(pinfo["similarity"])
        avg_sim = sum(sims) / len(sims) if sims else 1.0
        group_list.append({
            "representative": rep,
            "videos": sorted(vids),
            "count": len(vids),
            "avg_similarity": round(avg_sim, 6),
        })
    group_list.sort(key=lambda g: (-g["avg_similarity"], -g["count"]))
    return {
        "root": str(root),
        "videos_total": len(videos),
        "with_phash": n,
        "threshold": threshold,
        "pairs": pairs,
        "groups": group_list,
    }


def cmd_phash(ns) -> int:
    """Compute perceptual hash (single or multi-frame combined) for duplicates detection."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    total = len(videos)
    existing = 0
    q: Queue[Path] = Queue()
    for v in videos:
        if (not ns.force) and phash_path(v).exists():
            existing += 1
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"phash: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        if ns.output_format == "json":
            print(json.dumps({"results": []}, indent=2))
        return 0
    print(f"phash: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    done = 0
    generated = 0
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, str]] = []  # (video name, hash)

    def worker():
        nonlocal done, generated
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                prev_exists = phash_path(vid).exists()
                h = compute_phash_video(vid, ns.time, ns.frames, ns.force, algo=ns.algo, combine=ns.combine)
                with lock:
                    results.append((vid.name, h))
                    if (not prev_exists) or ns.force:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total: covered = total
                        print(f"phash gen {covered}/{total} {vid.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid}: {e}")
            finally:
                with lock:
                    done += 1
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        with lock:
            covered = existing + generated if not ns.force else generated
            if covered > total: covered = total
            print(f"phash scan {covered}/{total} (generated {generated})\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total: covered = total
    print(f"phash scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": [{"video": v, "phash": h} for v, h in results]}, indent=2))
    else:
        for v, h in results:
            print(f"{h}  {v}")
    if errors:
        print("Errors (some hashes missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    return 0 if not errors else 1


def cmd_phash_dupes(ns) -> int:
    """Report likely duplicate video pairs using existing pHash artifacts.

    Requires that phash artifacts already exist (run 'phash' first). Outputs
    similarity (1 - hamming/bits) and crossover (Jaccard of set bits) stats.
    """
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    data = find_phash_duplicates(root, ns.recursive, ns.threshold, ns.limit)
    if ns.output_format == "json":
        print(json.dumps(data, indent=2))
    else:
        pairs = data.get("pairs", [])
        if not pairs:
            print("No duplicate candidates above threshold")
        else:
            for p in pairs:
                print(f"{p['similarity']:.4f} crossover={p['crossover']:.4f} {p['a']} <> {p['b']} bits={p['bits']} dist={p['distance']}")
            # groups summary
            groups = data.get("groups", [])
            if groups:
                print("\nGroups:")
                for g in groups:
                    print(f"{g['avg_similarity']:.4f} x{g['count']} {', '.join(g['videos'])}")
    return 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    # Preprocess to allow directory-first style: index.py /path command ...
    if argv:
        try:
            first = Path(argv[0])
            # Known command names set (keep in sync with subparsers)
            # Keep this synchronized with the set in main(); only current, supported commands (no legacy aliases)
            command_names = {"list","report","orphans","metadata","thumbs","sprites","previews","subtitles","phash","dupes","heatmaps","scenes","embed","listing","rename","codecs","transcode","compare","batch","finish"}
            if first.exists() and len(argv) > 1 and argv[1] in command_names:
                # Swap to command-first
                argv = [argv[1], *argv[2:], str(first)]
            # Also support form: command path (original default) so no change needed.
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(description="Video utility (list, metadata)")
    p.add_argument("-d", "--dir", dest="root_dir", default=None, help="Root directory for commands (alternate to positional directory)")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="List mp4 files")
    lp.add_argument("directory", nargs="?", default=".")
    lp.add_argument("-r", "--recursive", action="store_true")
    lp.add_argument("--json", action="store_true")
    lp.add_argument("--show-size", action="store_true")
    lp.add_argument("--sort", choices=["name", "size"], default="name")

    mp = sub.add_parser("metadata", help="Generate ffprobe metadata for mp4 files (multi-threaded)")
    mp.add_argument("directory", nargs="?", default=".")
    mp.add_argument("-r", "--recursive", action="store_true")
    mp.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    mp.add_argument("--workers", type=int, default=1)

    tp = sub.add_parser("thumbs", help="Generate cover thumbnails (JPEG)")
    tp.add_argument("directory", nargs="?", default=".")
    tp.add_argument("-r", "--recursive", action="store_true")
    tp.add_argument("--time", dest="time_spec", default="middle", help="When to capture frame: seconds (e.g. 10), percentage (e.g. 25%), or 'middle' (default)")
    tp.add_argument("--force", action="store_true", help="Regenerate even if thumbnail exists")
    tp.add_argument("--workers", type=int, default=1, help="Worker threads (cap 4)")
    tp.add_argument("--quality", type=int, default=2, help="JPEG quality scale (ffmpeg -q:v, 2=high, 31=low)")

    sp = sub.add_parser("sprites", help="Generate sprite sheet + JSON for hover preview")
    sp.add_argument("directory", nargs="?", default=".")
    sp.add_argument("-r", "--recursive", action="store_true")
    sp.add_argument("--interval", type=float, default=10.0, help="Seconds between frames (default 10)")
    sp.add_argument("--width", type=int, default=320, help="Tile width (scaled; height auto)")
    sp.add_argument("--cols", type=int, default=10, help="Sheet columns (default 10)")
    sp.add_argument("--rows", type=int, default=10, help="Sheet rows (default 10)")
    sp.add_argument("--quality", type=int, default=4, help="JPEG quality for sheet (2=high)" )
    sp.add_argument("--force", action="store_true", help="Regenerate even if sheet/json exist")
    sp.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = auto up to cols*rows)")

    hp = sub.add_parser("previews", help="Generate short hover preview clips (decile segments)")
    hp.add_argument("directory", nargs="?", default=".")
    hp.add_argument("-r", "--recursive", action="store_true")
    hp.add_argument("--segments", type=int, default=9, help="Number of segments (default 9 for 10%-90% deciles)")
    hp.add_argument("--duration", type=float, default=1.0, help="Seconds per preview clip (default 1.0)")
    hp.add_argument("--width", type=int, default=320, help="Preview clip width (scale, keep aspect)")
    hp.add_argument("--format", choices=["mp4", "webm"], default="webm", help="Output container/codec (default webm)")
    hp.add_argument("--crf", type=int, default=30, help="Quality factor (x264/webm CRF, lower=better)")
    hp.add_argument("--bitrate", type=str, default="300k", help="Target bitrate for webm (if using webm)")
    hp.add_argument("--workers", type=int, default=2, help="Worker threads (cap 4)")
    hp.add_argument("--force", action="store_true", help="Regenerate existing previews")
    hp.add_argument("--no-index", action="store_true", help="Skip writing index JSON")
    hp.add_argument("--output-format", choices=["json","text"], default="text")

    sb = sub.add_parser("subtitles", help="Generate SRT subtitles via Whisper (whisper / faster-whisper / whisper.cpp)")
    sb.add_argument("directory", nargs="?", default=".")
    sb.add_argument("-r", "--recursive", action="store_true")
    sb.add_argument("--model", default="small", help="Model name (default small)")
    sb.add_argument("--backend", choices=["auto", "whisper", "faster-whisper", "whisper.cpp"], default="auto", help="Backend selection (default auto)")
    sb.add_argument("--language", default=None, help="Source language (auto detect if omitted)")
    sb.add_argument("--translate", action="store_true", help="Translate to English")
    sb.add_argument("--workers", type=int, default=1, help="Parallel videos (cap 2)")
    sb.add_argument("--force", action="store_true", help="Regenerate even if subtitle file exists")
    sb.add_argument("--output-dir", default=None, help="Directory to place subtitle files (defaults alongside videos)")
    sb.add_argument("--whisper-cpp-bin", default=None, help="Path to whisper.cpp binary (if backend whisper.cpp)")
    sb.add_argument("--whisper-cpp-model", default=None, help="Path to whisper.cpp model (.bin)")
    sb.add_argument("--compute-type", default=None, help="faster-whisper compute type override (auto=int8 if translate else float16). Fallbacks applied if unsupported.")

    bt = sub.add_parser("batch", help="Ephemeral multi-stage pipeline (metadata/thumbs/sprites/previews/subtitles) with per-type concurrency caps")
    bt.add_argument("directory", nargs="?", default=".")
    bt.add_argument("-r", "--recursive", action="store_true")
    bt.add_argument("--tasks", default="metadata,thumbs", help="Comma list of tasks: metadata,thumbs,sprites,previews,subtitles,phash (default metadata,thumbs)")
    bt.add_argument("--max-metadata", type=int, default=3)
    bt.add_argument("--max-thumbs", type=int, default=4)
    bt.add_argument("--max-sprites", type=int, default=2)
    bt.add_argument("--max-previews", type=int, default=2)
    bt.add_argument("--max-subtitles", type=int, default=1)
    bt.add_argument("--max-phash", type=int, default=4)
    bt.add_argument("--max-scenes", type=int, default=1)
    bt.add_argument("--preview-width", type=int, default=320)
    bt.add_argument("--preview-duration", type=float, default=1.0)
    bt.add_argument("--preview-segments", type=int, default=9)
    bt.add_argument("--sprites-interval", type=float, default=10.0)
    bt.add_argument("--sprites-width", type=int, default=320)
    bt.add_argument("--sprites-cols", type=int, default=10)
    bt.add_argument("--sprites-rows", type=int, default=10)
    bt.add_argument("--subtitles-model", default="small")
    bt.add_argument("--subtitles-backend", default="auto")
    bt.add_argument("--subtitles-language", default=None)
    bt.add_argument("--subtitles-translate", action="store_true")
    bt.add_argument("--force", action="store_true", help="Force regenerate all artifacts")

    ph = sub.add_parser("phash", help="Compute perceptual hash for video(s) for deduplication")
    ph.add_argument("directory", nargs="?", default=".")
    ph.add_argument("-r", "--recursive", action="store_true")
    ph.add_argument("--time", default="middle", help="Timestamp 'middle', seconds (e.g. 60), or percentage (e.g. 25%)")
    ph.add_argument("--frames", type=int, default=5, help="Number of evenly spaced frames to combine (XOR) (default 5 for broader coverage; use 1 for speed)")
    ph.add_argument("--force", action="store_true")
    ph.add_argument("--workers", type=int, default=2, help="Worker threads (cap 4)")
    ph.add_argument("--output-format", choices=["json","text"], default="json")
    ph.add_argument("--algo", choices=["ahash","dct"], default="ahash", help="Frame hash algorithm: ahash (32x32 avg) or dct (classic 8x8 pHash) (default ahash)")
    ph.add_argument("--combine", choices=["xor","majority","avg"], default="xor", help="Combine multi-frame hashes: xor (fast), majority/avg (bit vote) (default xor)")

    phd = sub.add_parser("dupes", help="Find likely duplicate videos using existing pHash artifacts")
    phd.add_argument("directory", nargs="?", default=".")
    phd.add_argument("-r", "--recursive", action="store_true")
    phd.add_argument("--threshold", type=float, default=0.90, help="Minimum similarity (0-1) (default 0.90)")
    phd.add_argument("--limit", type=int, default=0, help="Limit number of pairs reported (0 = all)")
    phd.add_argument("--output-format", choices=["json","text"], default="json")

    fe = sub.add_parser("embed", help="Extract distinct face signatures per video (tracking + clustering)")
    fe.add_argument("directory", nargs="?", default=".")
    fe.add_argument("-r", "--recursive", action="store_true")
    fe.add_argument("--sample-rate", type=float, default=1.0, help="Frames per second to sample (default 1.0)")
    fe.add_argument("--min-face-area", type=int, default=1600, help="Minimum face box area (pixels) (default 1600)")
    fe.add_argument("--blur-threshold", type=float, default=30.0, help="Reject faces from frames with very low sharpness (variance threshold)")
    fe.add_argument("--max-gap", type=float, default=0.75, help="Max seconds gap to continue a track (default 0.75)")
    fe.add_argument("--min-track-frames", type=int, default=3, help="Minimum frames for a track to be kept (default 3)")
    fe.add_argument("--match-distance", type=float, default=0.55, help="Embedding L2 distance to merge into existing track (default 0.55)")
    fe.add_argument("--cluster-eps", type=float, default=0.40, help="DBSCAN eps for merging split tracks (default 0.40)")
    fe.add_argument("--cluster-min-samples", type=int, default=1, help="DBSCAN min_samples (default 1)")
    fe.add_argument("--thumbnails", action="store_true", help="Write exemplar face thumbnails per signature")
    fe.add_argument("--output-format", choices=["json","text"], default="json")
    fe.add_argument("--force", action="store_true", help="Force regeneration (currently no per-video cache file)")

    fi = sub.add_parser("listing", help="Aggregate distinct face signatures across videos into a global index (people listing)")
    fi.add_argument("directory", nargs="?", default=".")
    fi.add_argument("-r", "--recursive", action="store_true")
    fi.add_argument("--sample-rate", type=float, default=1.0)
    fi.add_argument("--cluster-eps", type=float, default=0.45)
    fi.add_argument("--cluster-min-samples", type=int, default=1)
    fi.add_argument("--output", default=None, help="Write index JSON to file (otherwise stdout)")
    fi.add_argument("--output-format", choices=["json","text"], default="json")
    fi.add_argument("--gallery", default=None, help="Optional labeled people directory to auto-tag clusters (subdirs=person names, images/videos inside)")
    fi.add_argument("--gallery-sample-rate", type=float, default=1.0, help="Video sampling rate (fps) when scanning gallery videos")
    fi.add_argument("--label-threshold", type=float, default=0.40, help="Max L2 distance to assign a label (default 0.40)")

    hm = sub.add_parser("heatmaps", help="Generate brightness/motion heatmaps timeline JSON (and optional PNG)")
    hm.add_argument("directory", nargs="?", default=".")
    hm.add_argument("-r", "--recursive", action="store_true")
    hm.add_argument("--interval", type=float, default=5.0, help="Seconds between samples (default 5.0)")
    hm.add_argument("--mode", choices=["brightness","motion","both"], default="both", help="Metrics to compute (default both)")
    hm.add_argument("--png", action="store_true", help="Also write a small PNG stripe visualization")
    hm.add_argument("--workers", type=int, default=2, help="Parallel videos (cap 4)")
    hm.add_argument("--force", action="store_true")
    hm.add_argument("--output-format", choices=["json","text"], default="json")
    hm.add_argument("--verbose", action="store_true", help="Print per-video diagnostic details (durations, sample counts, skips, errors)")

    orp = sub.add_parser("orphans", help="List (and optionally delete) orphaned artifact files whose parent video is missing")
    orp.add_argument("directory", nargs="?", default=".")
    orp.add_argument("-r", "--recursive", action="store_true")
    orp.add_argument("--delete", action="store_true", help="Delete orphan artifact files")
    orp.add_argument("--prune-empty", action="store_true", help="After deletion, remove empty .artifacts directories")
    orp.add_argument("--output-format", choices=["text","json"], default="text")

    sc = sub.add_parser("scenes", help="Detect scene boundaries and generate markers (optional thumbs/clips)")
    sc.add_argument("directory", nargs="?", default=".")
    sc.add_argument("-r", "--recursive", action="store_true")
    sc.add_argument("--threshold", type=float, default=0.4, help="Scene score threshold (ffmpeg select gt(scene,TH)) (default 0.4)")
    sc.add_argument("--limit", type=int, default=0, help="Limit number of markers (0 = no limit)")
    sc.add_argument("--thumbs", action="store_true", help="Generate thumbnail per scene")
    sc.add_argument("--clips", action="store_true", help="Generate short clip per scene start")
    sc.add_argument("--clip-duration", type=float, default=2.0, help="Seconds per scene clip (default 2.0)")
    sc.add_argument("--thumbs-width", type=int, default=320, help="Thumbnail width (default 320)")
    sc.add_argument("--workers", type=int, default=2, help="Parallel videos (cap 4)")
    sc.add_argument("--force", action="store_true")
    sc.add_argument("--output-format", choices=["json","text"], default="json")

    # (Removed earlier simple 'faces' detector parser to avoid duplicate name conflict.)


    cd = sub.add_parser("codecs", help="Scan library for codec/profile compatibility")
    cd.add_argument("directory", nargs="?", default=".")
    cd.add_argument("-r", "--recursive", action="store_true")
    cd.add_argument("--target-v", default="h264", help="Target video codec (default h264)")
    cd.add_argument("--target-a", default="aac", help="Target audio codec (or copy) (default aac)")
    cd.add_argument("--allowed-profiles", default="high,main,constrained baseline", help="Comma list acceptable H.264 profiles")
    cd.add_argument("--workers", type=int, default=4)
    cd.add_argument("--log", default=None, help="Write incompatible entries to this log file")
    cd.add_argument("--output-format", choices=["json","text"], default="text")
    cd.add_argument("--all", action="store_true", help="Show all entries, not just incompatible ones")

    tc = sub.add_parser("transcode", help="Batch transcode videos toward target codecs")
    tc.add_argument("directory", nargs="?", default=".")
    tc.add_argument("dest", help="Output directory root")
    tc.add_argument("-r", "--recursive", action="store_true")
    tc.add_argument("--target-v", default="h264")
    tc.add_argument("--target-a", default="aac")
    tc.add_argument("--allowed-profiles", default="high,main,constrained baseline")
    tc.add_argument("--crf", type=int, default=28)
    tc.add_argument("--v-bitrate", default=None, help="Optional target video bitrate (e.g. 3000k) overrides CRF for HW encoders")
    tc.add_argument("--a-bitrate", default="128k")
    tc.add_argument("--preset", default="medium")
    tc.add_argument("--hardware", choices=["none","videotoolbox"], default="none")
    tc.add_argument("--drop-subtitles", action="store_true", help="Exclude subtitle streams")
    tc.add_argument("--workers", type=int, default=1)
    tc.add_argument("--force", action="store_true", help="Force transcode even if compatible")
    tc.add_argument("--dry-run", action="store_true")
    tc.add_argument("--progress", action="store_true", help="Show real-time progress for each file")
    tc.add_argument("--output-format", choices=["json","text"], default="json")

    cmpc = sub.add_parser("compare", help="Compare two video files (SSIM/PSNR)")
    cmpc.add_argument("original")
    cmpc.add_argument("other")
    cmpc.add_argument("--progress", action="store_true", help="Show real-time progress")
    cmpc.add_argument("--output-format", choices=["json","text"], default="text")

    rpt = sub.add_parser("report", help="Report artifact coverage percentages across videos")
    rpt.add_argument("directory", nargs="?", default=".")
    rpt.add_argument("-r", "--recursive", action="store_true")
    rpt.add_argument("--output-format", choices=["json","text"], default="text")

    fn = sub.add_parser("finish", help="Generate missing artifacts based on report")
    fn.add_argument("directory", nargs="?", default=".")
    fn.add_argument("-r", "--recursive", action="store_true")
    fn.add_argument("--workers", type=int, default=2)

    # Deprecated face-related commands removed in favor of embed / listing.

    rn = sub.add_parser("rename", help="Rename a video and associated artifacts")
    rn.add_argument("src")
    rn.add_argument("dst")

    return p.parse_args(argv)


def cmd_list(ns) -> int:
    """List .mp4 files with optional size sorting / JSON output."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    mp4s = find_mp4s(root, ns.recursive)
    records = [build_record(p) for p in mp4s]
    if ns.sort == "name":
        records.sort(key=lambda r: r["name"].lower())
    else:
        records.sort(key=lambda r: r["size_bytes"], reverse=True)
    if ns.json:
        json.dump(records, sys.stdout, indent=2)
        print()
    else:
        if not records:
            print("No MP4 files found.")
        else:
            for r in records:
                line = r['path'] if not ns.show_size else f"{r['name']}\t{human_size(r['size_bytes'])}\t{r['path']}"
                print(line)
            print(f"Total: {len(records)} file(s)")
    return 0


def cmd_heatmaps(ns) -> int:
    """Compute brightness & motion heatmaps timeline (JSON, optional PNG)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    total = len(videos)
    existing = 0
    q: Queue[Path] = Queue()
    missing_list: list[str] = []
    for v in videos:
        exists = heatmaps_json_path(v).exists()
        if (not ns.force) and exists:
            existing += 1
            if ns.verbose:
                print(f"heatmaps skip existing {v.name}", file=sys.stderr)
        else:
            missing_list.append(v.name)
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"heatmaps: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        return 0
    print(f"heatmaps: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    if ns.verbose and missing_list:
        preview = ", ".join(missing_list[:8]) + (" ..." if len(missing_list) > 8 else "")
        print(f"heatmaps missing targets ({len(missing_list)}): {preview}", file=sys.stderr)
    done = 0
    generated = 0
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, dict]] = []

    def worker():
        nonlocal done, generated
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                prev_exists = heatmaps_json_path(vid).exists()
                if ns.verbose:
                    # Pre-compute planned samples count using duration probe (best-effort)
                    dur_probe = extract_duration_from_file(vid) or 0
                    est_samples = 0
                    if dur_probe > 0 and ns.interval > 0:
                        est_samples = int(max(1, math.ceil(dur_probe / ns.interval)))
                    print(f"heatmaps start {vid.name} dur~{dur_probe:.2f}s interval={ns.interval}s est_samples~{est_samples}", file=sys.stderr)
                data = compute_heatmaps(vid, ns.interval, ns.mode, ns.force, ns.png)
                if ns.verbose:
                    print(f"heatmaps done  {vid.name} samples={len(data.get('samples', []))}", file=sys.stderr)
                if not isinstance(data, dict) or "samples" not in data:
                    raise RuntimeError("heatmaps: unexpected data structure")
                with lock:
                    results.append((vid.name, data))
                    if (not prev_exists) or ns.force:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total: covered = total
                        print(f"heatmaps gen {covered}/{total} {vid.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    err_msg = f"{vid.name}: {e}"
                    errors.append(err_msg)
                    if ns.verbose:
                        print(f"heatmaps error {err_msg}", file=sys.stderr)
            finally:
                with lock:
                    done += 1
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    # Periodic scan progress (mirrors thumbs/phash style)
    while any(t.is_alive() for t in threads):
        with lock:
            covered = existing + generated if not ns.force else generated
            if covered > total: covered = total
            print(f"heatmaps scan {covered}/{total} (generated {generated})\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total: covered = total
    print(f"heatmaps scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if ns.verbose:
        # Summary of any failures
        if errors:
            print(f"heatmaps failures: {len(errors)} (see above)", file=sys.stderr)
        produced = existing + generated if not ns.force else generated
        if produced < total and not errors:
            # Identify residual missing
            residual = [v.name for v in videos if not heatmaps_json_path(v).exists()]
            if residual:
                preview_r = ", ".join(residual[:8]) + (" ..." if len(residual) > 8 else "")
                print(f"heatmaps still missing {len(residual)}: {preview_r}", file=sys.stderr)
    if errors:
        for e in errors:
            print("Error:", e, file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": [
            {"video": v, "heatmaps": {k: val for k, val in d.items() if k != "samples"}, "samples": d.get("samples")}
            for v, d in results
        ]}, indent=2))
    else:
        for v, d in results:
            print(v, len(d.get("samples", [])))
    return 1 if errors else 0


def cmd_orphans(ns) -> int:
    """List (and optionally delete) orphan artifact files.

    An artifact is considered orphaned if it's inside a .artifacts directory whose parent
    directory does not contain a corresponding MP4 whose stem is a prefix match of the
    artifact filename (exact stem plus '.').
    """
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    artifact_dirs: list[Path] = []
    if ns.recursive:
        for p in root.rglob(ARTIFACTS_DIR):
            if p.is_dir():
                artifact_dirs.append(p)
    else:
        d = root / ARTIFACTS_DIR
        if d.is_dir():
            artifact_dirs.append(d)
    orphans: list[Path] = []
    scanned = 0
    ignore_names = {".DS_Store"}
    for ad in artifact_dirs:
        parent = ad.parent
        video_stems = {v.stem for v in parent.iterdir() if v.is_file() and v.suffix.lower() in fMEDIA_EXTS}
        for f in ad.iterdir():
            if f.is_dir():
                continue
            if f.name in ignore_names:
                continue
            scanned += 1
            # An artifact file is linked to a video if its filename begins with '<stem>.' EXACTLY (not partial word fragments)
            stem_part = f.name.split('.', 1)[0]
            if stem_part in video_stems:
                continue
            orphans.append(f)
    deleted = 0
    if ns.delete:
        for f in orphans:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:  # noqa: BLE001
                print(f"Failed to delete {f}: {e}", file=sys.stderr)
        if ns.prune_empty:
            # Remove any now-empty artifact dirs
            for ad in artifact_dirs:
                try:
                    if any(ad.iterdir()):
                        continue
                    ad.rmdir()
                except Exception:
                    pass
    if ns.output_format == "json":
        print(json.dumps({
            "root": str(root),
            "scanned_artifact_files": scanned,
            "orphans_found": len(orphans),
            "deleted": deleted,
            "files": [str(p) for p in orphans],
        }, indent=2))
    else:
        if not orphans:
            print("No orphan artifacts found")
        else:
            print(f"Orphan artifacts ({len(orphans)}):")
            for p in orphans:
                print(p)
            if ns.delete:
                print(f"Deleted: {deleted}")
    return 0


def cmd_scenes(ns) -> int:
    """Detect scene boundaries (FFmpeg select filter) and optionally thumbs/clips."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    total = len(videos)
    existing = 0
    q: Queue[Path] = Queue()
    for v in videos:
        if (not ns.force) and scenes_json_path(v).exists():
            existing += 1
        q.put(v)
    remaining = total - existing if not ns.force else total
    pct_existing = (existing / total * 100.0) if total else 0.0
    if remaining == 0:
        print(f"scenes: {existing}/{total} ({pct_existing:.1f}%) already present. Nothing to do (use --force).", file=sys.stderr)
        return 0
    print(f"scenes: {existing}/{total} ({pct_existing:.1f}%) existing. Generating {remaining}...", file=sys.stderr)
    done = 0
    generated = 0
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, dict]] = []

    def worker():
        nonlocal done, generated
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                prev_exists = scenes_json_path(vid).exists()
                data = generate_scene_artifacts(vid, ns.threshold, ns.limit, ns.thumbs, ns.clips, ns.thumbs_width, ns.clip_duration, ns.force)
                with lock:
                    results.append((vid.name, data))
                    if (not prev_exists) or ns.force:
                        generated += 1
                        covered = existing + generated if not ns.force else generated
                        if covered > total: covered = total
                        print(f"scenes gen {covered}/{total} {vid.name}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid.name}: {e}")
            finally:
                with lock:
                    done += 1
                    print(f"scenes {done}/{total}\r", end="", file=sys.stderr)
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with lock:
        covered = existing + generated if not ns.force else generated
        if covered > total: covered = total
    print(f"scenes scan {covered}/{total} (generated {generated})", file=sys.stderr)
    if errors:
        for e in errors:
            print("Error:", e, file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": [
            {"video": v, "count": len(d.get("markers", [])), "threshold": d.get("threshold"), "markers": d.get("markers")}
            for v, d in results
        ]}, indent=2))
    else:
        for v, d in results:
            print(v, len(d.get("markers", [])))
    return 1 if errors else 0

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def faces_json_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.faces.json"


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1e-12
    return x / n


def detect_faces_stub(video: Path) -> list[dict]:
    """Return a placeholder list of faces for the given video."""
    return [
        {
            "time": 0.0,
            "box": [0, 0, 100, 100],
            "score": 1.0,
            "embedding": [],
        }
    ]


def _ensure_openface_model() -> Path:
    """Download the OpenFace embedding model if not present."""
    url = (
        "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
    )
    cache = Path.home() / ".cache" / "vid"
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / "openface.nn4.small2.v1.t7"
    if not model_path.exists():
        try:
            import urllib.request

            urllib.request.urlretrieve(url, model_path)
        except Exception:
            return model_path
    return model_path


def detect_faces(video: Path, interval: float = 1.0) -> list[dict]:
    """Detect faces and generate embeddings using OpenCV + OpenFace.

    Frames are sampled roughly once per ``interval`` seconds. Each detection
    entry contains the timestamp (seconds), bounding box [x, y, w, h], and a
    128-d embedding vector. If any step fails, a stub response is returned so
    the command still produces output.
    """
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError("cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(fps * interval), 1)
        import cv2.data
        cascade = cv2.CascadeClassifier(
            f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("cascade not found")

        # Load embedding model once
        model_path = _ensure_openface_model()
        net = cv2.dnn.readNetFromTorch(str(model_path))

        results: list[dict] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = cascade.detectMultiScale(gray, 1.1, 5)
                t = frame_idx / fps if fps else 0.0
                for (x, y, w, h) in detections:
                    face = frame[y : y + h, x : x + w]
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(face, (96, 96)),
                        1 / 255.0,
                        (96, 96),
                        (0, 0, 0),
                        swapRB=True,
                        crop=False,
                    )
                    net.setInput(blob)
                    vec = net.forward()[0]
                    embedding = [round(float(v), 6) for v in vec.tolist()]
                    results.append(
                        {
                            "time": round(t, 3),
                            "box": [int(x), int(y), int(w), int(h)],
                            "score": 1.0,
                            "embedding": embedding,
                        }
                    )
            frame_idx += 1
        cap.release()
        return results
    except Exception:
        return detect_faces_stub(video)


def generate_faces(video: Path, force: bool) -> dict:
    out_json = faces_json_path(video)
    if out_json.exists() and not force:
        return json.loads(out_json.read_text())
    data = {"faces": detect_faces(video)}
    out_json.write_text(json.dumps(data, indent=2))
    return data

 # (Removed legacy actor recognition commands and deepface dependency.)
 # Restored actor recognition commands (optional DeepFace-based gallery build & match)

# ---------------------------------------------------------------------------
# Heatmaps generation (brightness/motion timeline)
# ---------------------------------------------------------------------------

def heatmaps_json_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.heatmaps.json"


def heatmaps_png_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.heatmaps.png"


def compute_heatmaps(video: Path, interval: float, mode: str, force: bool, write_png: bool) -> dict:
    out_json = heatmaps_json_path(video)
    if out_json.exists() and not force:
        return json.loads(out_json.read_text())
    duration = extract_duration_from_file(video) or 0
    if duration <= 0:
        duration = 1.0
    samples: list[dict] = []
    prev_pixels: list[int] | None = None
    timestamps: list[float] = []
    t = 0.0
    while t < duration:
        timestamps.append(min(t, max(0.0, duration - 0.001)))
        t += interval
    if not timestamps or timestamps[-1] < duration - (interval * 0.5):
        timestamps.append(duration * 0.999)
    # Extract frames one by one (keeps memory low)
    for ts in timestamps:
        if os.environ.get("FFPROBE_DISABLE") or not ffmpeg_available():
            # stub
            brightness = 128.0
            motion = 0.0
            samples.append({"t": ts, "brightness": brightness, "motion": motion})
            continue
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", "scale=32:32:force_original_aspect_ratio=decrease,pad=32:32:(ow-iw)/2:(oh-ih)/2:color=black,format=gray",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extract failed at {ts:.2f}s")
        img_bytes = proc.stdout
        try:
            from PIL import Image  # type: ignore
            import io
            im = Image.open(io.BytesIO(img_bytes))
            pixels = list(im.getdata())
        except Exception:
            pixels = list(img_bytes[:1024])  # crude fallback
        # brightness (guard against empty pixel list -> ffmpeg decode failure produced no bytes)
        if not pixels:
            pixels = [0]
        avg_brightness = sum(pixels) / len(pixels)
        motion_val = 0.0
        if prev_pixels is not None and (mode in ("motion", "both")):
            # mean absolute difference normalized
            diffs = 0
            for a, b in zip(pixels, prev_pixels):
                diffs += abs(a - b)
            if pixels:
                motion_val = diffs / (len(pixels) * 255.0)
        samples.append({
            "t": ts,
            "brightness": avg_brightness if mode in ("brightness", "both", "motion") else None,
            "motion": motion_val if mode in ("motion", "both") else None,
        })
        prev_pixels = pixels
    data = {
        "video": video.name,
        "duration": duration,
        "interval": interval,
        "mode": mode,
        "samples": samples,
        "generated_at": time.time(),
    }
    out_json.write_text(json.dumps(data, indent=2))
    if write_png:
        try:
            from PIL import Image  # type: ignore
            # Build a 1-pixel tall strip (or 2 if both) then scale vertically
            w = len(samples)
            if mode == "brightness":
                row = [int(min(255, max(0, s["brightness"])) ) for s in samples]
                img = Image.new("L", (w, 1))
                img.putdata(row)
                img = img.resize((w, 32), resample=Image.Resampling.NEAREST)
                img.save(heatmaps_png_path(video))
            elif mode == "motion":
                row = [int(min(255, max(0, (s["motion"] or 0) * 255))) for s in samples]
                img = Image.new("L", (w, 1))
                img.putdata(row)
                img = img.resize((w, 32), resample=Image.Resampling.NEAREST)
                img.save(heatmaps_png_path(video))
            else:  # both -> two rows stacked
                row1 = [int(min(255, max(0, s["brightness"])) ) for s in samples]
                row2 = [int(min(255, max(0, (s["motion"] or 0) * 255))) for s in samples]
                img = Image.new("L", (w, 2))
                img.putdata(row1 + row2)
                img = img.resize((w, 64), resample=Image.Resampling.NEAREST)
                img.save(heatmaps_png_path(video))
        except Exception:
            pass
    return data

# ---------------------------------------------------------------------------
# Scene boundary detection (markers with optional previews / thumbs)
# ---------------------------------------------------------------------------
# TODO (future): "Marker animated previews" feature.
# Concept: For each detected scene boundary, generate a short looping animated preview (e.g. 0.5–1.0s before and after the cut) to aid UI hover inspection.
# Proposed design:
#   - Input: existing scenes JSON (times + scores).
#   - For each marker time t, extract clip [t-Δ_pre, t+Δ_post] with safe clamps (Δ defaults 0.5s).
#   - Output formats: animated WebP (preferred), fallback MP4/WebM; store in .<video>.mp4.scenes/scene_XXX.preview.webp (or .mp4) plus update scenes JSON adding `preview` field.
#   - Option flags: --preview-format webp|mp4|webm, --preview-length 1.0, --preview-before 0.5, --preview-after 0.5, --preview-width 320, --preview-fps 12, --preview-quality (WebP quality or CRF).
#   - Performance: batch ffmpeg invocation per preview or consider filter_complex with select for multi-output (opt-in speed mode).
#   - Caching: skip if file exists unless --force.
#   - Integration: scenes subcommand optional flag --animated-previews; batch pipeline new task name 'scene-previews' (depends on scenes artifacts existing or performs detection if missing).
#   - Future enhancement: sprite sheet of scene previews for ultra-fast UI load.
# This is documentation only; not implemented yet.

def scenes_json_path(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.scenes.json"


def scenes_dir(video: Path) -> Path:
    return artifact_dir(video) / f"{video.stem}.scenes"


def detect_scenes(video: Path, threshold: float) -> list[tuple[float, float]]:
    """Detect scene boundaries using PySceneDetect (excluding time zero)."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video_stream = open_video(str(video))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video_stream)
        scenes = scene_manager.get_scene_list()
        boundaries = [start.get_seconds() for start, _ in scenes[1:]]
        return [(t, 1.0) for t in boundaries]
    except Exception:
        return [(1.0, 1.0)]


def generate_scene_artifacts(video: Path, threshold: float, limit: int, gen_thumbs: bool, gen_clips: bool, thumbs_width: int, clip_duration: float, force: bool) -> dict:
    out_json = scenes_json_path(video)
    if out_json.exists() and not force:
        return json.loads(out_json.read_text())
    markers = detect_scenes(video, threshold)
    if limit > 0:
        markers = markers[:limit]
    duration = extract_duration_from_file(video) or 0
    info_list = []
    asset_dir = scenes_dir(video)
    if (gen_thumbs or gen_clips) and not asset_dir.exists():
        asset_dir.mkdir()
    for idx, (t, score) in enumerate(markers):
        entry: dict = {"time": t, "score": score}
        safe_t = max(0.0, min(t, max(0.0, duration - 0.01)))
        if gen_thumbs:
            thumbs_file = asset_dir / f"scene_{idx:03d}.jpg"
            if (not thumbs_file.exists()) or force:
                if os.environ.get("FFPROBE_DISABLE") or not ffmpeg_available():
                    thumbs_file.write_text(f"stub thumbs {safe_t}")
                else:
                    cmd = [
                        "ffmpeg", "-v", "error", "-ss", f"{safe_t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={thumbs_width}:-1:force_original_aspect_ratio=decrease",
                        "-qscale:v", "4", str(thumbs_file)
                    ]
                    subprocess.run(cmd, capture_output=True)
            entry["thumbs"] = thumbs_file.name
        if gen_clips:
            clip_file = asset_dir / f"scene_{idx:03d}.mp4"
            if (not clip_file.exists()) or force:
                if os.environ.get("FFPROBE_DISABLE") or not ffmpeg_available():
                    clip_file.write_text(f"stub clip {safe_t}")
                else:
                    # Reuse build_preview_cmd? Simpler inline
                    cmd = [
                        "ffmpeg", "-v", "error", "-ss", f"{safe_t:.3f}", "-i", str(video), "-t", f"{clip_duration:.3f}",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", str(clip_file)
                    ]
                    subprocess.run(cmd, capture_output=True)
            entry["clip"] = clip_file.name
        info_list.append(entry)
    data = {
        "video": video.name,
        "duration": duration,
        "threshold": threshold,
        "markers": info_list,
        "generated_at": time.time(),
    }
    out_json.write_text(json.dumps(data, indent=2))
    return data

# ---------------------------------------------------------------------------
# Codec scanning & batch transcoding
# ---------------------------------------------------------------------------

def probe_codecs(video: Path) -> dict:
    if os.environ.get("FFPROBE_DISABLE") or not ffprobe_available():
        return {"video": video.name, "vcodec": "h264", "vprofile": "high", "acodecs": ["aac"], "container": video.suffix.lstrip('.'), "duration": 0.0, "size": video.stat().st_size}
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile", "-show_entries", "format=duration", "-of", "json", str(video)
    ]
    proc_v = subprocess.run(cmd, capture_output=True, text=True)
    vcodec = None
    vprofile = None
    duration = 0.0
    try:
        data = json.loads(proc_v.stdout)
        if data.get("streams"):
            st = data["streams"][0]
            vcodec = st.get("codec_name")
            vprofile = (st.get("profile") or "").lower()
        if data.get("format"):
            duration = float(data["format"].get("duration", 0.0) or 0.0)
    except Exception:
        pass
    # audio codecs (may be multiple)
    cmd_a = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)]
    proc_a = subprocess.run(cmd_a, capture_output=True, text=True)
    acodecs = []
    for line in proc_a.stdout.splitlines():
        if line.strip():
            acodecs.append(line.strip())
    return {
        "video": video.name,
        "path": str(video),
        "vcodec": vcodec or "unknown",
        "vprofile": vprofile or "unknown",
        "acodecs": acodecs or ["unknown"],
        "container": video.suffix.lstrip('.'),
        "duration": duration,
        "size": video.stat().st_size,
    }


def codec_is_compatible(info: dict, target_v: str, target_a: str, profiles: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if info["vcodec"] != target_v:
        reasons.append(f"vcodec={info['vcodec']}")
    if target_v == "h264" and profiles and info.get("vprofile") not in profiles:
        reasons.append(f"profile={info.get('vprofile')}")
    # Accept if any audio stream matches
    if target_a != "copy" and target_a not in info.get("acodecs", []):
        reasons.append(f"acodecs={','.join(info.get('acodecs', []))}")
    return (len(reasons) == 0, reasons)


def build_transcode_cmd(src: Path, dst: Path, target_v: str, target_a: str, crf: int, v_bitrate: str | None, a_bitrate: str, preset: str, hw: str, drop_subtitles: bool) -> list[str]:
    # Decide video encoder
    if hw == "videotoolbox" and target_v == "h264":
        vencoder = "h264_videotoolbox"
        v_part = ["-c:v", vencoder]
        if v_bitrate:
            v_part += ["-b:v", v_bitrate]
        else:
            # videotoolbox often ignores CRF; supply bitrate default if not provided
            v_part += ["-b:v", "3000k"]
    else:
        vencoder = {"h264": "libx264", "hevc": "libx265", "av1": "libaom-av1"}.get(target_v, "libx264")
        v_part = ["-c:v", vencoder, "-preset", preset, "-crf", str(crf)]
        if v_bitrate:
            v_part += ["-b:v", v_bitrate]
    # Audio encoder
    if target_a == "copy":
        a_part = ["-c:a", "copy"]
    else:
        aencoder = {"aac": "aac", "opus": "libopus"}.get(target_a, "aac")
        a_part = ["-c:a", aencoder, "-b:a", a_bitrate]
    map_part: list[str] = []
    if drop_subtitles:
        map_part = ["-map", "0", "-map", "-0:s"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)] + map_part + v_part + a_part + ["-movflags", "+faststart", str(dst)]
    return cmd


def cmd_codecs(ns) -> int:
    """Scan videos for codec/profile compatibility vs desired target codecs."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    vids = find_videos(root, ns.recursive, {".mp4", ".mkv", ".mov", ".avi", ".m4v"})
    if not vids:
        print("No video files found.")
        return 0
    worker_count = max(1, min(ns.workers, 8))
    q: Queue[Path] = Queue()
    for v in vids:
        q.put(v)
    results: list[dict] = []
    lock = threading.Lock()
    done = 0
    total = len(vids)
    profiles = set(x.strip().lower() for x in ns.allowed_profiles.split(',') if x.strip()) if ns.allowed_profiles else set()

    def worker():
        nonlocal done
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            info = probe_codecs(v)
            ok, reasons = codec_is_compatible(info, ns.target_v, ns.target_a, profiles)
            info["compatible"] = ok
            if not ok:
                info["why"] = reasons
            with lock:
                results.append(info)
                done += 1
                print(f"codecs {done}/{total}\r", end="", file=sys.stderr)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"codecs {done}/{total}", file=sys.stderr)
    results.sort(key=lambda r: r["video"].lower())
    if ns.log:
        try:
            with open(ns.log, "w") as lf:
                for r in results:
                    if not r.get("compatible"):
                        lf.write(f"{r['path']} | v={r['vcodec']}({r['vprofile']}) | a={','.join(r['acodecs'])} | reasons={','.join(r.get('why', []))}\n")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to write log: {e}", file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": results, "target_v": ns.target_v, "target_a": ns.target_a}, indent=2))
    else:
        for r in results:
            if r["compatible"] and not ns.all:
                continue
            status = "✓" if r["compatible"] else "X"
            why = "" if r["compatible"] else (" [" + ",".join(r.get("why", [])) + "]")
            print(f"{status} {r['video']}: v={r['vcodec']}({r['vprofile']}) a={','.join(r['acodecs'])} c={r['container']}{why}")
    return 0


def cmd_transcode(ns) -> int:
    """Transcode incompatible videos toward target codecs (supports dry-run & progress)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    out_root = Path(ns.dest).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    vids = find_videos(root, ns.recursive, {".mp4", ".mkv", ".mov", ".avi", ".m4v"})
    if not vids:
        print("No video files found.")
        return 0
    profiles = set(x.strip().lower() for x in ns.allowed_profiles.split(',') if x.strip()) if ns.allowed_profiles else set()
    dry = ns.dry_run
    worker_count = max(1, min(ns.workers, 2))  # encoding heavy
    q: Queue[Path] = Queue()
    for v in vids: q.put(v)
    done = 0
    lock = threading.Lock()
    errors: list[str] = []
    actions: list[dict] = []

    def worker():
        nonlocal done
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            try:
                info = probe_codecs(v)
                ok, reasons = codec_is_compatible(info, ns.target_v, ns.target_a, profiles)
                rel = v.relative_to(root)
                out_file = out_root / rel
                out_file = out_file.with_suffix('.mp4')  # normalize container
                out_file.parent.mkdir(parents=True, exist_ok=True)
                if ok and not ns.force:
                    action = {"video": str(v), "status": "skip-compatible"}
                else:
                    if dry:
                        action = {"video": str(v), "status": "plan-transcode", "reasons": reasons, "output": str(out_file)}
                    else:
                        cmd = build_transcode_cmd(v, out_file, ns.target_v, ns.target_a, ns.crf, ns.v_bitrate, ns.a_bitrate, ns.preset, ns.hardware, ns.drop_subtitles)
                        start = time.time()
                        if ns.progress:
                            # Use -progress pipe:1 for periodic key=value lines; rebuild command
                            prog_cmd = ["ffmpeg", "-v", "error", "-i", str(v)]
                            if ns.drop_subtitles:
                                prog_cmd += ["-map", "0", "-map", "-0:s"]
                            # replicate encoding settings roughly
                            # Simplify: rely on previously built cmd parts excluding initial ffmpeg -y -v error -i and output
                            # For consistency re-use build_transcode_cmd but without -y and with -progress
                            full_cmd = cmd.copy()
                            # Insert -progress before output file path
                            full_cmd.insert(-1, "-progress")
                            full_cmd.insert(-1, "pipe:1")
                            proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                            duration = info.get("duration") or extract_duration_from_file(v) or 0.0
                            last_pct = -1
                            if proc.stdout:
                                for line in proc.stdout:
                                    line = line.strip()
                                    if line.startswith("out_time_ms=") and duration > 0:
                                        try:
                                            ms = float(line.split('=')[1])
                                            sec = ms / 1_000_000.0
                                            pct = int((sec / duration) * 100)
                                            if pct != last_pct:
                                                print(f"{v.name} {pct}%\r", end="", file=sys.stderr)
                                                last_pct = pct
                                        except Exception:
                                            pass
                            proc.wait()
                            if proc.returncode != 0:
                                err = proc.stderr.read() if proc.stderr else "ffmpeg failed"
                                raise RuntimeError(err.strip() or "ffmpeg failed")
                        else:
                            proc = subprocess.run(cmd, capture_output=True, text=True)
                            if proc.returncode != 0:
                                raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")
                        elapsed = time.time() - start
                        action = {"video": str(v), "status": "transcoded", "output": str(out_file), "seconds": round(elapsed,2)}
                with lock:
                    actions.append(action)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{v}: {e}")
            finally:
                with lock:
                    done += 1
                    print(f"transcode {done}/{len(vids)}\r", end="", file=sys.stderr)
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"transcode {done}/{len(vids)}", file=sys.stderr)
    summary = {"actions": actions, "errors": errors, "total": len(vids), "dry_run": dry, "target_v": ns.target_v, "target_a": ns.target_a}
    if ns.output_format == "json":
        print(json.dumps(summary, indent=2))
    else:
        for a in actions:
            print(a)
        if errors:
            print("Errors:")
            for e in errors: print("  ", e)
    return 1 if errors else 0


def cmd_compare(ns) -> int:
    """Compare two videos computing SSIM & PSNR quality metrics."""
    src = Path(ns.original).expanduser().resolve()
    dst = Path(ns.other).expanduser().resolve()
    if not src.exists() or not dst.exists():
        print("Error: input file(s) missing", file=sys.stderr)
        return 2
    if os.environ.get("FFPROBE_DISABLE") or not ffmpeg_available():
        print("FFmpeg not available (or disabled)")
        return 1
    cmd = ["ffmpeg", "-i", str(src), "-i", str(dst), "-lavfi", "[0][1]ssim;[0][1]psnr", "-f", "null", "-"]
    if ns.progress:
        # Need duration for percentage
        dur = extract_duration_from_file(src) or 0
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1)
        ssim = None
        psnr = None
        lines: list[str] = []
        if proc.stderr:
            for line in proc.stderr:
                line_strip = line.strip()
                if "frame=" in line_strip and "time=" in line_strip and dur > 0:
                    # parse time=HH:MM:SS.mmm
                    for token in line_strip.split():
                        if token.startswith("time="):
                            tval = token[5:]
                            try:
                                h, m, s = tval.split(':')
                                sec = float(h)*3600 + float(m)*60 + float(s)
                                pct = int((sec/dur)*100)
                                print(f"compare {pct}%\r", end="", file=sys.stderr)
                            except Exception:
                                pass
                lines.append(line_strip)
        proc.wait()
        stderr_text = "\n".join(lines)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr_text = proc.stderr
    ssim = None
    psnr = None
    for line in stderr_text.splitlines():
        if "SSIM Y:" in line and "All:" in line:
            # find All: value
            parts = line.replace(',', ' ').split()
            for p in parts:
                if p.startswith('All:'):
                    try: ssim = float(p.split(':')[1])
                    except Exception: pass
        if "PSNR y:" in line and "average:" in line:
            parts = line.replace(',', ' ').split()
            for p in parts:
                if p.startswith('average:'):
                    try: psnr = float(p.split(':')[1])
                    except Exception: pass
    interp_ssim = (
        "very-high" if (ssim or 0) >= 0.97 else
        "high" if (ssim or 0) >= 0.95 else
        "medium" if (ssim or 0) >= 0.90 else
        "low"
    ) if ssim is not None else None
    interp_psnr = (
        "excellent" if (psnr or 0) >= 40 else
        "good" if (psnr or 0) >= 35 else
        "acceptable" if (psnr or 0) >= 30 else
        "poor"
    ) if psnr is not None else None
    result = {"original": str(src), "other": str(dst), "ssim": ssim, "ssim_quality": interp_ssim, "psnr": psnr, "psnr_quality": interp_psnr}
    if ns.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(result)
    return 0




def cmd_faces_embed(ns) -> int:
    """Extract distinct face signatures per video (tracking + clustering + optional thumbnails)."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    results: list[dict] = []
    for v in videos:
        try:
            sigs = extract_distinct_face_signatures(
                v,
                sample_rate=ns.sample_rate,
                min_face_area=ns.min_face_area,
                blur_threshold=ns.blur_threshold,
                max_gap=ns.max_gap,
                min_track_frames=ns.min_track_frames,
                match_distance=ns.match_distance,
                cluster_eps=ns.cluster_eps,
                cluster_min_samples=ns.cluster_min_samples,
            )
            if ns.thumbnails and sigs and cv2 is not None:
                try:
                    cap = cv2.VideoCapture(str(v))
                    for sig in sigs:
                        mid_t = (sig["start_t"] + sig["end_t"]) / 2.0
                        cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        out_path = artifact_dir(v) / f"{v.stem}.face_{sig['id']}.jpg"
                        try: cv2.imwrite(str(out_path), frame)
                        except Exception: pass
                    cap.release()
                except Exception:
                    pass
            # Persist per-video signatures artifact
            try:
                sig_path = artifact_dir(v) / f"{v.stem}.faces.signatures.json"
                sig_payload = {"video": v.name, "count": len(sigs), "signatures": sigs, "generated_at": time.time()}
                sig_path.write_text(json.dumps(sig_payload, indent=2))
            except Exception:
                pass
            results.append({"video": v.name, "count": len(sigs), "signatures": sigs})
        except Exception as e:  # noqa: BLE001
            results.append({"video": v.name, "error": str(e)})
    if ns.output_format == "json":
        print(json.dumps({"results": results}, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['video']}: ERROR {r['error']}")
            else:
                print(f"{r['video']}: {r['count']} signatures")
    return 0


def cmd_faces_index(ns):
    """Build a global face index across videos (deduplicated people list).

    Returns the index dict (people, videos, etc.) instead of exit code for easier reuse by API.
    """
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    index = build_face_index(
        root,
        recursive=ns.recursive,
        sample_rate=ns.sample_rate,
        cluster_eps=ns.cluster_eps,
        cluster_min_samples=ns.cluster_min_samples,
    )
    # Optional supervised labeling via gallery directory
    if getattr(ns, "gallery", None):
        gal_root = Path(ns.gallery).expanduser().resolve()
        if gal_root.is_dir():
            gallery_embs: list[np.ndarray] = []
            gallery_labels: list[str] = []
            img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            vid_exts = {".mp4"}
            try:
                import face_recognition  # type: ignore
                import cv2  # type: ignore
            except Exception:
                face_recognition = None  # type: ignore
                cv2 = None  # type: ignore
            for person_dir in gal_root.iterdir():
                if not person_dir.is_dir():
                    continue
                label = person_dir.name
                for p in person_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    ext = p.suffix.lower()
                    if ext in img_exts and face_recognition is not None:
                        try:
                            import PIL.Image  # type: ignore
                            img = np.array(PIL.Image.open(p).convert("RGB"))
                            boxes = face_recognition.face_locations(img)
                            if not boxes:
                                continue
                            encs = face_recognition.face_encodings(img, boxes)
                            for enc in encs:
                                gallery_embs.append(enc.astype("float32"))
                                gallery_labels.append(label)
                        except Exception:
                            pass
                    elif ext in vid_exts and face_recognition is not None and cv2 is not None:
                        cap = cv2.VideoCapture(str(p))
                        if not cap.isOpened():
                            continue
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        step = max(int(round(fps / max(ns.gallery_sample_rate, 0.1))), 1)
                        fidx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if fidx % step == 0:
                                rgb = frame[:, :, ::-1]
                                try:
                                    boxes = face_recognition.face_locations(rgb)
                                    if boxes:
                                        encs = face_recognition.face_encodings(rgb, boxes)
                                        for enc in encs:
                                            gallery_embs.append(enc.astype("float32"))
                                            gallery_labels.append(label)
                                except Exception:
                                    pass
                            fidx += 1
                        cap.release()
            if gallery_embs:
                gal_arr = np.vstack(gallery_embs).astype("float32")
                gal_arr_norm = gal_arr / (np.linalg.norm(gal_arr, axis=1, keepdims=True) + 1e-12)
                for person in index.get("people", []):
                    emb = np.array(person.get("embedding", []), dtype="float32")
                    if emb.size == 0:
                        continue
                    emb = emb / (np.linalg.norm(emb) or 1.0)
                    dists = np.linalg.norm(gal_arr_norm - emb[None, :], axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] <= ns.label_threshold:
                        person["label"] = gallery_labels[j]
                        person["label_distance"] = float(dists[j])
    if ns.output_format == "json":
        text = json.dumps(index, indent=2)
        if getattr(ns, "output", None):
            Path(ns.output).write_text(text)
        else:
            print(text)
    else:
        print(f"people={len(index.get('people', []))} videos={index.get('videos')}")
        for person in index.get("people", [])[:20]:
            print(person["id"], len(person.get("occurrences", [])))
    return index


def cmd_report(ns) -> int:
    """Summarize artifact coverage (counts & percentages) across videos."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    videos.sort()
    total = len(videos)
    if total == 0:
        # Attempt auto-recursive discovery if user omitted -r
        if not ns.recursive:
            # Quick probe: first few matches of mp4 or artifact signature
            found_any = False
            probe_limit = 25
            mp4s = []
            for i, p in enumerate(root.rglob('*.mp4')):
                mp4s.append(p)
                if i >= probe_limit:
                    break
            if mp4s:
                print("[report] Auto-enabling recursive scan (videos found in subdirectories)", file=sys.stderr)
                ns.recursive = True
                videos = find_mp4s(root, True)
                videos.sort()
                total = len(videos)
        # If still zero, short-circuit
        if total == 0:
            print("No MP4 files found.")
            return 0
    # Artifact detectors
    counts = {
        "metadata": 0,
        "thumbs": 0,
        "sprites": 0,
        "previews": 0,
        "subtitles": 0,
        "phash": 0,
        "heatmaps": 0,
        "scenes": 0,
        "faces": 0,
    }
    rows: list[dict] = []
    for v in videos:
        metadata_ok = metadata_path(v).exists()
        thumb_ok = thumbs_path(v).exists()
        s, j = sprite_sheet_paths(v)
        sprites_ok = s.exists() and j.exists()
        previews_ok = preview_index_path(v).exists()
        subtitles_ok = find_subtitles(v) is not None
        phash_ok = phash_path(v).exists()
        heatmap_ok = heatmaps_json_path(v).exists()
        scenes_ok = scenes_json_path(v).exists()
        faces_ok = faces_json_path(v).exists()
        if metadata_ok: counts["metadata"] += 1
        if thumb_ok: counts["thumbs"] += 1
        if sprites_ok: counts["sprites"] += 1
        if previews_ok: counts["previews"] += 1
        if subtitles_ok: counts["subtitles"] += 1
        if phash_ok: counts["phash"] += 1
        if heatmap_ok: counts["heatmaps"] += 1
        if scenes_ok: counts["scenes"] += 1
        if faces_ok: counts["faces"] += 1
        rows.append({
            "video": v.name,
            "metadata": metadata_ok,
            "thumbs": thumb_ok,
            "sprites": sprites_ok,
            "previews": previews_ok,
            "subtitles": subtitles_ok,
            "phash": phash_ok,
            "heatmaps": heatmap_ok,
            "scenes": scenes_ok,
            "faces": faces_ok,
        })
    coverage = {k: (counts[k] / total) for k in counts}
    if ns.output_format == "json":
        print(json.dumps({
            "total": total,
            "coverage": coverage,
            "counts": counts,
            "videos": rows,
        }, indent=2))
    else:
        print(f"Total videos: {total}")
        for k in counts:
            pct = (counts[k] / total) * 100
            print(f"{k:9s}: {counts[k]:4d} ({pct:5.1f}%)")
    return 0


def cmd_rename(ns) -> int:
    rename_with_artifacts(Path(ns.src), Path(ns.dst))
    return 0


def generate_report(root: Path, recursive: bool) -> dict:
    """Generate artifact coverage report for videos in a directory."""
    videos = find_mp4s(root, recursive)
    total = len(videos)
    counts = {
        "metadata": 0,
        "thumbs": 0,
        "sprites": 0,
        "previews": 0,
        "subtitles": 0,
        "phash": 0,
        "heatmaps": 0,
        "scenes": 0,
        "faces": 0,
    }
    for v in videos:
        if metadata_path(v).exists(): counts["metadata"] += 1
        if thumbs_path(v).exists(): counts["thumbs"] += 1
        s, j = sprite_sheet_paths(v)
        if s.exists() and j.exists(): counts["sprites"] += 1
        if preview_index_path(v).exists(): counts["previews"] += 1
        if find_subtitles(v) is not None: counts["subtitles"] += 1
        if phash_path(v).exists(): counts["phash"] += 1
        if heatmaps_json_path(v).exists(): counts["heatmaps"] += 1
        if scenes_json_path(v).exists(): counts["scenes"] += 1
        if faces_json_path(v).exists(): counts["faces"] += 1
    return {"total": total, "counts": counts}

def cmd_finish(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    rep = generate_report(root, ns.recursive)
    total = rep.get("total", 0)
    counts = rep.get("counts", {})
    mapping = [
        ("metadata", "metadata"),
        ("thumbs", "thumbs"),
        ("sprites", "sprites"),
        ("previews", "previews"),
        ("subtitles", "subtitles"),
        ("phash", "phash"),
        ("heatmaps", "heatmaps"),
        ("scenes", "scenes"),
        ("faces", "faces"),
    ]
    tasks = [t for k, t in mapping if counts.get(k, 0) < total]
    if not tasks:
        print("All artifacts present")
        return 0
    ns_batch = SimpleNamespace(
        directory=str(root),
        recursive=ns.recursive,
        tasks=",".join(tasks),
        max_meta=ns.workers,
        max_thumbs=ns.workers,
        max_sprites=ns.workers,
        max_previews=ns.workers,
        max_subtitles=1,
        max_phash=ns.workers,
        max_scenes=1,
        preview_width=320,
        preview_duration=1.0,
        preview_segments=9,
        sprites_interval=10.0,
        sprites_width=320,
        sprites_cols=10,
        sprites_rows=10,
        subtitles_model="small",
        subtitles_backend="auto",
        subtitles_format="srt",
        subtitles_language=None,
        subtitles_translate=False,
        force=False,
    )
    return cmd_batch(ns_batch)
    # The generate_report function is similar to the logic in cmd_report, but is used internally for the "finish" command to compute which artifact tasks are incomplete. It is not strictly necessary if you refactor cmd_report to return its summary instead of printing, but currently both exist for separation of CLI output and internal logic.

def cmd_metadata(ns) -> int:
    """Generate ffprobe metadata JSON sidecars for each video."""
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    if not ffprobe_available():
        print("Warning: ffprobe not found; set FFPROBE_DISABLE=1 to stub or install ffprobe", file=sys.stderr)
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    result = generate_metadata(videos, force=ns.force, workers=ns.workers)
    if result["errors"]:
        print("Errors (some files may be incomplete):", file=sys.stderr)
        for e in result["errors"]:
            print("  " + e, file=sys.stderr)
    print(f"Metadata done for {result['total']} file(s)")
    return 0

def resolve_directory(ns):
    """Consolidate logic for setting ns.directory from ns.dir."""
    if getattr(ns, "dir", None):
        # If ns.directory is missing or set to "." or None, use ns.dir
        if not hasattr(ns, "directory") or ns.directory == "." or ns.directory is None:
            ns.directory = ns.dir
    return ns

def main(argv: List[str] | None = None) -> int:
    raw_args = argv or sys.argv[1:]
    ns = parse_args(raw_args)
    # Fallback: user supplied directory then command (e.g. './index.py /path orphans')
    command_names = {"list","report","orphans","metadata","thumbs","sprites","previews","subtitles","phash","dupes","heatmaps","scenes","embed","listing","rename","codecs","transcode","compare","batch","finish"}
    if ns.cmd not in command_names and raw_args:
        first = raw_args[0]
        if Path(first).exists() and len(raw_args) > 1 and raw_args[1] in command_names:
            # Reinterpret as path-first; rebuild argv swapping first two
            reordered = [raw_args[1], first] + raw_args[2:]
            ns = parse_args(reordered)
            ns = resolve_directory(ns)
            print(f"(note) path-first invocation detected; treating '{first}' as directory and '{raw_args[1]}' as command. Preferred usage: {Path(sys.argv[0]).name} {raw_args[1]} {first}", file=sys.stderr)
    ns = resolve_directory(ns)

    if ns.cmd == "list": return cmd_list(ns)
    if ns.cmd == "report": return cmd_report(ns)
    if ns.cmd == "orphans": return cmd_orphans(ns)
    if ns.cmd == "batch": return cmd_batch(ns)
    if ns.cmd == "finish": return cmd_finish(ns)
    if ns.cmd == "rename": return cmd_rename(ns)

    if ns.cmd == "metadata": return cmd_metadata(ns)
    if ns.cmd == "thumbs": return cmd_thumbs(ns)
    if ns.cmd == "sprites": return cmd_sprites(ns)
    if ns.cmd == "previews": return cmd_previews(ns)
    if ns.cmd == "subtitles": return cmd_subtitles(ns)
    if ns.cmd == "heatmaps": return cmd_heatmaps(ns)
    if ns.cmd == "scenes": return cmd_scenes(ns)
    if ns.cmd == "phash": return cmd_phash(ns)
    if ns.cmd == "dupes": return cmd_phash_dupes(ns)
    if ns.cmd == "codecs": return cmd_codecs(ns)
    if ns.cmd == "transcode": return cmd_transcode(ns)
    if ns.cmd == "compare": return cmd_compare(ns)
    
    if ns.cmd == "embed": return cmd_faces_embed(ns)
    if ns.cmd == "listing":
        cmd_faces_index(ns)  # returns dict; CLI expects exit code so ignore value
        return 0

    print("Unknown command", file=sys.stderr)
    return 1

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
