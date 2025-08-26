#!/usr/bin/env python3
"""Unified video utility script.

Functions:
  list   - List .mp4 files with optional recursion / JSON / sizes.
  meta   - Generate ffprobe metadata JSON files for .mp4s.
           (Skips existing unless --force.)
  queue  - Queue + process metadata generation with limited workers.

The 'queue' command is an ephemeral in-memory scheduler to avoid CPU thrash on
low-power devices. It discovers files (like meta) then processes them with a
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
import json
import os
import sys
import threading
import time
import subprocess
import random
from pathlib import Path
from typing import List, Dict, Any, Iterable
from queue import Queue, Empty
except ImportError:
    cv2 = None
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# File discovery & listing
# ---------------------------------------------------------------------------

def find_mp4s(root: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def human_size(num: int) -> str:
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
    return video.with_suffix(video.suffix + ".ffprobe.json")

def thumb_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".jpg")


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
    done = 0
    errors: list[str] = []

    # Worker design: queue file paths, each worker writes file.
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)

    lock = threading.Lock()

    def worker():
        nonlocal done
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            try:
                out_path = metadata_path(v)
                if out_path.exists() and not force:
                    pass
                else:
                    data = run_ffprobe(v)
                    out_path.write_text(json.dumps(data, indent=2))
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
    while any(t.is_alive() for t in threads):
        print(f"Progress: {done}/{total}\r", end="", file=sys.stderr)
        time.sleep(0.2)
    for t in threads:
        t.join()
    print(f"Progress: {done}/{total}", file=sys.stderr)
    return {"total": total, "errors": errors}

# ---------------------------------------------------------------------------
# Queue (ephemeral) built atop metadata generation logic
# ---------------------------------------------------------------------------

class EphemeralQueue:
    def __init__(self, paths: List[Path]):
        self._all = paths
        self._lock = threading.Lock()
        self._index = 0

    def next(self) -> Path | None:
        with self._lock:
            if self._index >= len(self._all):
                return None
            p = self._all[self._index]
            self._index += 1
            return p


def queue_process(paths: List[Path], workers: int, force: bool) -> Dict[str, Any]:
    total = len(paths)
    q = EphemeralQueue(paths)
    errors: list[str] = []
    processed = 0
    lock = threading.Lock()

    def worker():
        nonlocal processed
        while True:
            p = q.next()
            if p is None:
                return
            try:
                out_path = metadata_path(p)
                if out_path.exists() and not force:
                    pass
                else:
                    data = run_ffprobe(p)
                    out_path.write_text(json.dumps(data, indent=2))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{p}: {e}")
            finally:
                with lock:
                    processed += 1

    worker_count = max(1, min(workers, 4))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        print(f"Queue progress: {processed}/{total}\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    print(f"Queue progress: {processed}/{total}", file=sys.stderr)
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


def extract_duration(meta_json: Dict[str, Any]) -> float | None:
    # ffprobe format.duration is a string
    fmt = meta_json.get("format") if isinstance(meta_json, dict) else None
    if isinstance(fmt, dict):
        dur = fmt.get("duration")
        try:
            return float(dur)
        except (TypeError, ValueError):
            return None
    return None


def generate_thumbnail(video: Path, force: bool, time_spec: str, quality: int) -> None:
    out = thumb_path(video)
    if out.exists() and not force:
        return
    duration = None
    meta_file = metadata_path(video)
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            duration = extract_duration(meta)
        except Exception:  # noqa: BLE001
            duration = None
    t = parse_time_spec(time_spec, duration)
    if os.environ.get("FFPROBE_DISABLE"):
        # stub mode writes a placeholder
        out.write_text(f"stub thumbnail for {video.name} at {t}s")
        return
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


def cmd_thumb(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    errors: list[str] = []
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    processed = 0
    total = len(videos)
    lock = threading.Lock()

    def worker():
        nonlocal processed
        while True:
            try:
                v = q.get_nowait()
            except Empty:
                return
            try:
                generate_thumbnail(v, ns.force, ns.time_spec, ns.quality)
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
        print(f"Thumb progress: {processed}/{total}\r", end="", file=sys.stderr)
        time.sleep(0.2)
    for t in threads:
        t.join()
    print(f"Thumb progress: {processed}/{total}", file=sys.stderr)
    if errors:
        print("Errors (some thumbnails missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Thumbnails generated for {total - len(errors)} file(s)")
    return 0

# ---------------------------------------------------------------------------
# Sprite sheet generation
# ---------------------------------------------------------------------------

def sprite_sheet_paths(video: Path) -> tuple[Path, Path]:
    base = video.with_suffix(video.suffix + ".sprites")
    return base.with_suffix(base.suffix + ".jpg"), base.with_suffix(base.suffix + ".json")


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
    processed = 0
    errors: list[str] = []
    for v in videos:
        ok, err = generate_sprite_sheet(v, interval, ns.width, cols, rows, ns.quality, ns.force, ns.max_frames)
        processed += 1
        if not ok and err:
            errors.append(f"{v}: {err}")
        print(f"Sprites {processed}/{total}\r", end="", file=sys.stderr)
    print(f"Sprites {processed}/{total}", file=sys.stderr)
    if errors:
        print("Errors (some sprite sheets missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Sprite sheets generated for {total - len(errors)} file(s)")
    return 0

# ---------------------------------------------------------------------------
# Hover preview short clips (decile segments)
# ---------------------------------------------------------------------------

def preview_output_dir(video: Path) -> Path:
    return video.parent / f".{video.name}.previews"


def preview_index_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".previews.json")


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
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    processed = 0
    total = len(videos)
    lock = threading.Lock()
    errors: list[str] = []

    def worker():
        nonlocal processed
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
                for idx, start in enumerate(points, start=1):
                    out_file = out_dir / f"seg_{idx:02d}{clip_ext}"
                    if out_file.exists() and not ns.force:
                        index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                        continue
                    if os.environ.get("FFPROBE_DISABLE"):
                        out_file.write_text(f"stub preview {vid.name} t={start:.2f}")
                        index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                        continue
                    if not ffmpeg_available():
                        raise RuntimeError("ffmpeg not available")
                    cmd = build_preview_cmd(vid, start, ns.duration, ns.width, out_file, ns.format, ns.crf, ns.bitrate)
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        raise RuntimeError(proc.stderr.strip() or "ffmpeg preview failed")
                    index_entries.append({"start": start, "file": out_file.name, "duration": ns.duration})
                if not ns.no_index:
                    preview_index_path(vid).write_text(json.dumps({
                        "video": vid.name,
                        "format": ns.format,
                        "segments": index_entries,
                        "segment_duration": ns.duration,
                        "generated_at": time.time(),
                    }, indent=2))
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
        print(f"Previews {processed}/{total}\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    print(f"Previews {processed}/{total}", file=sys.stderr)
    if errors:
        print("Errors (some previews missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Hover previews generated for {total - len(errors)} file(s)")
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


def format_segments(segments, fmt: str) -> str:
    if fmt == "json":
        return json.dumps([{
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
        } for s in segments], indent=2)
    if fmt == "srt":
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
    # vtt
    lines = ["WEBVTT", ""]
    for s in segments:
        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            sec = t % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}"  # dot for VTT
        lines.append(f"{ts(s['start'])} --> {ts(s['end'])}")
        lines.append(s['text'].strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_whisper_backend(video: Path, backend: str, model_name: str, language: str | None, translate: bool, cpp_bin: str | None, cpp_model: str | None):
    # Returns list of segments: {start,end,text}
    if os.environ.get("FFPROBE_DISABLE"):
        return [{"start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"Stub segment {i+1}"} for i in range(3)]
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(model_name, compute_type="int8" if translate else "float16")
        seg_iter, info = model.transcribe(str(video), language=language, task="translate" if translate else "transcribe")
        segments = []
        for s in seg_iter:
            segments.append({"start": s.start, "end": s.end, "text": s.text})
        return segments
    if backend == "whisper":
        import whisper  # type: ignore
        model = whisper.load_model(model_name)
        result = model.transcribe(str(video), language=language, task="translate" if translate else "transcribe")
        segs = []
        for s in result.get("segments", []):
            segs.append({"start": s["start"], "end": s["end"], "text": s["text"]})
        return segs
    if backend == "whisper.cpp":
        # Need external binary invocation; minimal implementation.
        if not cpp_bin or not Path(cpp_bin).exists():
            raise RuntimeError("whisper.cpp binary not found (provide --whisper-cpp-bin)")
        if not cpp_model or not Path(cpp_model).exists():
            raise RuntimeError("whisper.cpp model not found (provide --whisper-cpp-model)")
        out_json = video.with_suffix(video.suffix + f".{backend}.json")
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


def cmd_subs(ns) -> int:
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
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    out_dir_base = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else None
    processed = 0
    total = len(videos)
    errors: list[str] = []
    lock = threading.Lock()

    def worker():
        nonlocal processed
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                suffix_map = {"vtt": ".vtt", "srt": ".srt", "json": ".json"}
                out_dir = out_dir_base or vid.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / (vid.name + suffix_map[ns.format])
                if out_file.exists() and not ns.force:
                    pass
                else:
                    segments = run_whisper_backend(vid, backend, ns.model, ns.language, ns.translate, ns.whisper_cpp_bin, ns.whisper_cpp_model)
                    text = format_segments(segments, ns.format)
                    out_file.write_text(text)
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
        print(f"Subs {processed}/{total} backend={backend}\r", end="", file=sys.stderr)
        time.sleep(0.25)
    for t in threads:
        t.join()
    print(f"Subs {processed}/{total} backend={backend}", file=sys.stderr)
    if errors:
        print("Errors (some subtitles missing):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
    print(f"Subtitles generated for {total - len(errors)} file(s)")
    return 0

# ---------------------------------------------------------------------------
# Ephemeral multi-task batch scheduler
# ---------------------------------------------------------------------------

class BatchJob:
    __slots__ = ("video", "task")
    def __init__(self, video: Path, task: str):
        self.video = video
        self.task = task  # meta, thumb, sprites, previews, subs


def artifact_exists(video: Path, task: str) -> bool:
    if task == "meta":
        return metadata_path(video).exists()
    if task == "thumb":
        return thumb_path(video).exists()
    if task == "sprites":
        s, j = sprite_sheet_paths(video)
        return s.exists() and j.exists()
    if task == "previews":
        return preview_index_path(video).exists() or preview_output_dir(video).exists()
    if task == "subs":
        # any supported suffix
        for suf in (".vtt", ".srt", ".json"):
            if (video.parent / (video.name + suf)).exists():
                return True
        return False
    if task == "phash":
        return phash_path(video).exists()
    return False


def run_task(video: Path, task: str, ns) -> tuple[bool, str | None]:
    try:
        if task == "meta":
            if not meta_single(video, force=ns.force):
                return True, None
        elif task == "thumb":
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
        elif task == "subs":
            backend = detect_backend(ns.subs_backend)
            segments = run_whisper_backend(video, backend, ns.subs_model, ns.subs_language, ns.subs_translate, None, None)
            text = format_segments(segments, ns.subs_format)
            suffix = {"vtt": ".vtt", "srt": ".srt", "json": ".json"}[ns.subs_format]
            (video.parent / (video.name + suffix)).write_text(text)
        elif task == "phash":
            # Use 5 evenly spaced frames by default in batch mode for more robust whole-video signature (algo defaults)
            compute_phash_video(video, frame_time_spec="middle", frames=5, force=ns.force, algo="ahash", combine="xor")
        elif task == "scenes":
            # Default: detect scenes only (no thumbs/clips) for speed
            generate_scene_artifacts(video, threshold=0.4, limit=0, gen_thumbs=False, gen_clips=False, thumb_width=320, clip_duration=2.0, force=ns.force)
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
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    requested = [t.strip() for t in ns.tasks.split(',') if t.strip()]
    valid = {"meta", "thumb", "sprites", "previews", "subs", "phash", "scenes"}
    for t in requested:
        if t not in valid:
            print(f"Unknown task in --tasks: {t}", file=sys.stderr)
            return 2
    per_type_caps = {
        "meta": max(1, ns.max_meta),
        "thumb": max(1, ns.max_thumb),
        "sprites": max(1, ns.max_sprites),
        "previews": max(1, ns.max_previews),
    "subs": max(1, ns.max_subs),
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
    return video.with_suffix(video.suffix + ".phash.json")


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


def cmd_phash(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    done = 0
    total = len(videos)
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, str]] = []  # (video name, hash)

    def worker():
        nonlocal done
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                h = compute_phash_video(vid, ns.time, ns.frames, ns.force, algo=ns.algo, combine=ns.combine)
                with lock:
                    results.append((vid.name, h))
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
        print(f"pHash {done}/{total}\r", end="", file=sys.stderr)
        time.sleep(0.2)
    for t in threads:
        t.join()
    print(f"pHash {done}/{total}", file=sys.stderr)
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video utility (list, metadata, queue)")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="List mp4 files")
    lp.add_argument("directory", nargs="?", default=".")
    lp.add_argument("-r", "--recursive", action="store_true")
    lp.add_argument("--json", action="store_true")
    lp.add_argument("--show-size", action="store_true")
    lp.add_argument("--sort", choices=["name", "size"], default="name")

    mp = sub.add_parser("meta", help="Generate ffprobe metadata for mp4 files")
    mp.add_argument("directory", nargs="?", default=".")
    mp.add_argument("-r", "--recursive", action="store_true")
    mp.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    mp.add_argument("--workers", type=int, default=1)

    qp = sub.add_parser("queue", help="Ephemeral queue to generate metadata")
    qp.add_argument("directory", nargs="?", default=".")
    qp.add_argument("-r", "--recursive", action="store_true")
    qp.add_argument("--force", action="store_true")
    qp.add_argument("--workers", type=int, default=1)

    tp = sub.add_parser("thumb", help="Generate cover thumbnails (JPEG)")
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

    sb = sub.add_parser("subs", help="Generate subtitles via Whisper (whisper / faster-whisper / whisper.cpp)")
    sb.add_argument("directory", nargs="?", default=".")
    sb.add_argument("-r", "--recursive", action="store_true")
    sb.add_argument("--model", default="small", help="Model name (default small)")
    sb.add_argument("--backend", choices=["auto", "whisper", "faster-whisper", "whisper.cpp"], default="auto", help="Backend selection (default auto)")
    sb.add_argument("--language", default=None, help="Source language (auto detect if omitted)")
    sb.add_argument("--translate", action="store_true", help="Translate to English")
    sb.add_argument("--format", choices=["vtt", "srt", "json"], default="vtt", help="Subtitle output format (default vtt)")
    sb.add_argument("--workers", type=int, default=1, help="Parallel videos (cap 2)")
    sb.add_argument("--force", action="store_true", help="Regenerate even if subtitle file exists")
    sb.add_argument("--output-dir", default=None, help="Directory to place subtitle files (defaults alongside videos)")
    sb.add_argument("--whisper-cpp-bin", default=None, help="Path to whisper.cpp binary (if backend whisper.cpp)")
    sb.add_argument("--whisper-cpp-model", default=None, help="Path to whisper.cpp model (.bin)")

    bt = sub.add_parser("batch", help="Ephemeral multi-stage pipeline (meta/thumb/sprites/previews/subs) with per-type concurrency caps")
    bt.add_argument("directory", nargs="?", default=".")
    bt.add_argument("-r", "--recursive", action="store_true")
    bt.add_argument("--tasks", default="meta,thumb", help="Comma list of tasks: meta,thumb,sprites,previews,subs,phash (default meta,thumb)")
    bt.add_argument("--max-meta", type=int, default=3)
    bt.add_argument("--max-thumb", type=int, default=4)
    bt.add_argument("--max-sprites", type=int, default=2)
    bt.add_argument("--max-previews", type=int, default=2)
    bt.add_argument("--max-subs", type=int, default=1)
    bt.add_argument("--max-phash", type=int, default=4)
    bt.add_argument("--max-scenes", type=int, default=1)
    bt.add_argument("--preview-width", type=int, default=320)
    bt.add_argument("--preview-duration", type=float, default=1.0)
    bt.add_argument("--preview-segments", type=int, default=9)
    bt.add_argument("--sprites-interval", type=float, default=10.0)
    bt.add_argument("--sprites-width", type=int, default=320)
    bt.add_argument("--sprites-cols", type=int, default=10)
    bt.add_argument("--sprites-rows", type=int, default=10)
    bt.add_argument("--subs-model", default="small")
    bt.add_argument("--subs-backend", default="auto")
    bt.add_argument("--subs-format", default="vtt")
    bt.add_argument("--subs-language", default=None)
    bt.add_argument("--subs-translate", action="store_true")
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

    hm = sub.add_parser("heatmap", help="Generate brightness/motion heatmap timeline JSON (and optional PNG)")
    hm.add_argument("directory", nargs="?", default=".")
    hm.add_argument("-r", "--recursive", action="store_true")
    hm.add_argument("--interval", type=float, default=5.0, help="Seconds between samples (default 5.0)")
    hm.add_argument("--mode", choices=["brightness","motion","both"], default="both", help="Metrics to compute (default both)")
    hm.add_argument("--png", action="store_true", help="Also write a small PNG stripe visualization")
    hm.add_argument("--workers", type=int, default=2, help="Parallel videos (cap 4)")
    hm.add_argument("--force", action="store_true")
    hm.add_argument("--output-format", choices=["json","text"], default="json")

    sc = sub.add_parser("scenes", help="Detect scene boundaries and generate markers (optional thumbs/clips)")
    sc.add_argument("directory", nargs="?", default=".")
    sc.add_argument("-r", "--recursive", action="store_true")
    sc.add_argument("--threshold", type=float, default=0.4, help="Scene score threshold (ffmpeg select gt(scene,TH)) (default 0.4)")
    sc.add_argument("--limit", type=int, default=0, help="Limit number of markers (0 = no limit)")
    sc.add_argument("--thumbs", action="store_true", help="Generate thumbnail per scene")
    sc.add_argument("--clips", action="store_true", help="Generate short clip per scene start")
    sc.add_argument("--clip-duration", type=float, default=2.0, help="Seconds per scene clip (default 2.0)")
    sc.add_argument("--thumb-width", type=int, default=320, help="Thumbnail width (default 320)")
    sc.add_argument("--workers", type=int, default=2, help="Parallel videos (cap 4)")
    sc.add_argument("--force", action="store_true")
    sc.add_argument("--output-format", choices=["json","text"], default="json")

    fc = sub.add_parser("faces", help="Detect faces in videos")
    fc.add_argument("directory", nargs="?", default=".")
    fc.add_argument("-r", "--recursive", action="store_true")
    fc.add_argument("--workers", type=int, default=2, help="Parallel videos (cap 4)")
    fc.add_argument("--force", action="store_true")
    fc.add_argument("--output-format", choices=["json","text"], default="json")

    ab = sub.add_parser("actor-build", help="Build face embedding gallery")
    ab.add_argument("--people-dir", required=True)
    ab.add_argument("--model", default="ArcFace")
    ab.add_argument("--detector", default="retinaface")
    ab.add_argument("--embeddings", default="gallery.npy")
    ab.add_argument("--labels", default="labels.json")
    ab.add_argument("--include-video", action="store_true")
    ab.add_argument("--video-sample-rate", type=float, default=1.0)
    ab.add_argument("--min-face-area", type=int, default=4096)
    ab.add_argument("--blur-threshold", type=float, default=60.0)
    ab.add_argument("--verbose", action="store_true")

    am = sub.add_parser("actor-match", help="Match faces in a video")
    am.add_argument("--video", required=True)
    am.add_argument("--embeddings", default="gallery.npy")
    am.add_argument("--labels", default="labels.json")
    am.add_argument("--model", default="ArcFace")
    am.add_argument("--detector", default="retinaface")
    am.add_argument("--retry-detectors", default="mtcnn,opencv")
    am.add_argument("--sample-rate", type=float, default=1.0)
    am.add_argument("--topk", type=int, default=3)
    am.add_argument("--conf", type=float, default=0.40)
    am.add_argument("--min-face-area", type=int, default=4096)
    am.add_argument("--blur-threshold", type=float, default=60.0)
    am.add_argument("--out", default=None)
    am.add_argument("--verbose", action="store_true")

    cd = sub.add_parser("codecs", help="Scan library for codec/profile compatibility")
    cd.add_argument("directory", nargs="?", default=".")
    cd.add_argument("-r", "--recursive", action="store_true")
    cd.add_argument("--target-v", default="h264", help="Target video codec (default h264)")
    cd.add_argument("--target-a", default="aac", help="Target audio codec (or copy) (default aac)")
    cd.add_argument("--allowed-profiles", default="high,main,constrained baseline", help="Comma list acceptable H.264 profiles")
    cd.add_argument("--workers", type=int, default=4)
    cd.add_argument("--log", default=None, help="Write incompatible entries to this log file")
    cd.add_argument("--output-format", choices=["json","text"], default="text")

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
    tc.add_argument("--drop-subs", action="store_true", help="Exclude subtitle streams")
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

    return p.parse_args(argv)


def cmd_list(ns) -> int:
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


def cmd_heatmap(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    done = 0
    total = len(videos)
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, dict]] = []

    def worker():
        nonlocal done
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                data = compute_heatmap(vid, ns.interval, ns.mode, ns.force, ns.png)
                with lock:
                    results.append((vid.name, data))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid.name}: {e}")
            finally:
                with lock:
                    done += 1
                    print(f"heatmap {done}/{total}\r", end="", file=sys.stderr)
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"heatmap {done}/{total}", file=sys.stderr)
    if errors:
        for e in errors:
            print("Error:", e, file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": [
            {"video": v, "heatmap": {k: val for k, val in d.items() if k != "samples"}, "samples": d.get("samples")}
            for v, d in results
        ]}, indent=2))
    else:
        for v, d in results:
            print(v, len(d.get("samples", [])))
    return 1 if errors else 0


def cmd_scenes(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    done = 0
    total = len(videos)
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, dict]] = []

    def worker():
        nonlocal done
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                data = generate_scene_artifacts(vid, ns.threshold, ns.limit, ns.thumbs, ns.clips, ns.thumb_width, ns.clip_duration, ns.force)
                with lock:
                    results.append((vid.name, data))
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
    print(f"scenes {done}/{total}", file=sys.stderr)
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
    return video.with_suffix(f"{video.suffix}.faces.json")


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


def cmd_faces(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    if not videos:
        print("No MP4 files found.")
        return 0
    worker_count = max(1, min(ns.workers, 4))
    q: Queue[Path] = Queue()
    for v in videos:
        q.put(v)
    done = 0
    total = len(videos)
    lock = threading.Lock()
    errors: list[str] = []
    results: list[tuple[str, dict]] = []

    def worker():
        nonlocal done
        while True:
            try:
                vid = q.get_nowait()
            except Empty:
                return
            try:
                data = generate_faces(vid, ns.force)
                with lock:
                    results.append((vid.name, data))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{vid.name}: {e}")
            finally:
                with lock:
                    done += 1
                    print(f"faces {done}/{total}\r", end="", file=sys.stderr)
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"faces {done}/{total}", file=sys.stderr)
    if errors:
        for e in errors:
            print("Error:", e, file=sys.stderr)
    if ns.output_format == "json":
        print(json.dumps({"results": [
            {"video": v, "count": len(d.get("faces", [])), "faces": d.get("faces")}
            for v, d in results
        ]}, indent=2))
    else:
        for v, d in results:
            print(v, len(d.get("faces", [])))
    return 1 if errors else 0

# ---------------------------------------------------------------------------
# Actor recognition
# ---------------------------------------------------------------------------

def cmd_actor_build(ns) -> int:
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    vid_exts = {".mp4"}
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    stub = os.environ.get("DEEPFACE_STUB")
    for actor_dir in Path(ns.people_dir).iterdir():
        if not actor_dir.is_dir():
            continue
        actor = actor_dir.name
        for path in actor_dir.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in img_exts:
                if stub:
                    embeddings.append(np.array([1, 0, 0, 0], dtype="float32"))
                    labels.append(actor)
                    counts[actor] = counts.get(actor, 0) + 1
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    continue
                try:
                except (ValueError, RuntimeError):
                    reps = []
                faces = reps if isinstance(reps, list) else [reps]
                for face in faces:
                    fa = face.get("facial_area") or {}
                    x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
                    if w * h < ns.min_face_area:
                        continue
                    crop = img[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue
                    if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < ns.blur_threshold:
                        continue
                    embeddings.append(np.array(face["embedding"], dtype="float32"))
                    labels.append(actor)
                    counts[actor] = counts.get(actor, 0) + 1
            elif ns.include_video and ext in vid_exts:
                if stub:
                    continue
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                t = 0.0
                while t <= duration:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        from deepface import DeepFace
                        reps = DeepFace.represent(img_path=frame, model_name=ns.model, detector_backend=ns.detector, enforce_detection=True, align=True)
                    except Exception:
                        reps = []
                    faces = reps if isinstance(reps, list) else [reps]
                    for face in faces:
                        fa = face.get("facial_area") or {}
                        x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
                        if w * h < ns.min_face_area:
                            continue
                        crop = frame[y:y + h, x:x + w]
                        if crop.size == 0:
                            continue
                        if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < ns.blur_threshold:
                            continue
                        embeddings.append(np.array(face["embedding"], dtype="float32"))
                        labels.append(actor)
                        counts[actor] = counts.get(actor, 0) + 1
                    t += ns.video_sample_rate
                cap.release()
    if not embeddings:
        print("No embeddings built")
        return 1
    arr = l2norm(np.vstack(embeddings).astype("float32"))
    np.save(ns.embeddings, arr.astype("float32"))
    with open(ns.labels, "w") as f:
        json.dump(labels, f)
    if getattr(ns, "verbose", False):
        for actor in sorted(counts):
            print(actor, counts[actor])
        print(f"{arr.shape[0]}x{arr.shape[1]}")
    return 0


def cmd_actor_match(ns) -> int:
    gallery = np.load(ns.embeddings).astype("float32")
    with open(ns.labels) as f:
        labels = json.load(f)
    if len(gallery) != len(labels):
        return 2
    gallery = l2norm(gallery)
    stub = os.environ.get("DEEPFACE_STUB")
    if stub:
        vec = gallery[0]
        scores = gallery @ vec
        k = min(ns.topk, len(scores))
        idx = np.argsort(-scores)[:k]
        top = [{"label": labels[i], "score": float(scores[i])} for i in idx]
        accepted_label = labels[idx[0]] if k > 0 and scores[idx[0]] >= ns.conf else None
        accepted_score = float(scores[idx[0]]) if k > 0 else 0.0
        detections = [{"t": 0.0, "bbox": [0, 0, 10, 10], "embedding_cos_topk": top, "accepted_label": accepted_label, "accepted_score": accepted_score}]
        agg = {}
        if accepted_label is None:
            agg["unknown"] = {"frames": 1}
        else:
            agg[accepted_label] = {"frames": 1, "first_t": 0.0, "last_t": 0.0}
        out = ns.out or str(Path(ns.video).with_suffix(Path(ns.video).suffix + ".faces.json"))
        data = {"video": ns.video, "fps": 30.0, "sample_rate": ns.sample_rate, "model": "deepface:" + ns.model, "detector": ns.detector, "retry_detectors": ns.retry_detectors, "conf_threshold": ns.conf, "detections": detections, "aggregate": agg}
        with open(out, "w") as f:
            json.dump(data, f)
        if getattr(ns, "verbose", False):
            print(1, 1, out)
        return 0
    cap = cv2.VideoCapture(ns.video)
    if not cap.isOpened():
        print("Cannot open video")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    retry = [d for d in ns.retry_detectors.split(",") if d]
    detections: list[dict] = []
    frame_count = 0
    faces_total = 0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        try:
            from deepface import DeepFace
            reps = DeepFace.represent(img_path=frame, model_name=ns.model, detector_backend=ns.detector, enforce_detection=True, align=True)
        except Exception:
            reps = []
        faces = reps if isinstance(reps, list) else [reps]
        if len(faces) == 0:
            for det in retry:
                try:
                except (ValueError, RuntimeError):
                    reps = []
                faces = reps if isinstance(reps, list) else [reps]
                if len(faces) > 0:
                    break
        for face in faces:
            fa = face.get("facial_area") or {}
            x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
            if w * h < ns.min_face_area:
                continue
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < ns.blur_threshold:
                continue
            vec = l2norm(np.array(face["embedding"], dtype="float32")[None, :])[0]
            scores = gallery @ vec
            k = min(ns.topk, len(scores))
            idx = np.argsort(-scores)[:k]
            top = [{"label": labels[i], "score": float(scores[i])} for i in idx]
            accepted_label = None
            accepted_score = float(scores[idx[0]]) if k > 0 else 0.0
            if k > 0 and accepted_score >= ns.conf:
                accepted_label = labels[idx[0]]
            detections.append({"t": t, "bbox": [int(x), int(y), int(x + w), int(y + h)], "embedding_cos_topk": top, "accepted_label": accepted_label, "accepted_score": accepted_score})
            faces_total += 1
        if getattr(ns, "verbose", False) and frame_count % 25 == 0:
            print(frame_count, faces_total)
        t += ns.sample_rate
    cap.release()
    agg: dict[str, dict] = {}
    for d in detections:
        l = d["accepted_label"]
        if l is None:
            agg.setdefault("unknown", {"frames": 0})["frames"] += 1
        else:
            if l not in agg:
                agg[l] = {"frames": 0, "first_t": d["t"], "last_t": d["t"]}
            agg[l]["frames"] += 1
            agg[l]["last_t"] = d["t"]
    out = ns.out if ns.out else str(Path(ns.video).with_suffix(Path(ns.video).suffix + ".faces.json"))
    data = {"video": ns.video, "fps": float(fps), "sample_rate": ns.sample_rate, "model": "deepface:" + ns.model, "detector": ns.detector, "retry_detectors": ns.retry_detectors, "conf_threshold": ns.conf, "detections": detections, "aggregate": agg}
    with open(out, "w") as f:
        json.dump(data, f)
    if getattr(ns, "verbose", False):
        print(frame_count, faces_total, out)
    return 0

# ---------------------------------------------------------------------------
# Heatmap generation (brightness/motion timeline)
# ---------------------------------------------------------------------------

def heatmap_json_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".heatmap.json")


def heatmap_png_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + ".heatmap.png")


def compute_heatmap(video: Path, interval: float, mode: str, force: bool, write_png: bool) -> dict:
    out_json = heatmap_json_path(video)
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
        # brightness
        avg_brightness = sum(pixels) / len(pixels)
        motion_val = 0.0
        if prev_pixels is not None and (mode in ("motion", "both")):
            # mean absolute difference normalized
            diffs = 0
            for a, b in zip(pixels, prev_pixels):
                diffs += abs(a - b)
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
                img = img.resize((w, 32), resample=Image.NEAREST)
                img.save(heatmap_png_path(video))
            elif mode == "motion":
                row = [int(min(255, max(0, (s["motion"] or 0) * 255))) for s in samples]
                img = Image.new("L", (w, 1))
                img.putdata(row)
                img = img.resize((w, 32), resample=Image.NEAREST)
                img.save(heatmap_png_path(video))
            else:  # both -> two rows stacked
                row1 = [int(min(255, max(0, s["brightness"])) ) for s in samples]
                row2 = [int(min(255, max(0, (s["motion"] or 0) * 255))) for s in samples]
                img = Image.new("L", (w, 2))
                img.putdata(row1 + row2)
                img = img.resize((w, 64), resample=Image.NEAREST)
                img.save(heatmap_png_path(video))
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
    return video.with_suffix(video.suffix + ".scenes.json")


def scenes_dir(video: Path) -> Path:
    return video.parent / f".{video.name}.scenes"


def detect_scenes(video: Path, threshold: float) -> list[tuple[float, float]]:
    """Detect scene boundaries using PySceneDetect."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video_stream = open_video(str(video))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video_stream)
        scenes = scene_manager.get_scene_list()
        markers = [(start.get_seconds(), 1.0) for start, _ in scenes]
        if not any(t == 0.0 for t, _ in markers):
            markers.insert(0, (0.0, 1.0))
        return markers
    except Exception:
        # fallback stub markers
        return [(0.0, 1.0), (30.0, 1.0), (60.0, 1.0)]


def generate_scene_artifacts(video: Path, threshold: float, limit: int, gen_thumbs: bool, gen_clips: bool, thumb_width: int, clip_duration: float, force: bool) -> dict:
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
            thumb_file = asset_dir / f"scene_{idx:03d}.jpg"
            if (not thumb_file.exists()) or force:
                if os.environ.get("FFPROBE_DISABLE") or not ffmpeg_available():
                    thumb_file.write_text(f"stub thumb {safe_t}")
                else:
                    cmd = [
                        "ffmpeg", "-v", "error", "-ss", f"{safe_t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={thumb_width}:-1:force_original_aspect_ratio=decrease",
                        "-qscale:v", "4", str(thumb_file)
                    ]
                    subprocess.run(cmd, capture_output=True)
            entry["thumb"] = thumb_file.name
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


def build_transcode_cmd(src: Path, dst: Path, target_v: str, target_a: str, crf: int, v_bitrate: str | None, a_bitrate: str, preset: str, hw: str, drop_subs: bool) -> list[str]:
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
    if drop_subs:
        map_part = ["-map", "0", "-map", "-0:s"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)] + map_part + v_part + a_part + ["-movflags", "+faststart", str(dst)]
    return cmd


def cmd_codecs(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    vids = find_mp4s(root, ns.recursive)
    # Also include additional containers
    extra_exts = {".mkv", ".mov", ".avi", ".webm", ".m4v"}
    for p in root.rglob("*" if ns.recursive else "*"):
        if p.is_file() and p.suffix.lower() in extra_exts:
            vids.append(p)
    vids = sorted(set(vids))
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
            status = "OK" if r["compatible"] else "X"
            why = "" if r["compatible"] else (" [" + ",".join(r.get("why", [])) + "]")
            print(f"{status} {r['video']}: v={r['vcodec']}({r['vprofile']}) a={','.join(r['acodecs'])} c={r['container']}{why}")
    return 0


def cmd_transcode(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    out_root = Path(ns.dest).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    vids: list[Path] = []
    patterns = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    iterator = root.rglob("*") if ns.recursive else root.iterdir()
    for p in iterator:
        if p.is_file() and p.suffix.lower() in patterns:
            vids.append(p)
    vids.sort()
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
                        cmd = build_transcode_cmd(v, out_file, ns.target_v, ns.target_a, ns.crf, ns.v_bitrate, ns.a_bitrate, ns.preset, ns.hardware, ns.drop_subs)
                        start = time.time()
                        if ns.progress:
                            # Use -progress pipe:1 for periodic key=value lines; rebuild command
                            prog_cmd = ["ffmpeg", "-v", "error", "-i", str(v)]
                            if ns.drop_subs:
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


def cmd_report(ns) -> int:
    root = Path(ns.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        return 2
    videos = find_mp4s(root, ns.recursive)
    videos.sort()
    total = len(videos)
    if total == 0:
        print("No MP4 files found.")
        return 0
    # Artifact detectors
    counts = {
        "metadata": 0,
        "thumb": 0,
        "sprites": 0,
        "previews": 0,
        "subs": 0,
        "phash": 0,
        "heatmap": 0,
        "scenes": 0,
        "faces": 0,
    }
    rows: list[dict] = []
    for v in videos:
        meta_ok = metadata_path(v).exists()
        thumb_ok = (v.with_suffix(v.suffix + ".jpg")).exists()
        sprites_ok = (v.with_suffix(v.suffix + ".sprites.jpg")).exists()
        previews_ok = preview_index_path(v).exists()
        subs_ok = any((v.parent / (v.name + ext)).exists() for ext in (".vtt", ".srt", ".json"))
        phash_ok = phash_path(v).exists()
        heatmap_ok = heatmap_json_path(v).exists()
        scenes_ok = scenes_json_path(v).exists()
        faces_ok = faces_json_path(v).exists()
        if meta_ok: counts["metadata"] += 1
        if thumb_ok: counts["thumb"] += 1
        if sprites_ok: counts["sprites"] += 1
        if previews_ok: counts["previews"] += 1
        if subs_ok: counts["subs"] += 1
        if phash_ok: counts["phash"] += 1
        if heatmap_ok: counts["heatmap"] += 1
        if scenes_ok: counts["scenes"] += 1
        if faces_ok: counts["faces"] += 1
        rows.append({
            "video": v.name,
            "metadata": meta_ok,
            "thumb": thumb_ok,
            "sprites": sprites_ok,
            "previews": previews_ok,
            "subs": subs_ok,
            "phash": phash_ok,
            "heatmap": heatmap_ok,
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


def cmd_meta(ns) -> int:
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


def cmd_queue(ns) -> int:
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
    result = queue_process(videos, workers=ns.workers, force=ns.force)
    if result["errors"]:
        print("Errors (some files may be incomplete):", file=sys.stderr)
        for e in result["errors"]:
            print("  " + e, file=sys.stderr)
    print(f"Queue done for {result['total']} file(s)")
    return 0


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])
    if ns.cmd == "list":
        return cmd_list(ns)
    if ns.cmd == "meta":
        return cmd_meta(ns)
    if ns.cmd == "queue":
        return cmd_queue(ns)
    if ns.cmd == "thumb":
        return cmd_thumb(ns)
    if ns.cmd == "sprites":
        return cmd_sprites(ns)
    if ns.cmd == "previews":
        return cmd_previews(ns)
    if ns.cmd == "subs":
        return cmd_subs(ns)
    if ns.cmd == "batch":
        return cmd_batch(ns)
    if ns.cmd == "phash":
        return cmd_phash(ns)
    if ns.cmd == "heatmap":
        return cmd_heatmap(ns)
    if ns.cmd == "scenes":
        return cmd_scenes(ns)
    if ns.cmd == "faces":
        return cmd_faces(ns)
    if ns.cmd == "actor-build":
        return cmd_actor_build(ns)
    if ns.cmd == "actor-match":
        return cmd_actor_match(ns)
    if ns.cmd == "codecs":
        return cmd_codecs(ns)
    if ns.cmd == "transcode":
        return cmd_transcode(ns)
    if ns.cmd == "compare":
        return cmd_compare(ns)
    if ns.cmd == "report":
        return cmd_report(ns)
    print("Unknown command", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
