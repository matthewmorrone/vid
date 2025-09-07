"""Minimal FastAPI API exposing index functionality for a frontend client.

Endpoints (initial):
- GET /health : basic liveness
- GET /videos : list discovered MP4 files (query: directory='.', recursive=bool)
- GET /videos/{name}/artifacts : artifact presence booleans
- GET /videos/{name}/metadata : ffprobe metadata JSON (if exists)
- GET /report : coverage summary (reuses logic similar to report subcommand)
- GET /jobs : list submitted jobs
- GET /jobs/{id} : single job details
- POST /jobs : submit a job {"task": "metadata|thumbs|sprites|previews|subtitles|phash|heatmaps|scenes|codecs|transcode|report", "directory": ".", ...}

Job model (in-memory, ephemeral):
{
  id, task, params, status: queued|running|done|error, started_at, ended_at, result, error
}

NOTES / FUTURE (not yet implemented):
- Real-time progress via SSE/WebSocket (hook into per-file loops in index)
- Cancellable jobs & persistent queue (SQLite)
- Separate worker process pool for CPU isolation
- ML tasks (performer recognition) after base API stabilizes
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import sys
from pathlib import Path
import argparse
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Callable

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import index  # reuse existing functions

app = FastAPI(title="Video Tool API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static defaults for frontend consumption
DEFAULT_CONFIG = {
    "thumbs": {"time": "middle", "quality": 2},
    "sprites": {"interval": 10, "width": 320, "cols": 10, "rows": 10, "quality": 4},
    "previews": {"segments": 9, "duration": 1.0, "width": 320, "format": "webm", "crf": 30, "bitrate": "300k"},
    "subtitles": {"model": "small", "backend": "auto", "workers": 1},
    "phash": {"frames": 5, "algo": "ahash", "combine": "xor"},
    "heatmaps": {"interval": 5.0, "mode": "both"},
    "scenes": {"threshold": 0.4, "limit": 0},
    "faces": {"embed_sample_rate": 1.0, "listing_cluster_eps": 0.45},
    "tags": {"min_occurrences": 1},
}

@app.get("/config")
def get_config():
    return {"config": DEFAULT_CONFIG, "version": app.version}

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

class JobRecord(BaseModel):
    id: str
    task: str
    params: Dict[str, Any]
    status: str
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    # Basic progress counters (optional usage by tasks)
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    request_id: Optional[str] = None

_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()
_canceled_jobs: set[str] = set()
_job_cancel_events: Dict[str, threading.Event] = {}
MAX_CONCURRENT_JOBS = 4
_job_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

ALLOWED_TASKS = {
    # Core artifact & analysis tasks
    "metadata", "thumbs", "sprites", "previews", "subtitles", "phash", "dupes", "heatmaps", "scenes",
    # Face pipeline (renamed / consolidated)
    "embed", "listing",
    # Codec / conversion & reporting
    "codecs", "transcode", "report",
    # Generic clipping of arbitrary ranges
    "clip",
}

# In-memory face label assignments used by FaceLab
_face_labels: Dict[str, str] = {}


class FaceAssignRequest(BaseModel):
    id: str
    performer: str


class FaceMergeRequest(BaseModel):
    target: str
    sources: List[str]


class FaceSplitRequest(BaseModel):
    id: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Structured logging (JSON lines) with request IDs
_LOG_FH = None
_LOG_PATH = os.environ.get("VID_API_LOG")  # set path to enable persistent logging

def _ensure_log_handle():
    global _LOG_FH  # noqa: PLW0603
    if _LOG_PATH and _LOG_FH is None:
        try:
            _LOG_FH = open(_LOG_PATH, 'a', encoding='utf-8')
        except Exception:
            _LOG_FH = None

def log_event(event: str, **fields):
    _ensure_log_handle()
    rec = {"ts": time.time(), "event": event, **fields}
    line = json.dumps(rec, separators=(',',':'))
    try:
        if _LOG_FH:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        else:
            # Fallback stderr to avoid silent loss
            print(line, file=sys.stderr)
    except Exception:
        pass

@app.middleware("http")
async def request_logger(request: Request, call_next):  # noqa: D401
    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = req_id
    start = time.time()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:  # noqa: BLE001
        status = 500
        log_event("exception", request_id=req_id, path=request.url.path, method=request.method, error=str(e))
        raise
    finally:
        dur_ms = int((time.time() - start) * 1000)
        try:
            log_event("request", request_id=req_id, path=request.url.path, method=request.method, status=status, duration_ms=dur_ms)
        except Exception:
            pass
    response.headers["X-Request-ID"] = req_id
    return response


def build_artifact_info(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return artifact paths/URLs alongside existence flags."""
    s, j = index.sprite_sheet_paths(path)
    subs_path = index.artifact_dir(path) / f"{path.stem}.srt"
    thumb_artifact = index.thumbs_path(path)
    # Some callers/tests may store thumbnails alongside the video rather than
    # inside the artifact directory. In that case fall back to <video>.jpg next
    # to the source file.
    thumb_fallback = path.with_suffix(".jpg")
    thumb_exists = thumb_artifact.exists() or thumb_fallback.exists()
    thumb_url = str(thumb_artifact if thumb_artifact.exists() else thumb_fallback)

    data: Dict[str, Dict[str, Any]] = {
        "metadata": {
            "url": str(index.metadata_path(path)),
            "exists": index.metadata_path(path).exists(),
        },
        "thumbs": {
            "url": thumb_url,
            "exists": thumb_exists,
        },
        "sprites": {
            "sheet": str(s),
            "index": str(j),
            "exists": s.exists() and j.exists(),
        },
        "previews": {
            "url": str(index.preview_index_path(path)),
            "exists": index.preview_index_path(path).exists(),
        },
        "subtitles": {
            "url": str(subs_path),
            "exists": subs_path.exists(),
        },
        "phash": {
            "url": str(index.phash_path(path)),
            "exists": index.phash_path(path).exists(),
        },
    }
    heatmaps_fn = getattr(index, "heatmaps_json_path", None)
    hm = heatmaps_fn(path) if heatmaps_fn else None
    data["heatmaps"] = {
        "url": str(hm) if hm else None,
        "exists": hm.exists() if hm else False,
    }
    scenes_fn = getattr(index, "scenes_json_path", None)
    sc = scenes_fn(path) if scenes_fn else None
    data["scenes"] = {
        "url": str(sc) if sc else None,
        "exists": sc.exists() if sc else False,
    }
    faces_fn = getattr(index, "faces_json_path", None)
    fc = faces_fn(path) if faces_fn else None
    data["faces"] = {
        "url": str(fc) if fc else None,
        "exists": fc.exists() if fc else False,
    }
    return data

class JobExecutor(threading.Thread):
    def __init__(self, job_id: str):
        super().__init__(daemon=True)
        self.job_id = job_id

    def run(self) -> None:  # noqa: D401
        cancel_ev = _job_cancel_events.get(self.job_id)
        # Wait for an execution slot while allowing cancellation.
        while True:
            if cancel_ev and cancel_ev.is_set():
                with _jobs_lock:
                    job = _jobs.get(self.job_id)
                    if job:
                        job.status = "canceled"
                        job.error = "canceled"
                        job.ended_at = time.time()
                        _canceled_jobs.add(job.id)
                        log_event("job_canceled", job_id=job.id, task=job.task)
                return
            if _job_slots.acquire(timeout=0.1):
                break
        with _jobs_lock:
            job = _jobs[self.job_id]
            job.status = "running"
            job.started_at = time.time()
        stop_event = threading.Event()
        monitor_thread = None
        try:
            # Install cancellation event into index module for cooperative checks
            cancel_ev = _job_cancel_events.get(self.job_id)
            try:
                index.set_cancel_event(cancel_ev)
            except Exception:
                pass
            # Initialize progress counters when applicable
            directory = Path(job.params.get("directory", ".")).expanduser().resolve()
            recursive = bool(job.params.get("recursive", False))
            videos: list[Path] = []
            track_tasks = {"metadata","thumbs","sprites","previews","subtitles","phash","heatmaps","scenes","embed"}
            if job.task in track_tasks:
                try:
                    videos = list(index.find_mp4s(directory, recursive))
                except Exception:
                    videos = []
            with _jobs_lock:
                if videos:
                    job.progress_total = len(videos)
                    job.progress_current = 0
                else:
                    job.progress_total = None
                    job.progress_current = None
            if videos:
                monitor_thread = threading.Thread(target=_progress_monitor, args=(self.job_id, job.task, videos, stop_event), daemon=True)
                monitor_thread.start()
            result = self._execute(job.task, job.params)
            with _jobs_lock:
                job = _jobs.get(self.job_id)
                if job:
                    if job.id in _canceled_jobs:
                        # Preserve canceled status; do not overwrite with done
                        job.result = None
                        job.error = job.error or "canceled"
                        if job.status not in ("canceled", "error"):
                            job.status = "canceled"
                            job.ended_at = time.time()
                    else:
                        job.result = result
                        if job.progress_total is not None and job.progress_current is not None:
                            job.progress_current = job.progress_total
                        job.status = "done"
                        job.ended_at = time.time()
                        log_event("job_done", job_id=job.id, task=job.task)
        except index.CanceledError:
            with _jobs_lock:
                job = _jobs.get(self.job_id)
                if job:
                    job.status = "canceled"
                    job.error = "canceled"
                    job.ended_at = time.time()
                    _canceled_jobs.add(job.id)
                    log_event("job_canceled", job_id=job.id, task=job.task)
        except Exception as e:  # noqa: BLE001
            with _jobs_lock:
                job = _jobs.get(self.job_id)
                if job:
                    job.status = "error"
                    job.error = str(e)
                    job.ended_at = time.time()
                    log_event("job_error", job_id=job.id, task=job.task, error=str(e))
        finally:
            stop_event.set()
            if monitor_thread:
                try:
                    monitor_thread.join(timeout=1)
                except Exception:
                    pass
            try:
                index.set_cancel_event(None)
            except Exception:
                pass
            _job_slots.release()

    def _execute(self, task: str, params: Dict[str, Any]):  # noqa: C901
        directory = Path(params.get("directory", ".")).expanduser().resolve()
        recursive = bool(params.get("recursive", False))
        force = bool(params.get("force", False))
        if task == "report":
            return generate_report(directory, recursive)
        ns = argparse.Namespace(
            directory=str(directory),
            recursive=recursive,
            force=force,
            workers=params.get("workers", 2),
        )
        if task == "metadata":
            return index.cmd_metadata(ns)
        if task == "thumbs":
            ns.time = params.get("time", "middle")
            return index.cmd_thumbs(ns)
        if task == "sprites":
            ns.interval = params.get("interval", 10.0)
            ns.width = params.get("width", 320)
            ns.cols = params.get("cols", 10)
            ns.rows = params.get("rows", 10)
            ns.quality = params.get("quality", 4)
            ns.max_frames = params.get("max_frames", 0)
            return index.cmd_sprites(ns)
        if task == "previews":
            ns.segments = params.get("segments", 9)
            ns.duration = params.get("duration", 1.0)
            ns.width = params.get("width", 320)
            ns.format = params.get("format", "webm")
            ns.crf = params.get("crf", 30)
            ns.bitrate = params.get("bitrate", "300k")
            ns.no_index = params.get("no_index", False)
            return index.cmd_previews(ns)
        if task == "subtitles":
            ns.model = params.get("model", "small")
            ns.backend = params.get("backend", "auto")
            ns.language = params.get("language")
            ns.translate = params.get("translate", False)
            ns.output_dir = None
            ns.whisper_cpp_bin = None
            ns.whisper_cpp_model = None
            return index.cmd_subtitles(ns)
        if task == "phash":
            ns.time = params.get("time", "middle")
            ns.frames = params.get("frames", 5)
            ns.output_format = "json"
            ns.algo = params.get("algo", "ahash")
            ns.combine = params.get("combine", "xor")
            return index.cmd_phash(ns)
        if task == "dupes":
            # non-mutating analysis; direct function call
            threshold = float(params.get("threshold", 0.90))
            limit = int(params.get("limit", 0))
            return index.find_phash_duplicates(directory, recursive, threshold, limit)
        if task == "heatmaps":
            ns.interval = params.get("interval", 5.0)
            ns.mode = params.get("mode", "both")
            ns.png = params.get("png", False)
            ns.output_format = "json"
            return index.cmd_heatmaps(ns)
        if task == "scenes":
            ns.threshold = params.get("threshold", 0.4)
            ns.limit = params.get("limit", 0)
            ns.thumbs = params.get("thumbs", False)
            ns.clips = params.get("clips", False)
            ns.clip_duration = params.get("clip_duration", 2.0)
            ns.thumbs_width = params.get("thumbs_width", 320)
            ns.output_format = "json"
            return index.cmd_scenes(ns)
        if task == "embed":
            # Per-video distinct face signature extraction
            ns.sample_rate = params.get("sample_rate", 1.0)
            ns.min_face_area = params.get("min_face_area", 40 * 40)
            ns.blur_threshold = params.get("blur_threshold", 30.0)
            ns.max_gap = params.get("max_gap", 0.75)
            ns.min_track_frames = params.get("min_track_frames", 3)
            ns.match_distance = params.get("match_distance", 0.55)
            ns.cluster_eps = params.get("cluster_eps", 0.40)
            ns.cluster_min_samples = params.get("cluster_min_samples", 1)
            ns.thumbnails = params.get("thumbnails", False)
            ns.output_format = "json"
            return index.cmd_faces_embed(ns)
        if task == "listing":
            # Global deduplicated face index across videos
            ns.sample_rate = params.get("sample_rate", 1.0)
            ns.cluster_eps = params.get("cluster_eps", 0.45)
            ns.cluster_min_samples = params.get("cluster_min_samples", 1)
            ns.output = params.get("output")
            ns.output_format = params.get("output_format", "json")
            ns.gallery = params.get("gallery")
            ns.gallery_sample_rate = params.get("gallery_sample_rate", 1.0)
            ns.label_threshold = params.get("label_threshold", 0.40)
            return index.cmd_faces_index(ns)
        if task == "codecs":
            ns.target_v = params.get("target_v", "h264")
            ns.target_a = params.get("target_a", "aac")
            ns.allowed_profiles = params.get("allowed_profiles", "high,main,constrained baseline")
            ns.log = None
            ns.output_format = "json"
            return index.cmd_codecs(ns)
        if task == "transcode":
            ns.dest = params.get("dest") or str(directory / "_transcoded")
            ns.target_v = params.get("target_v", "h264")
            ns.target_a = params.get("target_a", "aac")
            ns.allowed_profiles = params.get("allowed_profiles", "high,main,constrained baseline")
            ns.crf = params.get("crf", 28)
            ns.v_bitrate = params.get("v_bitrate")
            ns.a_bitrate = params.get("a_bitrate", "128k")
            ns.preset = params.get("preset", "medium")
            ns.hardware = params.get("hardware", "none")
            ns.drop_subtitles = params.get("drop_subtitles", False)
            ns.force = force or params.get("force", False)
            ns.dry_run = params.get("dry_run", False)
            ns.progress = False
            ns.output_format = "json"
            return index.cmd_transcode(ns)
        if task == "clip":
            src = Path(params.get("file", "")).expanduser().resolve()
            ranges = params.get("ranges") or []
            dest = Path(params.get("dest") or (src.parent / "_clips"))
            fmt = params.get("format", "mp4")
            dest.mkdir(parents=True, exist_ok=True)
            out_files: list[str] = []
            with _jobs_lock:
                job = _jobs.get(self.job_id)
                if job:
                    job.progress_total = len(ranges)
                    job.progress_current = 0
            cancel_ev = _job_cancel_events.get(self.job_id)
            for idx, r in enumerate(ranges):
                if cancel_ev and cancel_ev.is_set():
                    raise index.CanceledError()
                start = float(r.get("start", 0))
                end = float(r.get("end", 0))
                outfile = dest / f"clip_{idx:03d}.{fmt}"
                if os.environ.get("FFPROBE_DISABLE") or shutil.which("ffmpeg") is None:
                    outfile.write_text(f"stub clip {start}-{end}")
                else:
                    try:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-ss",
                                f"{start:.3f}",
                                "-to",
                                f"{end:.3f}",
                                "-i",
                                str(src),
                                "-c",
                                "copy",
                                str(outfile),
                            ],
                            check=True,
                        )
                    except Exception:
                        outfile.write_text(f"stub clip {start}-{end}")
                out_files.append(str(outfile))
                with _jobs_lock:
                    job = _jobs.get(self.job_id)
                    if job:
                        job.progress_current = idx + 1
            return {"files": out_files}
        raise ValueError(f"Unsupported task {task}")


def generate_report(root: Path, recursive: bool) -> dict:
    videos = index.find_mp4s(root, recursive)
    total = len(videos)
    counts = {k: 0 for k in ("metadata","thumbs","sprites","previews","subtitles","phash","heatmaps","scenes","faces")}
    for v in videos:
        if index.metadata_path(v).exists(): counts["metadata"] += 1
        if index.thumbs_path(v).exists(): counts["thumbs"] += 1
        s, j = index.sprite_sheet_paths(v)
        if s.exists() and j.exists(): counts["sprites"] += 1
        if index.preview_index_path(v).exists(): counts["previews"] += 1
        if index.find_subtitles(v) is not None: counts["subtitles"] += 1
        if index.phash_path(v).exists(): counts["phash"] += 1
        try:
            if index.heatmaps_json_path(v).exists(): counts["heatmaps"] += 1
        except Exception:
            pass
        try:
            if index.scenes_json_path(v).exists(): counts["scenes"] += 1
        except Exception:
            pass
        try:
            if index.faces_json_path(v).exists(): counts["faces"] += 1
        except Exception:
            pass
    coverage = {k: (counts[k] / total if total else 0.0) for k in counts}
    return {"total": total, "counts": counts, "coverage": coverage}

# ---------------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------------

class JobSubmit(BaseModel):
    task: str
    directory: str = "."
    recursive: bool = False
    force: bool = False
    params: Dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}


@app.get("/videos")
def list_videos(request: Request, directory: str = Query("."), recursive: bool = Query(False), offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), q: Optional[str] = Query(None), tags: Optional[str] = Query(None), performers: Optional[str] = Query(None), match_any: bool = Query(False), detail: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    vids_all = [p for p in index.find_mp4s(root, recursive)]
    # Load metadata summary cache once
    summaries = index.load_metadata_summary(root, recursive)
    if q:
        qlow = q.lower()
        vids_all = [p for p in vids_all if qlow in p.name.lower()]
    # Tag filtering
    if tags or performers:
        tag_set = {t.strip() for t in (tags.split(',') if tags else []) if t.strip()}
        perf_set = {t.strip() for t in (performers.split(',') if performers else []) if t.strip()}
        filtered = []
        for p in vids_all:
            tfile = index.artifact_dir(p) / f"{p.stem}.tags.json"
            if not tfile.exists():
                continue
            try:
                data = json.loads(tfile.read_text())
            except Exception:
                continue
            vt = set(data.get("tags", []) or [])
            vp = set(data.get("performers", []) or [])
            cond_tags = not tag_set or ((vt & tag_set) if match_any else tag_set.issubset(vt))
            cond_perf = not perf_set or ((vp & perf_set) if match_any else perf_set.issubset(vp))
            if cond_tags and cond_perf:
                filtered.append(p)
        vids_all = filtered
    total = len(vids_all)
    slice_v = vids_all[offset: offset + limit]
    etag_base = f"{root}:{recursive}:{total}:{offset}:{limit}:{q or ''}:{tags or ''}:{performers or ''}:{int(match_any)}".encode()
    import hashlib
    etag = hashlib.sha1(etag_base).hexdigest()
    inm = request.headers.get("if-none-match")
    if inm and inm.strip('"') == etag:
        raise HTTPException(status_code=304, detail="not modified")
    videos_out: List[Dict[str, Any]] = [
        {"path": str(p), "name": p.name, "size": p.stat().st_size} for p in slice_v
    ]
    if detail:
        for i, info in enumerate(videos_out):
            p = slice_v[i]
            # tags
            tfile = index.artifact_dir(p) / f"{p.stem}.tags.json"
            if tfile.exists():
                try:
                    tdata = json.loads(tfile.read_text())
                    info["tags"] = tdata.get("tags", [])
                    info["performers"] = tdata.get("performers", [])
                except Exception:
                    info["tags"] = []
                    info["performers"] = []
            else:
                info["tags"] = []
                info["performers"] = []
            # duration / codecs from summary cache
            rel = None
            try:
                rel = str(p.relative_to(root))
            except Exception:
                rel = p.name
            summ = summaries.get(rel)
            if summ:
                if summ.get("duration") is not None:
                    info["duration"] = summ.get("duration")
                if summ.get("vcodec") is not None:
                    info["vcodec"] = summ.get("vcodec")
                if summ.get("acodec") is not None:
                    info["acodec"] = summ.get("acodec")
            # artifact paths/presence
            info["artifacts"] = build_artifact_info(p)
    resp = {"directory": str(root), "count": total, "videos": videos_out, "offset": offset, "limit": limit, "etag": etag}
    return resp


@app.get("/videos/{name}/artifacts")
def video_artifacts(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    s, j = index.sprite_sheet_paths(path)
    thumb_artifact = index.thumbs_path(path)
    thumb_fallback = path.with_suffix(".jpg")
    thumbs_exist = thumb_artifact.exists() or thumb_fallback.exists()

    return {
        "video": name,
        "metadata": index.metadata_path(path).exists(),
        "thumbs": thumbs_exist,
        "sprites": s.exists() and j.exists(),
        "previews": index.preview_index_path(path).exists(),
        "subtitles": index.find_subtitles(path) is not None,
        "phash": index.phash_path(path).exists(),
        "heatmaps": hasattr(index, 'heatmaps_json_path') and index.heatmaps_json_path(path).exists(),
        "scenes": hasattr(index, 'scenes_json_path') and index.scenes_json_path(path).exists(),
        "faces": hasattr(index, 'faces_json_path') and index.faces_json_path(path).exists(),
    }


@app.get("/videos/{name}/metadata")
def video_metadata(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    metadata_file = index.metadata_path(path)
    if not metadata_file.exists():
        raise HTTPException(404, "metadata not found")
    try:
        return json.loads(metadata_file.read_text())
    except Exception:
        return {"raw": metadata_file.read_text()}

@app.get("/videos/{name}/tags")
def get_video_tags(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    tfile = index.artifact_dir(path) / f"{path.stem}.tags.json"
    if not tfile.exists():
        return {"video": name, "tags": [], "performers": [], "description": ""}
    try:
        data = json.loads(tfile.read_text())
    except Exception as e:
        raise HTTPException(500, "invalid tags file") from e
    if "description" not in data:
        data["description"] = ""
    return data

class TagUpdate(BaseModel):
    add: list[str] | None = None
    remove: list[str] | None = None
    performers_add: list[str] | None = None
    performers_remove: list[str] | None = None
    replace: bool = False
    description: str | None = None

@app.patch("/videos/{name}/tags")
def update_video_tags(name: str, payload: TagUpdate, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    tfile = index.artifact_dir(path) / f"{path.stem}.tags.json"
    if tfile.exists():
        try:
            data = json.loads(tfile.read_text())
        except Exception:
            data = {"video": name, "tags": [], "performers": [], "description": ""}
    else:
        data = {"video": name, "tags": [], "performers": [], "description": ""}
    data.setdefault("description", "")
    if payload.replace and payload.add is not None:
        data["tags"] = []
    if payload.add:
        for t in payload.add:
            if t not in data["tags"]:
                data["tags"].append(t)
    if payload.remove:
        data["tags"] = [t for t in data["tags"] if t not in payload.remove]
    if payload.performers_add:
        for t in payload.performers_add:
            if t not in data["performers"]:
                data["performers"].append(t)
    if payload.performers_remove:
        data["performers"] = [t for t in data["performers"] if t not in payload.performers_remove]
    if payload.description is not None:
        data["description"] = payload.description
    # Write back
    try:
        tfile.write_text(json.dumps(data, indent=2))
    except Exception:
        raise HTTPException(500, "failed to write tags")
    return data


class RenameRequest(BaseModel):
    new_name: str


@app.post("/videos/{name}/rename")
def rename_video(name: str, payload: RenameRequest, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    src = root / name
    dst = root / payload.new_name
    if not src.exists():
        raise HTTPException(404, "video not found")
    # Case-insensitive collision check
    lower_new_name = payload.new_name.lower()
    for entry in root.iterdir():
        if entry.is_file() and entry.name.lower() == lower_new_name and entry != src:
            raise HTTPException(409, "destination exists (case-insensitive collision)")
    if dst.exists():
        raise HTTPException(409, "destination exists")
    try:
        index.rename_with_artifacts(src, dst)
    except FileNotFoundError as e:
        raise HTTPException(404, "video not found") from e
    except Exception as e:
        raise HTTPException(500, "rename failed") from e
    return {"old_name": name, "new_name": payload.new_name}

@app.get("/videos/{name}")
def video_detail(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    # Base info
    info: Dict[str, Any] = {"video": name, "size": path.stat().st_size}
    # Artifact presence
    info["artifacts"] = build_artifact_info(path)
    # Tags
    tfile = index.artifact_dir(path) / f"{path.stem}.tags.json"
    if tfile.exists():
        try:
            tdata = json.loads(tfile.read_text())
            info["tags"] = tdata.get("tags", [])
            info["performers"] = tdata.get("performers", [])
            info["description"] = tdata.get("description", "")
        except Exception:
            info["tags"] = []
            info["performers"] = []
            info["description"] = ""
    else:
        info["tags"] = []
        info["performers"] = []
        info["description"] = ""
    # Duration/codecs via summary cache (single-file mode)
    try:
        summaries = index.load_metadata_summary(root, recursive=False)
        rel = str(path.relative_to(root))
        summ = summaries.get(rel)
        if summ:
            if summ.get("duration") is not None:
                info["duration"] = summ.get("duration")
            if summ.get("vcodec") is not None:
                info["vcodec"] = summ.get("vcodec")
            if summ.get("acodec") is not None:
                info["acodec"] = summ.get("acodec")
    except Exception:
        pass
    return info

@app.get("/tags/export")
def export_tags(directory: str = Query("."), recursive: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    result = []
    for v in index.find_mp4s(root, recursive):
        tfile = index.artifact_dir(v) / f"{v.stem}.tags.json"
        if tfile.exists():
            try:
                data = json.loads(tfile.read_text())
                result.append(data)
            except Exception:
                pass
    return {"videos": result}

class BulkImport(BaseModel):
    videos: list[Dict[str, Any]]

@app.post("/tags/import")
def import_tags(payload: BulkImport, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    written = 0
    for entry in payload.videos:
        name = entry.get("video")
        if not name:
            continue
        path = root / name
        if not path.exists():
            continue
        try:
            (index.artifact_dir(path) / f"{path.stem}.tags.json").write_text(json.dumps({
                "video": name,
                "tags": entry.get("tags", []),
                "performers": entry.get("performers", []),
                "description": entry.get("description", ""),
            }, indent=2))
            written += 1
        except Exception:
            pass
    return {"written": written}

@app.get("/tags/summary")
def tags_summary(directory: str = Query("."), recursive: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    tag_counts: Dict[str, int] = {}
    perf_counts: Dict[str, int] = {}
    videos_with_tags = 0
    for v in index.find_mp4s(root, recursive):
        tfile = index.artifact_dir(v) / f"{v.stem}.tags.json"
        if not tfile.exists():
            continue
        try:
            data = json.loads(tfile.read_text())
        except Exception:
            continue
        tags_list = data.get("tags", []) or []
        perf_list = data.get("performers", []) or []
        if tags_list or perf_list:
            videos_with_tags += 1
        for t in tags_list:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        for p in perf_list:
            perf_counts[p] = perf_counts.get(p, 0) + 1
    return {"tags": tag_counts, "performers": perf_counts, "videos": videos_with_tags}


@app.get("/videos/{name}/faces")
def video_faces(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    faces_file = index.faces_json_path(path)
    if not faces_file.exists():
        raise HTTPException(404, "faces not found")
    return json.loads(faces_file.read_text())


@app.get("/faces/{name}/signatures")
def face_signatures(name: str, directory: str = Query(".")):
    """Return persisted per-video face signatures produced by embed (404 if missing)."""
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    sig_file = index.artifact_dir(path) / f"{path.stem}.faces.signatures.json"
    if not sig_file.exists():
        raise HTTPException(404, "signatures not found")
    return json.loads(sig_file.read_text())


@app.get("/faces/listing")
def face_listing(directory: str = Query("."), recursive: bool = Query(False), sample_rate: float = Query(1.0), cluster_eps: float = Query(0.45), cluster_min_samples: int = Query(1), offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000), gallery: Optional[str] = Query(None), gallery_sample_rate: float = Query(1.0), label_threshold: float = Query(0.40)):
    """Generate (or return cached) global face listing index; supports pagination and optional gallery labeling."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    cache_file = root / ".artifacts" / ".faces.index.json"
    regenerate = True
    if cache_file.exists():
        try:
            stat = cache_file.stat()
            # Simple heuristic: reuse if younger than 10 minutes
            if time.time() - stat.st_mtime < 600:
                regenerate = False
        except Exception:
            pass
    if regenerate:
        params = {
            "sample_rate": sample_rate,
            "cluster_eps": cluster_eps,
            "cluster_min_samples": cluster_min_samples,
            "gallery": gallery,
            "gallery_sample_rate": gallery_sample_rate,
            "label_threshold": label_threshold,
        }
        ns = argparse.Namespace(directory=str(root), recursive=recursive, sample_rate=sample_rate, cluster_eps=cluster_eps, cluster_min_samples=cluster_min_samples, output=None, output_format="json", gallery=gallery, gallery_sample_rate=gallery_sample_rate, label_threshold=label_threshold)
        index_data = index.cmd_faces_index(ns)
        try:
            cache_file.parent.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps({"index": index_data, "params": params, "generated_at": time.time()}))
        except Exception:
            pass
    try:
        data = json.loads(cache_file.read_text()) if cache_file.exists() else {"index": {"people": [], "videos": 0}}
        people = data.get("index", {}).get("people", [])
        for p in people:
            pid = p.get("id")
            if pid in _face_labels:
                p["performer"] = _face_labels[pid]
    except Exception:
        people = []
    total = len(people)
    slice_people = people[offset: offset + limit]
    return {"people": slice_people, "total": total, "offset": offset, "limit": limit, "videos": data.get("index", {}).get("videos", 0)}


@app.post("/faces/assign")
def faces_assign(payload: FaceAssignRequest):
    _face_labels[payload.id] = payload.performer
    return {"id": payload.id, "performer": payload.performer}


@app.post("/faces/merge")
def faces_merge(payload: FaceMergeRequest):
    label = _face_labels.get(payload.target)
    for sid in payload.sources:
        if label:
            _face_labels[sid] = label
        else:
            _face_labels.pop(sid, None)
    return {"merged": [payload.target, *payload.sources]}


@app.post("/faces/split")
def faces_split(payload: FaceSplitRequest):
    _face_labels.pop(payload.id, None)
    return {"id": payload.id}


@app.get("/report")
def report(directory: str = Query("."), recursive: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    return generate_report(root, recursive)


@app.get("/phash/duplicates")
def phash_duplicates(directory: str = Query("."), recursive: bool = Query(False), threshold: float = Query(0.90), limit: int = Query(0)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    return index.find_phash_duplicates(root, recursive, threshold, limit)


@app.post("/jobs", response_model=JobRecord)
def submit_job(payload: JobSubmit, background: BackgroundTasks):
    if payload.task not in ALLOWED_TASKS:
        raise HTTPException(400, f"unsupported task {payload.task}")
    job_id = uuid.uuid4().hex
    # Carry forward request id if within request context
    req_id = None
    try:
        from fastapi import Request as _Req  # local import
        # Not directly accessible here unless dependency injection; leave as None unless global context is used.
    except Exception:
        pass
    record = JobRecord(id=job_id, task=payload.task, params={**(payload.params or {}), "directory": payload.directory, "recursive": payload.recursive, "force": payload.force}, status="queued", request_id=req_id)
    with _jobs_lock:
        _jobs[job_id] = record
    _job_cancel_events[job_id] = threading.Event()
    log_event("job_queued", job_id=job_id, task=payload.task, directory=payload.directory, recursive=payload.recursive)
    background.add_task(_start_job, job_id)
    return record


def _start_job(job_id: str):
    executor = JobExecutor(job_id)
    executor.start()

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job.status in ("done","error","canceled"):
            return {"job_id": job_id, "status": job.status}
        # Signal cancellation
        ev = _job_cancel_events.get(job_id)
        if ev:
            ev.set()
        # Mark intent; final status set by worker when it observes cancellation
        job.status = "cancel_requested" if job.status == "running" else "canceled"
        if job.status == "canceled":
            job.error = "canceled"
            job.ended_at = time.time()
            _canceled_jobs.add(job_id)
            log_event("job_canceled", job_id=job_id, task=job.task)
            return {"job_id": job_id, "status": "canceled"}
        log_event("job_cancel_requested", job_id=job_id, task=job.task)
        return {"job_id": job_id, "status": job.status}


def _progress_monitor(job_id: str, task: str, videos: list[Path], stop_event: threading.Event):
    """Background thread to approximate per-job progress by counting produced artifacts.

    It samples every 0.3s until stop_event is set or job finishes.
    """
    artifact_checkers: Dict[str, Callable[[Path], bool]] = {
        "metadata": lambda v: index.metadata_path(v).exists(),
        "thumbs": lambda v: index.thumbs_path(v).exists(),
        "sprites": lambda v: all(p.exists() for p in index.sprite_sheet_paths(v)),
        "previews": lambda v: index.preview_index_path(v).exists(),
        "subtitles": lambda v: index.find_subtitles(v) is not None,
        "phash": lambda v: index.phash_path(v).exists(),
        "heatmaps": lambda v: hasattr(index, 'heatmaps_json_path') and index.heatmaps_json_path(v).exists(),
        "scenes": lambda v: hasattr(index, 'scenes_json_path') and index.scenes_json_path(v).exists(),
        "embed": lambda v: (index.artifact_dir(v) / f"{v.stem}.faces.signatures.json").exists(),
    }
    checker = artifact_checkers.get(task)
    if not checker:
        return
    while not stop_event.is_set():
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job or job.status != 'running':
                    break
                done = 0
                for v in videos:
                    try:
                        if checker(v):
                            done += 1
                    except Exception:
                        pass
                job.progress_current = done
        except Exception:
            pass
        stop_event.wait(0.3)


@app.get("/jobs")
def list_jobs():
    with _jobs_lock:
        return {"jobs": [j.dict() for j in _jobs.values()]}


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Server-Sent Events stream of job progress/status changes.

    Emits events of type 'progress' with data containing status, progress_current/total,
    and timestamps until the job reaches a terminal state (done|error|canceled).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")

    async def event_gen():
        last_sig = None
        # Send initial snapshot immediately
        while True:
            await asyncio.sleep(0.0)
            with _jobs_lock:
                j = _jobs.get(job_id)
                if not j:
                    # Job disappeared; terminate
                    payload = {"event": "gone", "job_id": job_id}
                    yield f"event: gone\ndata: {json.dumps(payload)}\n\n"
                    return
                sig = (j.status, j.progress_current, j.progress_total)
                terminal = j.status in ("done", "error", "canceled")
                if sig != last_sig:
                    payload = {
                        "job_id": j.id,
                        "status": j.status,
                        "progress_current": j.progress_current,
                        "progress_total": j.progress_total,
                        "started_at": j.started_at,
                        "ended_at": j.ended_at,
                        "error": j.error,
                    }
                    yield f"event: progress\ndata: {json.dumps(payload, separators=(',',':'))}\n\n"
                    last_sig = sig
                if terminal:
                    return
            await asyncio.sleep(0.5)

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), headers=headers)


# Run: uvicorn api:app --reload
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
