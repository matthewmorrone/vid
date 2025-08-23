"""Minimal FastAPI API exposing index functionality for a frontend client.

Endpoints (initial):
- GET /health : basic liveness
- GET /videos : list discovered MP4 files (query: directory='.', recursive=bool)
- GET /videos/{name}/artifacts : artifact presence booleans
- GET /videos/{name}/metadata : ffprobe metadata JSON (if exists)
- GET /report : coverage summary (reuses logic similar to report subcommand)
- GET /jobs : list submitted jobs
- GET /jobs/{id} : single job details
- POST /jobs : submit a job {"task": "meta|thumb|sprites|previews|subs|phash|heatmap|scenes|codecs|transcode|report", "directory": ".", ...}

Job model (in-memory, ephemeral):
{
  id, task, params, status: queued|running|done|error, started_at, ended_at, result, error
}

NOTES / FUTURE (not yet implemented):
- Real-time progress via SSE/WebSocket (hook into per-file loops in index)
- Cancellable jobs & persistent queue (SQLite)
- Separate worker process pool for CPU isolation
- ML tasks (faces / performers) after base API stabilizes
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
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

_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()

ALLOWED_TASKS = {
    "meta", "thumb", "sprites", "previews", "subs", "phash", "heatmap", "scenes", "codecs", "transcode", "report"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class JobExecutor(threading.Thread):
    def __init__(self, job_id: str):
        super().__init__(daemon=True)
        self.job_id = job_id

    def run(self) -> None:  # noqa: D401
        with _jobs_lock:
            job = _jobs[self.job_id]
            job.status = "running"
            job.started_at = time.time()
        try:
            result = self._execute(job.task, job.params)
            with _jobs_lock:
                job.result = result
                job.status = "done"
                job.ended_at = time.time()
        except Exception as e:  # noqa: BLE001
            with _jobs_lock:
                job.status = "error"
                job.error = str(e)
                job.ended_at = time.time()

    def _execute(self, task: str, params: Dict[str, Any]):  # noqa: C901
        directory = Path(params.get("directory", ".")).expanduser().resolve()
        recursive = bool(params.get("recursive", False))
        force = bool(params.get("force", False))
        if task == "report":
            return generate_report(directory, recursive)
        # Build synthetic namespace for selected commands (only minimal flags for now)
        ns = SimpleNamespace(
            directory=str(directory),
            recursive=recursive,
            force=force,
            workers=params.get("workers", 2),
        )
        if task == "meta":
            return index.cmd_meta(ns)
        if task == "thumb":
            ns.time = params.get("time", "middle")
            return index.cmd_thumb(ns)
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
        if task == "subs":
            ns.model = params.get("model", "small")
            ns.backend = params.get("backend", "auto")
            ns.language = params.get("language")
            ns.translate = params.get("translate", False)
            ns.format = params.get("format", "vtt")
            ns.output_dir = None
            ns.whisper_cpp_bin = None
            ns.whisper_cpp_model = None
            return index.cmd_subs(ns)
        if task == "phash":
            ns.time = params.get("time", "middle")
            ns.frames = params.get("frames", 5)
            ns.output_format = "json"
            ns.algo = params.get("algo", "ahash")
            ns.combine = params.get("combine", "xor")
            return index.cmd_phash(ns)
        if task == "heatmap":
            ns.interval = params.get("interval", 5.0)
            ns.mode = params.get("mode", "both")
            ns.png = params.get("png", False)
            ns.output_format = "json"
            return index.cmd_heatmap(ns)
        if task == "scenes":
            ns.threshold = params.get("threshold", 0.4)
            ns.limit = params.get("limit", 0)
            ns.thumbs = params.get("thumbs", False)
            ns.clips = params.get("clips", False)
            ns.clip_duration = params.get("clip_duration", 2.0)
            ns.thumb_width = params.get("thumb_width", 320)
            ns.output_format = "json"
            return index.cmd_scenes(ns)
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
            ns.drop_subs = params.get("drop_subs", False)
            ns.force = force or params.get("force", False)
            ns.dry_run = params.get("dry_run", False)
            ns.progress = False
            ns.output_format = "json"
            return index.cmd_transcode(ns)
        raise ValueError(f"Unsupported task {task}")


def generate_report(root: Path, recursive: bool) -> dict:
    videos = index.find_mp4s(root, recursive)
    total = len(videos)
    counts = {k: 0 for k in ("metadata","thumb","sprites","previews","subs","phash","heatmap","scenes")}
    for v in videos:
        if index.metadata_path(v).exists(): counts["metadata"] += 1
        if (v.with_suffix(v.suffix + ".jpg")).exists(): counts["thumb"] += 1
        if (v.with_suffix(v.suffix + ".sprites.jpg")).exists(): counts["sprites"] += 1
        if index.preview_index_path(v).exists(): counts["previews"] += 1
        if any((v.parent / (v.name + ext)).exists() for ext in (".vtt", ".srt", ".json")): counts["subs"] += 1
        if index.phash_path(v).exists(): counts["phash"] += 1
        try:
            if index.heatmap_json_path(v).exists(): counts["heatmap"] += 1
        except Exception:
            pass
        try:
            if index.scenes_json_path(v).exists(): counts["scenes"] += 1
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
def list_videos(directory: str = Query("."), recursive: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    vids = [str(p) for p in index.find_mp4s(root, recursive)]
    return {"directory": str(root), "count": len(vids), "videos": vids}


@app.get("/videos/{name}/artifacts")
def video_artifacts(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    return {
        "video": name,
        "metadata": index.metadata_path(path).exists(),
        "thumb": (path.with_suffix(path.suffix + ".jpg")).exists(),
        "sprites": (path.with_suffix(path.suffix + ".sprites.jpg")).exists(),
        "previews": index.preview_index_path(path).exists(),
        "subs": any((path.parent / (path.name + ext)).exists() for ext in (".vtt", ".srt", ".json")),
        "phash": index.phash_path(path).exists(),
        "heatmap": hasattr(index, 'heatmap_json_path') and index.heatmap_json_path(path).exists(),
        "scenes": hasattr(index, 'scenes_json_path') and index.scenes_json_path(path).exists(),
    }


@app.get("/videos/{name}/metadata")
def video_metadata(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    meta_file = index.metadata_path(path)
    if not meta_file.exists():
        raise HTTPException(404, "metadata not found")
    try:
        return json.loads(meta_file.read_text())
    except Exception:
        return {"raw": meta_file.read_text()}


@app.get("/report")
def report(directory: str = Query("."), recursive: bool = Query(False)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    return generate_report(root, recursive)


@app.post("/jobs", response_model=JobRecord)
def submit_job(payload: JobSubmit, background: BackgroundTasks):
    if payload.task not in ALLOWED_TASKS:
        raise HTTPException(400, f"unsupported task {payload.task}")
    job_id = uuid.uuid4().hex
    record = JobRecord(id=job_id, task=payload.task, params={**(payload.params or {}), "directory": payload.directory, "recursive": payload.recursive, "force": payload.force}, status="queued")
    with _jobs_lock:
        _jobs[job_id] = record
    background.add_task(_start_job, job_id)
    return record


def _start_job(job_id: str):
    executor = JobExecutor(job_id)
    executor.start()


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


# Run: uvicorn api:app --reload
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
