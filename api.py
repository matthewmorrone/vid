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
    # Basic progress counters (optional usage by tasks)
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None

_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()

ALLOWED_TASKS = {
    # Core artifact & analysis tasks
    "metadata", "thumbs", "sprites", "previews", "subtitles", "phash", "dupes", "heatmaps", "scenes",
    # Face pipeline (renamed / consolidated)
    "embed", "listing",
    # Codec / conversion & reporting
    "codecs", "transcode", "report"
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
        ns = SimpleNamespace(
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
def list_videos(directory: str = Query("."), recursive: bool = Query(False), offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), q: Optional[str] = Query(None)):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, "directory not found")
    vids_all = [p for p in index.find_mp4s(root, recursive)]
    if q:
        qlow = q.lower()
        vids_all = [p for p in vids_all if qlow in p.name.lower()]
    total = len(vids_all)
    slice_v = vids_all[offset: offset + limit]
    etag_base = f"{root}:{recursive}:{total}:{offset}:{limit}:{q or ''}".encode()
    import hashlib
    etag = hashlib.sha1(etag_base).hexdigest()
    return {"directory": str(root), "count": total, "videos": [str(p) for p in slice_v], "offset": offset, "limit": limit, "etag": etag}


@app.get("/videos/{name}/artifacts")
def video_artifacts(name: str, directory: str = Query(".")):
    root = Path(directory).expanduser().resolve()
    path = root / name
    if not path.exists():
        raise HTTPException(404, "video not found")
    s, j = index.sprite_sheet_paths(path)
    return {
        "video": name,
        "metadata": index.metadata_path(path).exists(),
    "thumbs": index.thumbs_path(path).exists(),
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
        ns = SimpleNamespace(directory=str(root), recursive=recursive, sample_rate=sample_rate, cluster_eps=cluster_eps, cluster_min_samples=cluster_min_samples, output=None, output_format="json", gallery=gallery, gallery_sample_rate=gallery_sample_rate, label_threshold=label_threshold)
        index_data = index.cmd_faces_index(ns)
        try:
            cache_file.parent.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps({"index": index_data, "params": params, "generated_at": time.time()}))
        except Exception:
            pass
    try:
        data = json.loads(cache_file.read_text()) if cache_file.exists() else {"index": {"people": [], "videos": 0}}
        people = data.get("index", {}).get("people", [])
    except Exception:
        people = []
    total = len(people)
    slice_people = people[offset: offset + limit]
    return {"people": slice_people, "total": total, "offset": offset, "limit": limit, "videos": data.get("index", {}).get("videos", 0)}


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
