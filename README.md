# Video Utility

Unified Python script `index.py` to:
1. List MP4 files
2. Generate ffprobe metadata JSON files
3. Run an ephemeral in-memory queue to batch metadata generation with limited concurrency
4. Face pipeline (two steps):
	- embed: per‑video distinct face signatures
	- listing: global cross‑video clustering (people index) with optional supervised labeling

Dependencies:
- [DeepFace](https://pypi.org/project/deepface/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [FFmpeg](https://ffmpeg.org/)

## Commands
```
python index.py list  [dir] [-r] [--json] [--show-size] [--sort name|size]
python index.py metadata  [dir] [-r] [--force] [--workers N]
python index.py queue [dir] [-r] [--force] [--workers N]
python index.py thumbs [dir] [-r] [--force] [--workers N] [--time middle|10|25%] [--quality 2]
python index.py sprites [dir] [-r] [--interval 10] [--width 320] [--cols 10] [--rows 10] [--quality 4] [--max-frames 0] [--force]
python index.py previews [dir] [-r] [--segments 9] [--duration 1.0] [--width 320] [--format mp4|webm] [--crf 30] [--bitrate 300k] [--workers 2] [--force] [--no-index]
python index.py subtitles [dir] [-r] [--model small] [--backend auto|whisper|faster-whisper|whisper.cpp] [--language xx] [--translate] [--workers 1] [--force]
python index.py batch [dir] [-r] [--tasks meta,thumbs] [--max-meta 3] [--max-thumbs 4] [--max-sprites 2] [--max-previews 2] [--max-subtitles 1] [--force]
python index.py phash [dir] [-r] [--time middle|60|25%] [--frames 5] [--algo ahash|dct] [--combine xor|majority|avg] [--workers 2] [--force] [--output-format json|text]
python index.py heatmaps [dir] [-r] [--interval 5.0] [--mode brightness|motion|both] [--png] [--workers 2] [--force] [--output-format json|text]
python index.py scenes [dir] [-r] [--threshold 0.4] [--limit N] [--thumbs] [--clips] [--clip-duration 2.0] [--thumbs-width 320] [--workers 2] [--force] [--output-format json|text]
python index.py phash-dupes [dir] [-r] [--threshold 0.92] [--limit 25]
python index.py embed [dir] [-r] [--sample-rate 1.0] [--min-face-area 1600] [--blur-threshold 30.0] [--max-gap 0.75] [--min-track-frames 3] [--match-distance 0.55] [--cluster-eps 0.40] [--cluster-min-samples 1] [--thumbnails] [--force]
python index.py listing [dir] [-r] [--sample-rate 1.0] [--cluster-eps 0.45] [--cluster-min-samples 1] [--gallery GALLERY_DIR] [--gallery-sample-rate 1.0] [--label-threshold 0.40]
python index.py codecs [dir] [-r] [--target-v h264] [--target-a aac] [--allowed-profiles list] [--workers 4] [--output-format json|text]
python index.py transcode [dir] dest [-r] [--target-v h264] [--target-a aac] [--crf 28] [--v-bitrate 3000k] [--a-bitrate 128k] [--preset medium] [--hardware none|videotoolbox] [--drop-subtitles] [--workers 1] [--force] [--dry-run] [--output-format json|text]
python index.py compare original.mp4 other.mp4 [--output-format json|text]
python index.py report [dir] [-r] [--output-format json|text]
python index.py faces [dir] [-r] [--frame-rate 1.0] [--eps 0.5] [--min-samples 2] [--output faces.json]
```

`list`:
	Show MP4 files. Optional JSON output, human sizes, recursion, sorting.

`meta`:
	Generate ffprobe metadata files `<video>.ffprobe.json`. Skips existing unless `--force`.

`queue`:
	In-memory queue variant that processes the same metadata generation with a small worker pool (capped at 4) to prevent thrash on low-power devices.

`thumbs`:
	Generate a single JPEG thumbnail per video (`<video>.mp4.jpg`). By default grabs a frame from the middle. `--time` can be seconds, percentage (e.g. `25%`), or `middle`. Uses existing metadata JSON (if present) to determine duration; otherwise falls back to a static second.

`sprites`:
	Produce a sprite sheet (`<video>.mp4.sprites.jpg`) and an accompanying JSON index (`<video>.mp4.sprites.json`) for hover scrubbing. Frames are sampled every `--interval` seconds (default 10) up to `cols*rows` (or `--max-frames` if set and smaller). Tiles are scaled to `--width` keeping aspect ratio. JSON includes grid and interval metadata for client-side lookup.

`previews`:
	Generate short hover preview clips (default 1s each) at decile positions (10%..90% of duration). Default format is webm (VP9). Outputs per-video directory `.<name>.mp4.previews/seg_XX.webm` (or .mp4 if chosen) and an index JSON `<video>.mp4.previews.json` unless `--no-index` given. Configure number of segments, clip duration, width, format, quality, and worker threads.

`subtitles`:
	Transcribe audio to subtitles using Whisper models. Automatic backend detection prefers `faster-whisper`, then `whisper`, else falls back to external `whisper.cpp`. Output formats: VTT (default), SRT, JSON. Use `--translate` to translate to English. Provide `--whisper-cpp-bin` and `--whisper-cpp-model` when using whisper.cpp.

`batch`:
	Ephemeral multi-task scheduler for large sets. Runs selected tasks (`--tasks` CSV) with per-type concurrency caps (`--max-<task>`). Skips artifacts that already exist unless `--force`. Dynamically balances workers across task types to avoid overload (e.g., only 1 subtitles job while multiple thumbs run).

`phash`:
	Compute a perceptual hash. Algorithms:
	- `ahash` (default): 32x32 average hash (1024 bits when single frame; combined via method below).
	- `dct`: classic pHash style 8x8 low-frequency DCT (64 bits).
	Multi-frame combination (`--frames > 1`):
	- `xor` (default): fast XOR of per-frame hashes.
	- `majority` / `avg`: bit vote (>50% set). (For DCT both are identical; `avg` alias.)
	Default samples 5 evenly spaced frames. Single-frame (`--frames 1`) yields the raw per-frame hash. Output `<video>.mp4.phash.json` includes algorithm, combine mode, frame positions. Compare via Hamming distance.

`heatmaps`:
	Sample frames every `--interval` seconds to produce a lightweight timeline of brightness and/or motion (mean absolute frame diff) values. Writes `<video>.mp4.heatmaps.json` (and optionally a small stripe visualization `<video>.mp4.heatmaps.png` with `--png`). Useful for spotting dark sections, scene density, or high-action regions.

`scenes`:
        Detect scene boundaries using the PySceneDetect library's content detector. Produces `<video>.mp4.scenes.json` with marker times and scores; optionally generates per-scene thumbnail JPEGs and/or short clip previews inside a hidden directory `.<name>.mp4.scenes/`. Use `--threshold` to tune sensitivity (lower = more markers). Limit output with `--limit`.

`phash-dupes`:
	Scan existing perceptual hash JSON artifacts to find likely duplicate / near-duplicate videos by Hamming distance. Adjustable similarity threshold.

`embed`:
	Extract per‑video distinct face signatures (track clustering + representative embedding). Persists `<video>.faces.signatures.json` for reuse. Supports basic blur filtering, minimum face area, temporal track merging, and optional thumbnail extraction.

`listing`:
	Aggregate all per‑video face signatures into a global deduplicated index of people with cross‑video occurrences. Optional supervised labeling from a gallery directory (`--gallery`) of subfolders (one per person) containing images or videos. Uses cosine/L2 distance threshold to assign labels. Produces an in-memory result (and cached `.artifacts/.faces.index.json`).

`codecs`:
        Scan video files (mp4/mkv/mov/avi/webm/m4v) and list container, video codec/profile, audio codecs, size, duration, and whether they match target constraints. Supports JSON or text output. Useful pre-pass before deciding what to transcode.

`transcode`:
	Batch transcode incompatible (or all with `--force`) videos into a normalized MP4 (default H.264 + AAC) tree mirroring source structure. Supports CRF/preset software encoders and optional VideoToolbox hardware acceleration. Dry-run mode plans actions without encoding.

`compare`:
        Compute SSIM and PSNR between two files (original vs other) and classify quality levels.

`report`:
        Summarize library artifact coverage: counts & percentages for metadata, thumbnail, sprites, previews, subtitles, phash, heatmaps, scenes. Outputs per-file presence matrix (JSON mode) or a concise table (text mode).

Face pipeline workflow:
1. Run `embed` (parallelizable) to create per-video signature artifacts.
2. Run `listing` to build / refresh the global index (fast if signatures already present). Add `--gallery` to auto-label clusters.


Set `FFPROBE_DISABLE=1` to produce stub metadata JSON without calling ffprobe (useful on systems lacking ffprobe or for tests).

## Examples
```
python index.py list ~/videos -r --show-size
python index.py meta ~/videos -r --workers 2
python index.py queue ~/videos -r --workers 2
python index.py thumbs ~/videos -r --workers 2 --time 25% --quality 4
python index.py sprites ~/videos -r --interval 8 --cols 12 --rows 8 --width 256
python index.py previews ~/videos -r --segments 9 --duration 1.0 --width 320
FFPROBE_DISABLE=1 python index.py meta ./samples
python index.py subtitles ./videos -r --model small --backend auto
```

## Exit Codes
- 0 success
- 2 invalid directory

## Development
Requires `ffprobe` (part of ffmpeg) for real metadata, PySceneDetect for scene boundary detection, and OpenCV (`opencv-python-headless`) for face detection/embeddings. Install via your package manager (e.g., `brew install ffmpeg`) and `pip install -r requirements.txt`.

### Consolidation

Legacy helper scripts (`list_mp4s.py`, `video_queue.py`) were removed; all CLI functionality now lives in `index.py` (see `index.py --help`). Tests updated accordingly.

## API Server

A lightweight FastAPI server (`api.py`) exposes core functionality for a frontend client.

Run locally:
```
pip install -r requirements.txt
uvicorn api:app --reload
```

Initial endpoints:
- `GET /health` – Liveness probe
- `GET /videos?directory=.&recursive=0` – List MP4 files
- `GET /videos/{name}/artifacts?directory=.` – Artifact presence flags
- `GET /videos/{name}/metadata?directory=.` – ffprobe metadata JSON
- `GET /report?directory=.&recursive=0` – Coverage summary (same logic as CLI `report`)
- `POST /jobs` – Submit an async job. Body shape:
  ```json
  {"task":"meta","directory":".","recursive":false,"force":false,"params":{}}
  ```
- `GET /jobs` – List jobs
- `GET /jobs/{id}` – Job detail

Supported job tasks: metadata, thumbs, sprites, previews, subtitles, phash, dupes, heatmaps, scenes, embed, listing, codecs, transcode, report

Pagination & filtering:
- `GET /videos?offset=0&limit=100&q=foo` for paged video lists.
- `GET /faces/listing?offset=0&limit=200` for paged people clusters (auto-caches for 10 minutes).

Face artifacts:
- Per video: `.artifacts/<name>.faces.signatures.json`
- Global cache: `.artifacts/.faces.index.json`

Jobs are executed in-memory (ephemeral). Future enhancements (not yet implemented): cancellation, progress streaming (SSE/WebSocket), persistent queue, ML tasks (performer recognition), dedupe.

## License
MIT (add a LICENSE file if distributing).
Test change for PR flow — no functional code changes.
