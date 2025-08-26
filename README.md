# Video Utility

Unified Python script `index.py` to:
1. List MP4 files
2. Generate ffprobe metadata JSON files
3. Run an ephemeral in-memory queue to batch metadata generation with limited concurrency
4. Build and match actor face embeddings

## Actor Recognition

Dependencies:
- [DeepFace](https://pypi.org/project/deepface/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [FFmpeg](https://ffmpeg.org/)

Examples:
```
python index.py actor-build --people-dir people --model ArcFace --detector retinaface --embeddings gallery.npy --labels labels.json --verbose
python index.py actor-match --video videos/sample.mp4 --embeddings gallery.npy --labels labels.json --model ArcFace --detector retinaface --retry-detectors mtcnn,opencv --sample-rate 1.0 --conf 0.40 --topk 3 --verbose
```

## Commands
```
python index.py list  [dir] [-r] [--json] [--show-size] [--sort name|size]
python index.py meta  [dir] [-r] [--force] [--workers N]
python index.py queue [dir] [-r] [--force] [--workers N]
python index.py thumb [dir] [-r] [--force] [--workers N] [--time middle|10|25%] [--quality 2]
python index.py sprites [dir] [-r] [--interval 10] [--width 320] [--cols 10] [--rows 10] [--quality 4] [--max-frames 0] [--force]
python index.py previews [dir] [-r] [--segments 9] [--duration 1.0] [--width 320] [--format mp4|webm] [--crf 30] [--bitrate 300k] [--workers 2] [--force] [--no-index]
python index.py subs [dir] [-r] [--model small] [--backend auto|whisper|faster-whisper|whisper.cpp] [--language xx] [--translate] [--format vtt|srt|json] [--workers 1] [--force]
python index.py batch [dir] [-r] [--tasks meta,thumb] [--max-meta 3] [--max-thumb 4] [--max-sprites 2] [--max-previews 2] [--max-subs 1] [--force]
python index.py phash [dir] [-r] [--time middle|60|25%] [--frames 5] [--algo ahash|dct] [--combine xor|majority|avg] [--workers 2] [--force] [--output-format json|text]
python index.py heatmap [dir] [-r] [--interval 5.0] [--mode brightness|motion|both] [--png] [--workers 2] [--force] [--output-format json|text]
python index.py scenes [dir] [-r] [--threshold 0.4] [--limit N] [--thumbs] [--clips] [--clip-duration 2.0] [--thumb-width 320] [--workers 2] [--force] [--output-format json|text]
python index.py faces [dir] [-r] [--workers N] [--force] [--output-format json|text]
python index.py actor-build --people-dir PEOPLE [--model ArcFace] [--detector retinaface] [--embeddings gallery.npy] [--labels labels.json] [--include-video] [--video-sample-rate 1.0] [--min-face-area 4096] [--blur-threshold 60.0]
python index.py actor-match --video INPUT.mp4 [--embeddings gallery.npy] [--labels labels.json] [--model ArcFace] [--detector retinaface] [--retry-detectors mtcnn,opencv] [--sample-rate 1.0] [--topk 3] [--conf 0.40] [--min-face-area 4096] [--blur-threshold 60.0] [--out OUTPUT.json]
python index.py codecs [dir] [-r] [--target-v h264] [--target-a aac] [--allowed-profiles list] [--workers 4] [--output-format json|text]
python index.py transcode [dir] dest [-r] [--target-v h264] [--target-a aac] [--crf 28] [--v-bitrate 3000k] [--a-bitrate 128k] [--preset medium] [--hardware none|videotoolbox] [--drop-subs] [--workers 1] [--force] [--dry-run] [--output-format json|text]
python index.py compare original.mp4 other.mp4 [--output-format json|text]
python index.py report [dir] [-r] [--output-format json|text]
```

`list`:
	Show MP4 files. Optional JSON output, human sizes, recursion, sorting.

`meta`:
	Generate ffprobe metadata files `<video>.ffprobe.json`. Skips existing unless `--force`.

`queue`:
	In-memory queue variant that processes the same metadata generation with a small worker pool (capped at 4) to prevent thrash on low-power devices.

`thumb`:
	Generate a single JPEG thumbnail per video (`<video>.mp4.jpg`). By default grabs a frame from the middle. `--time` can be seconds, percentage (e.g. `25%`), or `middle`. Uses existing metadata JSON (if present) to determine duration; otherwise falls back to a static second.

`sprites`:
	Produce a sprite sheet (`<video>.mp4.sprites.jpg`) and an accompanying JSON index (`<video>.mp4.sprites.json`) for hover scrubbing. Frames are sampled every `--interval` seconds (default 10) up to `cols*rows` (or `--max-frames` if set and smaller). Tiles are scaled to `--width` keeping aspect ratio. JSON includes grid and interval metadata for client-side lookup.

`previews`:
	Generate short hover preview clips (default 1s each) at decile positions (10%..90% of duration). Default format is webm (VP9). Outputs per-video directory `.<name>.mp4.previews/seg_XX.webm` (or .mp4 if chosen) and an index JSON `<video>.mp4.previews.json` unless `--no-index` given. Configure number of segments, clip duration, width, format, quality, and worker threads.

`subs`:
	Transcribe audio to subtitles using Whisper models. Automatic backend detection prefers `faster-whisper`, then `whisper`, else falls back to external `whisper.cpp`. Output formats: VTT (default), SRT, JSON. Use `--translate` to translate to English. Provide `--whisper-cpp-bin` and `--whisper-cpp-model` when using whisper.cpp.

`batch`:
	Ephemeral multi-task scheduler for large sets. Runs selected tasks (`--tasks` CSV) with per-type concurrency caps (`--max-<task>`). Skips artifacts that already exist unless `--force`. Dynamically balances workers across task types to avoid overload (e.g., only 1 subs job while multiple thumbs run).

`phash`:
	Compute a perceptual hash. Algorithms:
	- `ahash` (default): 32x32 average hash (1024 bits when single frame; combined via method below).
	- `dct`: classic pHash style 8x8 low-frequency DCT (64 bits).
	Multi-frame combination (`--frames > 1`):
	- `xor` (default): fast XOR of per-frame hashes.
	- `majority` / `avg`: bit vote (>50% set). (For DCT both are identical; `avg` alias.)
	Default samples 5 evenly spaced frames. Single-frame (`--frames 1`) yields the raw per-frame hash. Output `<video>.mp4.phash.json` includes algorithm, combine mode, frame positions. Compare via Hamming distance.

`heatmap`:
	Sample frames every `--interval` seconds to produce a lightweight timeline of brightness and/or motion (mean absolute frame diff) values. Writes `<video>.mp4.heatmap.json` (and optionally a small stripe visualization `<video>.mp4.heatmap.png` with `--png`). Useful for spotting dark sections, scene density, or high-action regions.

`scenes`:
        Detect scene boundaries using the PySceneDetect library's content detector. Produces `<video>.mp4.scenes.json` with marker times and scores; optionally generates per-scene thumbnail JPEGs and/or short clip previews inside a hidden directory `.<name>.mp4.scenes/`. Use `--threshold` to tune sensitivity (lower = more markers). Limit output with `--limit`.

`faces`:
        Detect faces using OpenCV's Haar cascade and generate 128-d embeddings via the OpenFace model, outputting `<video>.mp4.faces.json` with time-stamped bounding boxes and descriptors. Falls back to placeholder data when OpenCV or a readable video is unavailable.

`codecs`:
        Scan video files (mp4/mkv/mov/avi/webm/m4v) and list container, video codec/profile, audio codecs, size, duration, and whether they match target constraints. Supports JSON or text output. Useful pre-pass before deciding what to transcode.

`transcode`:
	Batch transcode incompatible (or all with `--force`) videos into a normalized MP4 (default H.264 + AAC) tree mirroring source structure. Supports CRF/preset software encoders and optional VideoToolbox hardware acceleration. Dry-run mode plans actions without encoding.

`compare`:
	Compute SSIM and PSNR between two files (original vs other) and classify quality levels.

`report`:
	Summarize library artifact coverage: counts & percentages for metadata, thumbnail, sprites, previews, subtitles, phash, heatmap, scenes. Outputs per-file presence matrix (JSON mode) or a concise table (text mode).

Set `FFPROBE_DISABLE=1` to produce stub metadata JSON without calling ffprobe (useful on systems lacking ffprobe or for tests).

## Examples
```
python index.py list ~/videos -r --show-size
python index.py meta ~/videos -r --workers 2
python index.py queue ~/videos -r --workers 2
python index.py thumb ~/videos -r --workers 2 --time 25% --quality 4
python index.py sprites ~/videos -r --interval 8 --cols 12 --rows 8 --width 256
python index.py previews ~/videos -r --segments 9 --duration 1.0 --width 320
FFPROBE_DISABLE=1 python index.py meta ./samples
python index.py subs ./videos -r --model small --backend auto --format vtt
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

Supported job tasks: meta, thumb, sprites, previews, subs, phash, heatmap, scenes, faces, codecs, transcode, report

Jobs are executed in-memory (ephemeral). Future enhancements (not yet implemented): cancellation, progress streaming (SSE/WebSocket), persistent queue, ML tasks (performer recognition), dedupe.

## License
MIT (add a LICENSE file if distributing).
Test change for PR flow — no functional code changes.
