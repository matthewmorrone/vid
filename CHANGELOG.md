# Changelog

All notable changes to this project will be documented in this file. The format loosely follows Keep a Changelog.

## [Unreleased]
### Added
- `--list-commands` flag (standalone) to enumerate available CLI commands with short summaries.
- Command help summaries (`COMMAND_HELP`) for quick listing.

### Changed
- Unified command parsing description to enumerate supported commands.
- Added type hint to `cmd_rename`.

### Deprecated
- Legacy `--json` flag for `list` (still accepted, prefer `--output-format json`).

## [0.1.0] - Initial Consolidation
### Added
- Unified `index.py` multi-command CLI (list, metadata, thumbs, sprites, previews, subtitles, phash, dupes, heatmaps, scenes, embed, listing, rename, codecs, transcode, compare, batch, report, orphans, finish, tags).
- Batch rename integrated into `rename` via `--batch` (replacing prior separate command).
- Tag management system (user + performer tags) with CLI and FastAPI endpoints.
- FastAPI server (`api.py`) exposing videos, artifacts, tags, jobs, and summary endpoints.
- Perceptual hashing + duplicate detection.
- Heatmaps, scenes detection, face embedding + global listing.
- Transcode planning and codec scan utilities.
- Multi-command comma-delimited shortcut (e.g. `metadata,thumbs`).
- Comprehensive pytest suite for CLI + API behaviors.

### Notes
This version establishes the foundation prior to upcoming features: config file, metadata cache, job progress streaming, cancellation, structured logging, and enhanced search.
