import argparse
import json
import os
from pathlib import Path

fMEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".mp3", ".m4a", ".wav", ".flac"}

def cover_path_for(media_path: Path) -> Path:
    d = media_path.parent / ".preview"
    d.mkdir(exist_ok=True)
    return d / f"{media_path.stem}.jpg"

def hover_path_for(media_path: Path) -> Path:
    d = media_path.parent / ".preview"
    d.mkdir(exist_ok=True)
    return d / f"{media_path.stem}.hover.webm"

def metadata_path_for(media_path: Path) -> Path:
    d = media_path.parent / ".preview"
    d.mkdir(exist_ok=True)
    return d / f"{media_path.stem}.meta.json"

def subtitles_path_candidates(media_path: Path):
    b = media_path.with_suffix("")
    return [Path(f"{b}.srt"), Path(f"{b}.vtt")]

def find_subtitles(media_path: Path):
    for p in subtitles_path_candidates(media_path):
        if p.exists():
            return p
    return None

def find_misnamed_assets(media_path: Path):
    folder = media_path.parent
    stem = media_path.stem
    wrong = []
    for cand in folder.glob("*"):
        if not cand.is_file():
            continue
        if cand.name.lower() in {"._", ".ds_store"}:
            continue
        name = cand.name.lower()
        if name in {f"{stem}.cover.jpg", f"{stem}.preview.jpg", f"{stem}.thumb.jpg"}:
            wrong.append((cand, cover_path_for(media_path)))
        if name in {f"{stem}.hover.mp4", f"{stem}.preview.webm", f"{stem}.hoverpreview.webm"}:
            wrong.append((cand, hover_path_for(media_path)))
        if name in {f"{stem}.json", f"{stem}.metadata.json", f"{stem}.meta"}:
            wrong.append((cand, metadata_path_for(media_path)))
    for cand in (folder / ".covers").glob("*") if (folder / ".covers").exists() else []:
        if cand.is_file() and cand.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            wrong.append((cand, cover_path_for(media_path)))
    for cand in (folder / ".preview").glob("*") if (folder / ".preview").exists() else []:
        if cand.is_file():
            if cand.name == f"{stem}.preview.jpg":
                wrong.append((cand, cover_path_for(media_path)))
            if cand.name == f"{stem}.hover.mp4":
                wrong.append((cand, hover_path_for(media_path)))
            if cand.name == f"{stem}.metadata.json":
                wrong.append((cand, metadata_path_for(media_path)))
    return wrong

def apply_moves(moves, verbose=False):
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(json.dumps({"action":"rename","src":str(src),"dst":str(dst)}))
        os.replace(src, dst)

def audit(root: Path, exts, do_rename=False, verbose=False):
    for media in root.rglob("*"):
        if not media.is_file():
            continue
        if media.suffix.lower() not in exts:
            continue
        cover = cover_path_for(media)
        hover = hover_path_for(media)
        meta = metadata_path_for(media)
        sub = find_subtitles(media)
        moves = []
        for src, dst in find_misnamed_assets(media):
            if src != dst:
                moves.append((src, dst))
        if do_rename and moves:
            apply_moves(moves, verbose)
        exists = {
            "cover": cover.exists(),
            "hover": hover.exists(),
            "meta": meta.exists(),
            "subs": sub is not None
        }
        missing = [k for k, v in exists.items() if not v]
        out = {
            "video": str(media),
            "assets": {
                "cover": str(cover),
                "hover": str(hover),
                "meta": str(meta),
                "subs": str(sub) if sub else None
            },
            "exists": exists,
            "missing": missing
        }
        print(json.dumps(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ext", action="append", default=[".mp4", ".mkv", ".webm", ".mov", ".mp3", ".m4a", ".wav", ".flac"]) 
    ap.add_argument("--rename", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    audit(Path(a.root), set(x.lower() for x in a.ext), do_rename=a.rename, verbose=a.verbose)

if __name__ == "__main__":
    main()