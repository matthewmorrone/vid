import sys
import shutil
from pathlib import Path
import pytest

# Ensure project root (one level up from this file) is on sys.path for imports like `import index`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def _cleanup_repo_root_files():
    """Clean up any stray test-created video/artifact files in the repo root.

    Most tests use tmp_path, but defensive cleanup prevents polluting the
    working directory if a test accidentally omits a path argument.
    """
    before_mp4 = {p.name for p in ROOT.glob('*.mp4')}
    artifacts_dir = ROOT / '.artifacts'
    had_artifacts_dir = artifacts_dir.exists()
    before_artifacts = set()
    if had_artifacts_dir:
        before_artifacts = {p.name for p in artifacts_dir.iterdir()}
    yield
    # Remove new mp4 files
    for p in ROOT.glob('*.mp4'):
        if p.name not in before_mp4:
            try:
                p.unlink()
            except Exception:
                pass
    # Clean artifacts
    if artifacts_dir.exists():
        for p in artifacts_dir.iterdir():
            if p.name not in before_artifacts:
                try:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink()
                except Exception:
                    pass
        if not had_artifacts_dir:
            try:
                if not any(artifacts_dir.iterdir()):
                    artifacts_dir.rmdir()
            except Exception:
                pass
