import sys
from pathlib import Path
# Ensure project root (one level up from this file) is on sys.path for imports like `import index`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
