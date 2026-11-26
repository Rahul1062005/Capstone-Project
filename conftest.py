import os
import sys
from pathlib import Path

# Insert project root into sys.path for test discovery
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep Numba JIT cache inside the workspace so restricted/sandboxed runs never stall
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".pytest_cache" / "numba"))

