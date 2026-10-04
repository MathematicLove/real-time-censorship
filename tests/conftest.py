import os
import sys
import tempfile
from pathlib import Path

# Config reads the environment at import time, so point data and logs to a
# throwaway directory before anything from `app` is imported.
_TMP = tempfile.mkdtemp(prefix="censorship-tests-")
os.environ["NSFW_DATA_DIR"] = _TMP
os.environ["NSFW_LOG_DIR"] = _TMP
os.environ["NSFW_DEVICE"] = "cpu"
os.environ.pop("NSFW_DB_URL", None)
os.environ.pop("DATABASE_URL", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
