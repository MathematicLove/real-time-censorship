import os
import uuid
from urllib.parse import quote
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

def flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")

DATA_DIR = os.environ.get("NSFW_DATA_DIR", os.path.join(ROOT, "data"))
LOG_DIR = os.environ.get("NSFW_LOG_DIR", os.path.join(ROOT, "logs"))
LOG_FILE = os.path.join(LOG_DIR, "censorship.log")
SESSION_ID = uuid.uuid4().hex[:12]

def database_url():
    direct = os.environ.get("NSFW_DB_URL") or os.environ.get("DATABASE_URL")
    if direct:
        if direct.startswith("postgres://"):
            return direct.replace("postgres://", "postgresql+psycopg://", 1)
        if direct.startswith("postgresql://"):
            return direct.replace("postgresql://", "postgresql+psycopg://", 1)
        return direct
    parts = ["NSFW_DB_HOST", "NSFW_DB_NAME", "NSFW_DB_USER", "NSFW_DB_PASSWORD"]
    if any(os.environ.get(name) for name in parts):
        host = os.environ.get("NSFW_DB_HOST", "localhost")
        port = os.environ.get("NSFW_DB_PORT", "5432")
        name = os.environ.get("NSFW_DB_NAME", "nsfw_censorship")
        user = os.environ.get("NSFW_DB_USER", "postgres")
        password = os.environ.get("NSFW_DB_PASSWORD", "")
        return "postgresql+psycopg://{0}:{1}@{2}:{3}/{4}".format(
            quote(user, safe=""), quote(password, safe=""), host, port, quote(name, safe=""))
    return "sqlite:///" + os.path.join(DATA_DIR, "censorship.db")

DB_URL = database_url()
DB_MAINTENANCE = os.environ.get("NSFW_DB_MAINTENANCE", "postgres")
DB_AUTO_CREATE = flag("NSFW_DB_AUTO_CREATE", "1")
DB_CONNECT_RETRIES = int(os.environ.get("NSFW_DB_CONNECT_RETRIES", "10"))
DB_CONNECT_DELAY = float(os.environ.get("NSFW_DB_CONNECT_DELAY", "2.0"))
DB_LOG_INTERVAL = float(os.environ.get("NSFW_DB_LOG_INTERVAL", "1.0"))
API_HOST = os.environ.get("NSFW_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("NSFW_API_PORT", "8000"))
API_CORS = os.environ.get("NSFW_API_CORS", "*")
AUTO_CAMERA = flag("NSFW_AUTO_CAMERA", "0")
CAMERA_INDEX = int(os.environ.get("NSFW_CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.environ.get("NSFW_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("NSFW_CAMERA_HEIGHT", "720"))
INFERENCE_SIZE = int(os.environ.get("NSFW_INFERENCE_SIZE", "320"))
SCORE_THRESHOLD = float(os.environ.get("NSFW_SCORE_THRESHOLD", "0.25"))
NMS_THRESHOLD = float(os.environ.get("NSFW_NMS_THRESHOLD", "0.45"))
BOX_PADDING = int(os.environ.get("NSFW_BOX_PADDING", "85"))
BLUR_MAX = int(os.environ.get("NSFW_BLUR_MAX", "301"))
BLUR_STRENGTH = int(os.environ.get("NSFW_BLUR_STRENGTH", str(BLUR_MAX)))
FORCE_DEVICE = os.environ.get("NSFW_DEVICE", "auto")

ALL_CLASSES = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

NSFW_CLASSES = [
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]
DEFAULT_CENSOR_CLASSES = list(NSFW_CLASSES)
RECORD_CLASSES = [name.strip().upper() for name in os.environ.get("NSFW_RECORD_CLASSES", ",".join(NSFW_CLASSES)).split(",") if name.strip()]