"""
DeepGuard Configuration Module
Constants and environment variable overrides as specified in Section 5.
"""

import os
from pathlib import Path

# Pipeline constants
T: int = int(os.environ.get("DEEPGUARD_T", "16"))
IMG_SIZE: int = int(os.environ.get("DEEPGUARD_IMG_SIZE", "224"))
FEAT_DIM: int = int(os.environ.get("DEEPGUARD_FEAT_DIM", "1280"))
HIDDEN: int = int(os.environ.get("DEEPGUARD_HIDDEN", "256"))
THRESHOLD: float = float(os.environ.get("DEEPGUARD_THRESHOLD", "0.5"))

# Upload & Server constraints
MAX_UPLOAD_MB: int = int(os.environ.get("DEEPGUARD_MAX_UPLOAD_MB", "200"))
ALLOWED_EXT_STR: str = os.environ.get("DEEPGUARD_ALLOWED_EXT", ".mp4 .avi .mov .mkv .webm")
ALLOWED_EXT: set[str] = {ext.strip().lower() for ext in ALLOWED_EXT_STR.split() if ext.strip()}

# Ollama settings
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

# File and database paths
BASE_DIR: Path = Path(__file__).resolve().parent
WEIGHTS_PATH: str = os.environ.get("DEEPGUARD_WEIGHTS_PATH", str(BASE_DIR / "weights.pt"))
UPLOAD_DIR: Path = BASE_DIR / "uploads"
DB_PATH: Path = BASE_DIR / "db.json"
STATIC_DIR: Path = BASE_DIR / "static"

# Ensure ffmpeg and exiftool directories are included in PATH if present
_extra_paths = [
    r"C:\Users\GTU-SET\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
    r"C:\Users\GTU-SET\AppData\Local\Programs\ExifTool",
    r"C:\Users\GTU-SET\AppData\Local\Programs\Ollama",
]
_path_elements = os.environ.get("PATH", "").split(os.pathsep)
for _p in _extra_paths:
    if os.path.exists(_p) and _p not in _path_elements:
        os.environ["PATH"] = f"{_p}{os.pathsep}{os.environ.get('PATH', '')}"

