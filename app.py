"""
DeepGuard FastAPI Application & Background Worker
Implements:
- REST API specified in Section 7
- Health check endpoint (/api/health)
- Stats aggregation (/api/stats)
- Video upload, listing, retrieval, deletion, and streaming
- Single-worker queue with threading.Lock for serial model inference
- Graceful degradation and comprehensive error handling
"""

import os
import re
import json
import uuid
import queue
import shutil
import logging
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepguard")

app = FastAPI(
    title="DeepGuard - Deepfake Video Detection Platform",
    description="Multimodal deepfake video detection backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure runtime directories exist
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Database concurrency lock and state
_db_lock = threading.Lock()
_worker_queue: "queue.Queue[str]" = queue.Queue()
_model_inference_lock = threading.Lock()


def load_db() -> Dict[str, Dict[str, Any]]:
    """Loads records from db.json safely."""
    with _db_lock:
        if not config.DB_PATH.exists():
            return {}
        try:
            with open(config.DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    return {item["id"]: item for item in data if "id" in item}
                return {}
        except Exception as e:
            logger.error("Error reading db.json: %s", e)
            return {}


def save_db(data: Dict[str, Dict[str, Any]]) -> None:
    """Atomically writes records to db.json."""
    with _db_lock:
        temp_path = config.DB_PATH.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(config.DB_PATH)
        except Exception as e:
            logger.error("Error saving db.json: %s", e)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


def update_record(video_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates fields of a single record and persists immediately."""
    db = load_db()
    if video_id in db:
        db[video_id].update(updates)
        save_db(db)
        return db[video_id]
    return None


def sanitize_filename(filename: str) -> str:
    """Sanitizes uploaded filename against directory traversal and dangerous characters."""
    name = Path(filename).name
    # Keep only alphanumeric, underscores, hyphens, and dots
    safe_name = re.sub(r"[^a-zA-Z0-9_\.\-]", "_", name)
    # Strip leading dots or multiple dots
    safe_name = re.sub(r"^\.+", "", safe_name)
    if not safe_name:
        safe_name = f"video_{uuid.uuid4().hex[:6]}.mp4"
    return safe_name


def ensure_browser_preview(file_path: Path) -> Path:
    """
    Ensures a fast, browser-compatible H.264 preview with +faststart exists.
    Transcodes non-standard codecs (mpeg4, avi, mkv, etc.) to a web-safe preview.
    """
    if not file_path.exists():
        return file_path

    # Check if a converted preview already exists
    preview_path = file_path.with_name(f"{file_path.stem}_webpreview.mp4")
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", str(file_path),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "26",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(preview_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            if res.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0:
                logger.info("Generated browser-compatible preview: %s", preview_path.name)
                return preview_path
        except Exception as e:
            logger.warning("Could not generate web preview for %s: %s", file_path, e)

    return file_path


# -----------------------------------------------------------------------------
# Background Worker Thread
# -----------------------------------------------------------------------------
def background_analysis_worker() -> None:
    """Single worker consuming video IDs sequentially to avoid GPU/CPU contention."""
    logger.info("DeepGuard background analysis worker started.")
    while True:
        video_id = _worker_queue.get()
        if video_id is None:
            break

        db = load_db()
        record = db.get(video_id)
        if not record:
            _worker_queue.task_done()
            continue

        file_path = Path(record["path"])
        if not file_path.exists():
            update_record(video_id, {
                "status": "error",
                "error": f"Uploaded file not found on disk: {file_path.name}",
            })
            _worker_queue.task_done()
            continue

        # Step 1: Update status to analyzing
        update_record(video_id, {"status": "analyzing"})
        logger.info("Analyzing video id=%s file=%s", video_id, file_path.name)

        try:
            # Metadata extraction
            meta = pipeline.metadata(file_path)

            # Ensure browser-compatible preview stream is ready
            ensure_browser_preview(file_path)

            # Single inference with lock
            with _model_inference_lock:
                probability, trained, frame_scores, frame_analyses = pipeline.predict(file_path)

            verdict = "FAKE" if probability >= config.THRESHOLD else "REAL"

            # Ollama natural language explanation
            explanation = pipeline.explain_prediction(
                verdict=verdict,
                probability=probability,
                metadata_dict=meta,
                trained=trained,
            )

            # Mark completed
            update_record(video_id, {
                "status": "done",
                "meta": meta,
                "score": round(probability, 4),
                "trained": trained,
                "verdict": verdict,
                "explanation": explanation,
                "frame_scores": frame_scores,
                "frame_analyses": frame_analyses,
                "error": None,
            })
            logger.info("Completed analysis for video id=%s: verdict=%s prob=%.4f", video_id, verdict, probability)

        except Exception as e:
            logger.exception("Analysis failed for video id=%s: %s", video_id, e)
            update_record(video_id, {
                "status": "error",
                "error": str(e),
            })
        finally:
            _worker_queue.task_done()


# Start background worker thread as daemon
_worker_thread = threading.Thread(target=background_analysis_worker, daemon=True)
_worker_thread.start()


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = config.STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
def get_health():
    """
    Checks availability of FFmpeg, FFprobe, ExifTool, Ollama, and trained model weights.
    Returns booleans for each component.
    """
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    exiftool_ok = shutil.which("exiftool") is not None
    weights_ok = Path(config.WEIGHTS_PATH).exists()

    ollama_ok = False
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0)
        ollama_ok = (r.status_code == 200)
    except Exception:
        ollama_ok = False

    return {
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "exiftool": exiftool_ok,
        "ollama": ollama_ok,
        "weights": weights_ok,
    }


@app.get("/api/stats")
def get_stats():
    """
    Computes summary metrics: total, analyzed, fake, real, avg_score, errors.
    """
    db = load_db()
    total = len(db)
    analyzed = 0
    fake = 0
    real = 0
    scores: List[float] = []
    errors = 0

    for item in db.values():
        status_val = item.get("status")
        if status_val == "done":
            analyzed += 1
            verdict = item.get("verdict")
            if verdict == "FAKE":
                fake += 1
            elif verdict == "REAL":
                real += 1
            score = item.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    pass
        elif status_val == "error":
            errors += 1

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "total": total,
        "analyzed": analyzed,
        "fake": fake,
        "real": real,
        "avg_score": avg_score,
        "errors": errors,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives video upload, enforces extension and size checks,
    persists file and db record, and enqueues background analysis.
    """
    filename = file.filename or "unknown.mp4"
    ext = Path(filename).suffix.lower()

    # Rule: Extension check
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {config.ALLOWED_EXT_STR}",
        )

    # Generate 8-character ID
    video_id = uuid.uuid4().hex[:8]
    safe_name = sanitize_filename(filename)
    stored_filename = f"{video_id}_{safe_name}"
    target_path = config.UPLOAD_DIR / stored_filename

    # Read and stream chunk by chunk while enforcing MAX_UPLOAD_MB
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0

    try:
        with open(target_path, "wb") as f_out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    f_out.close()
                    if target_path.exists():
                        target_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum upload size of {config.MAX_UPLOAD_MB} MB",
                    )
                f_out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Build DB record
    record: Dict[str, Any] = {
        "id": video_id,
        "name": safe_name,
        "path": str(target_path),
        "status": "queued",
        "meta": None,
        "score": None,
        "trained": None,
        "verdict": None,
        "explanation": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    db = load_db()
    db[video_id] = record
    save_db(db)

    # Enqueue analysis
    _worker_queue.put(video_id)

    return {"id": video_id}


@app.post("/api/load-sample/{sample_type}")
def load_sample(sample_type: str):
    """
    Loads a verified benchmark test sample ('real' or 'fake') from sample_test_videos/
    directly into DeepGuard for immediate forensic verification.
    """
    sample_type = sample_type.lower().strip()
    sample_dir = config.BASE_DIR / "sample_test_videos"
    if sample_type == "real":
        source_file = sample_dir / "REAL_authentic_celebrity_sample.mp4"
        display_name = "REAL_authentic_celebrity_sample.mp4"
    elif sample_type == "fake":
        source_file = sample_dir / "FAKE_deepfake_celebrity_sample.mp4"
        display_name = "FAKE_deepfake_celebrity_sample.mp4"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sample type '{sample_type}'. Allowed types are 'real' or 'fake'."
        )

    if not source_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample video file not found at {source_file}."
        )

    video_id = uuid.uuid4().hex[:8]
    stored_filename = f"{video_id}_{display_name}"
    target_path = config.UPLOAD_DIR / stored_filename

    shutil.copyfile(source_file, target_path)

    record: Dict[str, Any] = {
        "id": video_id,
        "name": display_name,
        "path": str(target_path),
        "status": "queued",
        "meta": None,
        "score": None,
        "trained": None,
        "verdict": None,
        "explanation": None,
        "frame_scores": None,
        "frame_analyses": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    db = load_db()
    db[video_id] = record
    save_db(db)

    _worker_queue.put(video_id)
    return {"id": video_id, "name": display_name, "sample_type": sample_type}


@app.get("/api/videos")
def list_videos():
    """Lists all video records, sorted newest first."""
    db = load_db()
    records = list(db.values())
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


@app.get("/api/videos/{video_id}")
def get_video_record(video_id: str):
    """Retrieves a single record by its 8-char id."""
    db = load_db()
    if video_id not in db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video id '{video_id}' not found.")
    return db[video_id]


@app.get("/api/video/{video_id}")
def stream_video(video_id: str):
    """Streams stored video file for browser preview with range support."""
    db = load_db()
    if video_id not in db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video id '{video_id}' not found.")
    record = db[video_id]
    file_path = Path(record["path"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file does not exist on disk.")

    serve_path = ensure_browser_preview(file_path)
    return FileResponse(serve_path, media_type="video/mp4")


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    """Removes a video record from db.json and deletes the stored file from disk."""
    db = load_db()
    if video_id not in db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video id '{video_id}' not found.")

    record = db.pop(video_id)
    save_db(db)

    file_path = Path(record.get("path", ""))
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning("Could not delete file %s: %s", file_path, e)

    # Clean up webpreview file if present
    preview_path = file_path.with_name(f"{file_path.stem}_webpreview.mp4")
    if preview_path.exists():
        try:
            preview_path.unlink()
        except Exception:
            pass

    return {"deleted": True, "id": video_id}
