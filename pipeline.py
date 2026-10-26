"""
DeepGuard Processing Pipeline
Implements:
1. Metadata extraction (ffprobe + exiftool) with graceful fallbacks
2. Video frame extraction (uniform 16 frames, face detection & crop, ImageNet norm)
3. Audio frame extraction (FFmpeg/librosa, 128-band mel spectrograms, ImageNet norm)
4. EfficientNet-B0 frozen feature extraction (T=16, 2560-d)
5. BiLSTM + MLP Detector architecture
6. Predict function (loads weights.pt if available, else Demo Mode)
7. LLM natural-language forensic explanation via Ollama
"""

import os
import re
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Tuple, Optional, Dict, List

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import config

# Global singletons / caches
_DEVICE: Optional[torch.device] = None
_BACKBONE: Optional[nn.Module] = None
_FACE_CASCADE: Optional[Any] = None
_DETECTOR: Optional[nn.Module] = None
_DETECTOR_TRAINED: Optional[bool] = None

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def get_backbone(device: Optional[torch.device] = None) -> nn.Module:
    global _BACKBONE
    if device is None:
        device = get_device()
    if _BACKBONE is None:
        # Pretrained EfficientNet-B0 with ImageNet weights, frozen backbone in eval mode
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _BACKBONE = model.to(device)
    return _BACKBONE


def get_face_cascade() -> Optional[Any]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        local_cascade = config.BASE_DIR / "data" / "haarcascade_frontalface_default.xml"
        if local_cascade.exists() and hasattr(cv2, "CascadeClassifier"):
            try:
                clf = cv2.CascadeClassifier(str(local_cascade))
                if not clf.empty():
                    _FACE_CASCADE = clf
            except Exception:
                pass
        if _FACE_CASCADE is None and hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                try:
                    clf = cv2.CascadeClassifier(cascade_path)
                    if not clf.empty():
                        _FACE_CASCADE = clf
                except Exception:
                    pass
    return _FACE_CASCADE


# -----------------------------------------------------------------------------
# 6.1 Metadata Extraction
# -----------------------------------------------------------------------------
def run_command_json(cmd: List[str]) -> Optional[Any]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:
        pass
    return None


def metadata(path: str | Path) -> Dict[str, Any]:
    """
    Extracts container, video/audio stream, and EXIF metadata.
    Flags suspicious indicators according to Section 6.1 rules.
    Gracefully degrades if ffprobe or exiftool is unavailable.
    """
    path = Path(path).resolve()
    file_size_mb = round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0

    duration_s = 0.0
    codec = "unknown"
    resolution = "unknown"
    fps = 0.0
    has_audio = False
    format_tags: Dict[str, Any] = {}
    stream_tags: Dict[str, Any] = {}
    video_stream_found = False

    # Check ffprobe
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        ffprobe_data = run_command_json([
            ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ])
        if ffprobe_data and isinstance(ffprobe_data, dict):
            fmt = ffprobe_data.get("format", {})
            format_tags = fmt.get("tags", {})
            try:
                duration_s = round(float(fmt.get("duration", 0.0)), 2)
            except (ValueError, TypeError):
                duration_s = 0.0

            streams = ffprobe_data.get("streams", [])
            for stream in streams:
                codec_type = stream.get("codec_type")
                if codec_type == "video" and not video_stream_found:
                    video_stream_found = True
                    codec = stream.get("codec_name", "unknown")
                    width = stream.get("width")
                    height = stream.get("height")
                    if width and height:
                        resolution = f"{width}x{height}"
                    stream_tags.update(stream.get("tags", {}))

                    # Parse FPS
                    r_fps = stream.get("r_frame_rate", "")
                    if "/" in r_fps:
                        try:
                            num, den = map(float, r_fps.split("/"))
                            if den > 0:
                                fps = round(num / den, 2)
                        except Exception:
                            fps = 0.0
                    if duration_s == 0.0:
                        try:
                            duration_s = round(float(stream.get("duration", 0.0)), 2)
                        except Exception:
                            pass
                elif codec_type == "audio":
                    has_audio = True

    # Fallback to OpenCV if ffprobe did not provide video properties
    if resolution == "unknown" or fps == 0.0 or duration_s == 0.0:
        try:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                c_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                c_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                c_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                c_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if fps == 0.0 and c_fps > 0:
                    fps = round(c_fps, 2)
                if resolution == "unknown" and c_w > 0 and c_h > 0:
                    resolution = f"{c_w}x{c_h}"
                if duration_s == 0.0 and c_fps > 0 and c_frames > 0:
                    duration_s = round(c_frames / c_fps, 2)
        except Exception:
            pass

    # Check exiftool
    exif_raw: Dict[str, Any] = {}
    exiftool_bin = shutil.which("exiftool")
    if exiftool_bin:
        exif_data = run_command_json([exiftool_bin, "-j", str(path)])
        if exif_data and isinstance(exif_data, list) and len(exif_data) > 0:
            exif_raw = exif_data[0]

    # Clean EXIF: exclude internal tags, keep first 12, truncate values to 60 chars
    excluded_keys = {"SourceFile", "ExifToolVersion", "Directory", "FileName", "FilePermissions"}
    cleaned_exif: Dict[str, str] = {}
    for k, v in exif_raw.items():
        if k in excluded_keys:
            continue
        val_str = str(v)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        cleaned_exif[k] = val_str
        if len(cleaned_exif) >= 12:
            break

    # Build flags according to Section 6.1 rules
    flags: List[str] = []

    # Rule 1: No creation_time and no EXIF CreateDate
    has_creation_time = False
    for tags in (format_tags, stream_tags):
        for k in tags:
            if "creation_time" in k.lower():
                has_creation_time = True
                break
    for k in exif_raw:
        if k.lower() in ("createdate", "creationdate", "datetimeoriginal", "trackcreatedate", "mediacreatedate"):
            has_creation_time = True
            break
    if not has_creation_time:
        flags.append("no creation timestamp")

    # Rule 2: Encoder tag contains lavf, ffmpeg or handbrake -> "re-encoded by <encoder>"
    encoder_val: Optional[str] = None
    all_tags = {**format_tags, **stream_tags}
    for k, v in all_tags.items():
        if any(term in k.lower() for term in ("encoder", "writing_application", "software", "handler")):
            val = str(v)
            if any(target in val.lower() for target in ("lavf", "ffmpeg", "handbrake")):
                encoder_val = val
                break
    if not encoder_val:
        for k, v in exif_raw.items():
            if any(term in k.lower() for term in ("encoder", "software", "compressorname")):
                val = str(v)
                if any(target in val.lower() for target in ("lavf", "ffmpeg", "handbrake")):
                    encoder_val = val
                    break
    if encoder_val:
        flags.append(f"re-encoded by {encoder_val}")

    # Rule 3: No audio stream
    if not has_audio:
        flags.append("no audio track")

    # Rule 4: No EXIF Make and no Model
    has_camera_info = any(k in exif_raw for k in ("Make", "Model", "DeviceModelName", "CameraModelName"))
    if not has_camera_info:
        flags.append("no camera make/model")

    return {
        "duration_s": duration_s,
        "size_mb": file_size_mb,
        "codec": codec,
        "resolution": resolution,
        "fps": fps,
        "has_audio": has_audio,
        "exif": cleaned_exif,
        "flags": flags,
    }


# -----------------------------------------------------------------------------
# 6.2 Video Frame Extraction
# -----------------------------------------------------------------------------
def video_frames(path: str | Path, t_steps: int = config.T) -> np.ndarray:
    """
    Samples T frames evenly, detects faces via Haar Cascade, crops largest face
    with 20% margin, resizes to 224x224, and normalizes with ImageNet mean/std.
    Output: ndarray of shape (T, 3, 224, 224), float32.
    """
    path_str = str(path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return np.zeros((t_steps, 3, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Read manually to count frames if metadata count is invalid
        frames_cache = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames_cache.append(f)
        cap.release()
        total_frames = len(frames_cache)
        if total_frames == 0:
            return np.zeros((t_steps, 3, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
        indices = np.linspace(0, total_frames - 1, t_steps, dtype=int)
        sampled_raw = [frames_cache[idx] for idx in indices]
    else:
        indices = np.linspace(0, max(0, total_frames - 1), t_steps, dtype=int)
        sampled_raw = []
        last_good_frame = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                last_good_frame = frame
                sampled_raw.append(frame)
            else:
                if last_good_frame is not None:
                    sampled_raw.append(last_good_frame)
                else:
                    sampled_raw.append(np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8))
        cap.release()

    face_cascade = get_face_cascade()
    processed_frames: List[np.ndarray] = []

    for frame in sampled_raw:
        if frame is None or frame.size == 0:
            frame = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Detect faces with adaptive downscaling for high-resolution performance
        faces = ()
        if face_cascade is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            max_d = max(h, w)
            if max_d > 480:
                scale = 480.0 / max_d
                small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                min_sz = max(24, int(30 * scale))
                det_faces = face_cascade.detectMultiScale(
                    small_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(min_sz, min_sz),
                )
                faces = [(int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale)) for (fx, fy, fw, fh) in det_faces]
            else:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )

        if len(faces) > 0:
            # Pick largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # 20% margin
            mx = int(0.20 * fw)
            my = int(0.20 * fh)
            x1 = max(0, fx - mx)
            y1 = max(0, fy - my)
            x2 = min(w, fx + fw + mx)
            y2 = min(h, fy + fh + my)
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                resized = cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(frame_rgb, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        else:
            # Fallback to full frame
            resized = cv2.resize(frame_rgb, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Normalize with ImageNet mean/std
        norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose to channel-first (3, 224, 224)
        norm_ch_first = norm.transpose(2, 0, 1)
        processed_frames.append(norm_ch_first)

    return np.stack(processed_frames, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# 6.3 Audio Frame Extraction
# -----------------------------------------------------------------------------
def audio_frames(path: str | Path, t_steps: int = config.T) -> np.ndarray:
    """
    Extracts audio to mono 16kHz WAV, splits into T segments, computes 128-band mel-spectrogram,
    converts to dB, normalizes, resizes to 224x224, replicates to 3 channels, and normalizes
    with ImageNet mean/std.
    Output: ndarray of shape (T, 3, 224, 224), float32.
    """
    import librosa

    zero_tensor = np.zeros((t_steps, 3, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
    path = Path(path).resolve()
    if not path.exists():
        return zero_tensor

    ffmpeg_bin = shutil.which("ffmpeg")
    y: Optional[np.ndarray] = None
    sr = 16000

    # Try FFmpeg extraction first
    if ffmpeg_bin:
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Close file descriptor immediately on Windows so FFmpeg can write
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(path),
                "-vn",
                "-ac", "1",
                "-ar", str(sr),
                temp_wav,
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)
            if res.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 100:
                try:
                    from scipy.io import wavfile
                    w_sr, w_data = wavfile.read(temp_wav)
                    if w_data.dtype == np.int16:
                        y = w_data.astype(np.float32) / 32768.0
                    elif w_data.dtype == np.int32:
                        y = w_data.astype(np.float32) / 2147483648.0
                    else:
                        y = w_data.astype(np.float32)
                except Exception:
                    y, _ = librosa.load(temp_wav, sr=sr, mono=True)
        except Exception:
            y = None
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    # Fallback to direct librosa load only if FFmpeg was missing entirely
    if not ffmpeg_bin and (y is None or len(y) == 0):
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=True)
        except Exception:
            y = None

    if y is None or len(y) == 0:
        return zero_tensor

    # Split waveform into T equal segments
    n_samples = len(y)
    seg_len = max(1, n_samples // t_steps)
    processed_segs: List[np.ndarray] = []

    for i in range(t_steps):
        start = i * seg_len
        end = (i + 1) * seg_len if i < t_steps - 1 else n_samples
        seg = y[start:end]
        if len(seg) < 2048:
            seg = np.pad(seg, (0, 2048 - len(seg)), mode="constant")

        # 128-band mel-spectrogram
        try:
            mel = librosa.feature.melspectrogram(
                y=seg,
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            m_min, m_max = float(mel_db.min()), float(mel_db.max())
            if m_max - m_min > 1e-6:
                mel_norm = (mel_db - m_min) / (m_max - m_min)
            else:
                mel_norm = np.zeros_like(mel_db, dtype=np.float32)
        except Exception:
            mel_norm = np.zeros((128, 128), dtype=np.float32)

        # Resize to 224x224
        mel_resized = cv2.resize(mel_norm, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        # Replicate to 3 channels (224, 224, 3)
        mel_3ch = np.stack([mel_resized, mel_resized, mel_resized], axis=-1)
        # ImageNet normalization
        mel_norm_img = (mel_3ch.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose to channel-first (3, 224, 224)
        processed_segs.append(mel_norm_img.transpose(2, 0, 1))

    return np.stack(processed_segs, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# 6.4 Feature Extraction
# -----------------------------------------------------------------------------
def features(path: str | Path, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extracts video and audio features using frozen EfficientNet-B0 backbone.
    Concatenates per time step to produce Tensor of shape (T, 2560).
    """
    if device is None:
        device = get_device()

    v_arr = video_frames(path)
    a_arr = audio_frames(path)

    v_t = torch.from_numpy(v_arr).to(device)
    a_t = torch.from_numpy(a_arr).to(device)

    backbone = get_backbone(device)

    with torch.no_grad():
        t_len = v_t.shape[0]
        # Batched single pass for both visual and audio frames
        combined_in = torch.cat([v_t, a_t], dim=0)  # (2*T, 3, 224, 224)
        all_out = backbone.features(combined_in)
        all_out = backbone.avgpool(all_out)
        all_feats = torch.flatten(all_out, 1)  # (2*T, 1280)

        v_feats = all_feats[:t_len]
        a_feats = all_feats[t_len:]

        combined = torch.cat([v_feats, a_feats], dim=-1)  # (T, 2560)

    return combined.cpu()


# -----------------------------------------------------------------------------
# 6.5 Classifier Architecture – Detector
# -----------------------------------------------------------------------------
# 6.5 Classifier Architecture – Attention Detector & Frame Scoring
# -----------------------------------------------------------------------------
class Detector(nn.Module):
    """
    BiLSTM + temporal attention pooling + MLP classifier.
    Input: (B, T, 2560)
    BiLSTM: (B, T, 512) -> attention pooling -> (B, 512)
    MLP: 512 -> 128 -> ReLU -> Dropout(0.3) -> 1 -> logit
    Frame Head: computes per-frame manipulation scores
    """
    def __init__(self, input_dim: int = config.FEAT_DIM * 2, hidden_dim: int = config.HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2560)
        lstm_out, _ = self.lstm(x)  # (B, T, 512)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        pooled = torch.sum(attn_w * lstm_out, dim=1)  # (B, 512)
        logits = self.mlp(pooled).squeeze(-1)  # (B,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_frame_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Computes frame-level manipulation probability: (B, T)."""
        lstm_out, _ = self.lstm(x)  # (B, T, 512)
        frame_logits = self.mlp(lstm_out).squeeze(-1)  # (B, T)
        return torch.sigmoid(frame_logits)


def extract_frame_thumbnails(path: str | Path, t_steps: int = config.T) -> List[str]:
    """
    Extracts 16 lightweight base64 JPEG thumbnails for visual filmstrip inspection in the UI.
    """
    import base64
    path_str = str(path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = t_steps
    indices = np.linspace(0, max(0, total_frames - 1), t_steps, dtype=int)

    thumbnails: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            thumb = cv2.resize(frame, (120, 120), interpolation=cv2.INTER_AREA)
        else:
            thumb = np.zeros((120, 120, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        b64 = base64.b64encode(buf).decode('ascii')
        thumbnails.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return thumbnails


def get_detector(weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[nn.Module, bool]:
    """
    Returns (detector_instance, is_trained)
    Lazy loads and checks if trained weights exist.
    """
    global _DETECTOR, _DETECTOR_TRAINED
    if device is None:
        device = get_device()
    if weights_path is None:
        weights_path = config.WEIGHTS_PATH

    weights_file = Path(weights_path)
    trained = weights_file.exists()

    if _DETECTOR is None or _DETECTOR_TRAINED != trained:
        model = Detector().to(device)
        if trained:
            try:
                state_dict = torch.load(weights_file, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[Warning] Failed to load weights from {weights_file}: {e}. Running in Demo Mode.")
                trained = False
        model.eval()
        _DETECTOR = model
        _DETECTOR_TRAINED = trained

    return _DETECTOR, _DETECTOR_TRAINED


# -----------------------------------------------------------------------------
# 6.6 Prediction & Frame-by-Frame Visibility
# -----------------------------------------------------------------------------
def predict(path: str | Path, device: Optional[torch.device] = None) -> Tuple[float, bool, List[float], List[Dict[str, Any]]]:
    """
    Performs multimodal inference on a video path.
    Returns:
      (probability: float, trained_flag: bool, frame_scores: List[float], frame_analyses: List[Dict[str, Any]])
    """
    if device is None:
        device = get_device()

    detector, trained = get_detector(config.WEIGHTS_PATH, device)
    feats = features(path, device=device)  # (T, 2560)
    feats_batch = feats.unsqueeze(0).to(device)  # (1, T, 2560)

    with torch.no_grad():
        prob = detector.predict_proba(feats_batch)
        probability = float(prob.item())
        frame_scores_t = detector.get_frame_scores(feats_batch)[0]  # (T,)
        frame_scores = [round(float(s), 4) for s in frame_scores_t.cpu().tolist()]

    # Extract thumbnails for visual filmstrip
    thumbnails = extract_frame_thumbnails(path, t_steps=len(frame_scores))

    # Generate frame-by-frame diagnostic reasons
    frame_analyses: List[Dict[str, Any]] = []
    for i, s in enumerate(frame_scores):
        if s < 0.25:
            reason = "Natural biological motion; clean facial boundaries & authentic acoustic timing"
            verdict = "REAL"
        elif s < 0.50:
            reason = "Authentic facial dynamics with minor natural compression; no synthetic manipulation detected"
            verdict = "REAL"
        elif s < 0.75:
            reason = "Moderate anomaly; subtle boundary warping or articulatory desynchronization detected"
            verdict = "FAKE"
        else:
            reason = "High-confidence deepfake artifact; spatial facial warping, blending seam, or cloned voice pattern"
            verdict = "FAKE"

        frame_analyses.append({
            "frame_idx": i + 1,
            "score": s,
            "verdict": verdict,
            "reason": reason,
            "thumbnail": thumbnails[i] if i < len(thumbnails) else None,
        })

    return probability, trained, frame_scores, frame_analyses


# -----------------------------------------------------------------------------
# 6.7 LLM Forensic Explanation (Ollama)
# -----------------------------------------------------------------------------
def explain_prediction(
    verdict: str,
    probability: float,
    metadata_dict: Dict[str, Any],
    trained: bool,
) -> str:
    """
    Dispatches request to local Ollama API to generate an analytical forensic report.
    Returns graceful error message if Ollama is unreachable.
    """
    prompt = f"""You are a specialized multimedia forensic analyst for the DeepGuard video detection platform.
Model description: Pretrained EfficientNet-B0 visual & audio feature extractor with BiLSTM temporal aggregation and MLP classifier.
Detector Result:
- Verdict: {verdict}
- Fake Probability: {probability:.4f} ({probability * 100:.1f}%)
- Model State: {'Trained weights loaded' if trained else 'DEMO MODE (untrained model)'}

Video Metadata & Forensics:
- Duration: {metadata_dict.get('duration_s', 'unknown')}s
- Resolution: {metadata_dict.get('resolution', 'unknown')}
- FPS: {metadata_dict.get('fps', 'unknown')}
- Codec: {metadata_dict.get('codec', 'unknown')}
- Audio Track: {'Present' if metadata_dict.get('has_audio') else 'None'}
- Informational Flags: {', '.join(metadata_dict.get('flags', [])) or 'None'}
- Camera EXIF: {metadata_dict.get('exif', {})}

Instructions:
Write a comprehensive, professional laboratory forensic report:
1. Executive Verdict: State the final determination clearly with probability assessment.
2. Analytical Reasons: Detail 3 key analytical findings regarding facial boundary coherence, unnatural temporal jitter across the 16 sampled frames, and audio-visual synchronization.
3. Confidence Caveats & Limitations: Explicitly outline laboratory caveats (e.g., compressed web video artifacts, facial lighting variations).
4. Conclusion: Provide a definitive concluding forensic summary.
Ensure all sections are completely written with a full final conclusion. Do not leave any sentence incomplete.
CRITICAL FORMATTING RULES:
- Absolutely DO NOT use asterisks (*) or markdown bold (**).
- Do NOT use emojis, smileys, or decorative unicode symbols.
- Use clean plain text headers and standard hyphens (-) for bullet points.
{'CRITICAL NOTE: The detector is untrained (Demo Mode); explicitly state that this score is purely demonstration and not legally or forensically reliable.' if not trained else ''}
Format as clean, professional laboratory forensic text."""

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 800,
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            raw_text = data.get("response", "").strip() or "(No explanation generated by Ollama)"
            # Completely strip all asterisks (* and **)
            clean_text = raw_text.replace("*", "")
            # Remove emojis and non-standard unicode symbols
            clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
            clean_text = re.sub(r'[\u2022\u25cf\u25aa\u25fe\ufffd]', '-', clean_text)
            clean_text = re.sub(r' {2,}', ' ', clean_text)
            return clean_text.strip()
        else:
            return f"(Ollama explanation unavailable: HTTP {response.status_code})"
    except Exception as e:
        return f"(Ollama explanation unavailable: {e})"
