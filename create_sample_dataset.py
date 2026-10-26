"""
Enhanced Dataset Generator for FakeAVCeleb and Cross-Dataset Validation.
Generates video clips with realistic synthetic face movements and audio tracks.
Includes import config to ensure FFmpeg and ExifTool are in PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

import config


def render_face_frame(
    frame_idx: int,
    total_frames: int = 16,
    is_fake: bool = False,
    base_color=(210, 180, 160),
    w: int = 240,
    h: int = 240,
) -> np.ndarray:
    """Renders a frame with a face-like structure, eyes, and mouth with temporal movement."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Background
    bg_val = 40 + (frame_idx * 2) % 30
    frame[:, :] = (bg_val, bg_val + 10, bg_val + 20)

    # Face center with subtle natural sway
    center_x = w // 2 + int(np.sin(frame_idx * 0.4) * 4)
    center_y = h // 2 + int(np.cos(frame_idx * 0.3) * 3)

    if is_fake and frame_idx % 4 == 0:
        # Manipulation artifact: slight boundary glitch / chromatic shift
        center_x += np.random.randint(-5, 6)
        face_col = (base_color[0] + 30, base_color[1] - 20, base_color[2] + 40)
    else:
        face_col = base_color

    # Draw head / face oval
    cv2.ellipse(frame, (center_x, center_y), (50, 68), 0, 0, 360, face_col, -1)

    # Hair
    cv2.ellipse(frame, (center_x, center_y - 25), (52, 45), 0, 180, 360, (30, 25, 20), -1)

    # Eyes
    eye_y = center_y - 10
    left_eye_x = center_x - 18
    right_eye_x = center_x + 18

    # Blinking logic
    blink = (frame_idx in (6, 7)) or (is_fake and frame_idx in (2, 9))
    if blink:
        cv2.line(frame, (left_eye_x - 8, eye_y), (left_eye_x + 8, eye_y), (30, 30, 30), 2)
        cv2.line(frame, (right_eye_x - 8, eye_y), (right_eye_x + 8, eye_y), (30, 30, 30), 2)
    else:
        cv2.circle(frame, (left_eye_x, eye_y), 6, (250, 250, 250), -1)
        cv2.circle(frame, (left_eye_x, eye_y), 3, (40, 40, 80), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 6, (250, 250, 250), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 3, (40, 40, 80), -1)

    # Nose
    cv2.line(frame, (center_x, center_y - 4), (center_x, center_y + 8), (170, 140, 120), 2)

    # Mouth with speech-like opening
    mouth_y = center_y + 24
    open_amt = max(2, int(abs(np.sin(frame_idx * 0.8)) * 8))
    if is_fake and frame_idx % 3 == 0:
        # Desynced or distorted mouth opening
        open_amt = (open_amt + 6) % 12
    cv2.ellipse(frame, (center_x, mouth_y), (14, open_amt), 0, 0, 360, (140, 60, 60), -1)

    return frame


def generate_clip(dest_path: Path, freq: int = 440, is_fake: bool = False) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_raw = dest_path.with_name(f"temp_{dest_path.name}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_raw), fourcc, 15.0, (240, 240))
    for i in range(16):
        frame = render_face_frame(i, is_fake=is_fake)
        out.write(frame)
    out.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        audio_src = f"sine=frequency={freq}:duration=1.2"
        if is_fake:
            # Add subtle harmonic distortion or higher pitch
            audio_src = f"sine=frequency={freq + 200}:duration=1.2"

        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(temp_raw),
            "-f", "lavfi", "-i", audio_src,
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest", str(dest_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_raw.exists():
            temp_raw.unlink()
        if res.returncode == 0 and dest_path.exists():
            return

    if temp_raw.exists():
        temp_raw.rename(dest_path)


def main():
    root = Path("FakeAVCeleb_samples")
    if root.exists():
        shutil.rmtree(root)

    samples = [
        # Real videos (8 samples, 8 distinct identities)
        ("RealVideo-RealAudio", "id00001", "v1.mp4", 320, False),
        ("RealVideo-RealAudio", "id00002", "v2.mp4", 360, False),
        ("RealVideo-RealAudio", "id00003", "v3.mp4", 400, False),
        ("RealVideo-RealAudio", "id00004", "v4.mp4", 440, False),
        ("RealVideo-RealAudio", "id00005", "v5.mp4", 480, False),
        ("RealVideo-RealAudio", "id00006", "v6.mp4", 520, False),
        ("RealVideo-RealAudio", "id00007", "v7.mp4", 560, False),
        ("RealVideo-RealAudio", "id00008", "v8.mp4", 600, False),

        # Fake videos (8 samples, 8 distinct identities across various sub-folders)
        ("FakeVideo-FakeAudio", "id00009", "v9.mp4", 720, True),
        ("FakeVideo-RealAudio", "id00010", "v10.mp4", 350, True),
        ("RealVideo-FakeAudio", "id00011", "v11.mp4", 780, True),
        ("FakeVideo-FakeAudio", "id00012", "v12.mp4", 820, True),
        ("FakeVideo-RealAudio", "id00013", "v13.mp4", 410, True),
        ("RealVideo-FakeAudio", "id00014", "v14.mp4", 860, True),
        ("FakeVideo-FakeAudio", "id00015", "v15.mp4", 900, True),
        ("FakeVideo-FakeAudio", "id00016", "v16.mp4", 940, True),
    ]

    for cat, identity, fname, freq, is_fake in samples:
        dest = root / cat / identity / fname
        generate_clip(dest, freq=freq, is_fake=is_fake)
        print(f"Generated {dest}")

    # Cross dataset samples
    cross_root = Path("CelebDF_samples")
    if cross_root.exists():
        shutil.rmtree(cross_root)

    cross_samples = [
        ("RealVideo-RealAudio", "cdf_id01", "c1.mp4", 430, False),
        ("RealVideo-RealAudio", "cdf_id02", "c2.mp4", 450, False),
        ("RealVideo-RealAudio", "cdf_id03", "c3.mp4", 470, False),
        ("FakeVideo-FakeAudio", "cdf_id04", "c4.mp4", 750, True),
        ("FakeVideo-FakeAudio", "cdf_id05", "c5.mp4", 810, True),
        ("FakeVideo-FakeAudio", "cdf_id06", "c6.mp4", 870, True),
    ]
    for cat, identity, fname, freq, is_fake in cross_samples:
        dest = cross_root / cat / identity / fname
        generate_clip(dest, freq=freq, is_fake=is_fake)
        print(f"Generated cross sample {dest}")


if __name__ == "__main__":
    main()
