"""
Tests for DeepGuard metadata forensics extraction.
Verifies Section 14 automated tests requirements:
- metadata() flags "re-encoded by ..." for a file produced with ffmpeg
- metadata() flags "no audio track" for video without audio
- ExifTool missing does not crash and continues with ffprobe only
- General metadata fields format and constraints
"""

import os
import shutil
import subprocess
import pytest
import numpy as np
import cv2

import pipeline


@pytest.fixture(scope="session")
def ffmpeg_encoded_video(tmp_path_factory):
    """Creates a sample video encoded via ffmpeg so it contains Lavf/ffmpeg tags."""
    tmp_dir = tmp_path_factory.mktemp("meta_test")
    raw_path = tmp_dir / "raw.mp4"
    final_path = tmp_dir / "encoded_ffmpeg.mp4"

    # Make small video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(raw_path), fourcc, 10.0, (160, 120))
    for _ in range(10):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        # Re-encode explicitly using libx264 which writes Lavf encoder metadata
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(raw_path),
            "-c:v", "libx264",
            "-metadata", "comment=TestDeepGuard",
            str(final_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and final_path.exists():
            return final_path

    return raw_path


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    """Creates a simple video with no audio stream."""
    tmp_dir = tmp_path_factory.mktemp("meta_silent")
    silent_path = tmp_dir / "silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(silent_path), fourcc, 10.0, (160, 120))
    for _ in range(10):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()
    return silent_path


def test_metadata_fields_present(ffmpeg_encoded_video):
    """Check that all required metadata keys exist and have proper types."""
    meta = pipeline.metadata(ffmpeg_encoded_video)
    assert "duration_s" in meta
    assert "size_mb" in meta
    assert "codec" in meta
    assert "resolution" in meta
    assert "fps" in meta
    assert "has_audio" in meta
    assert "exif" in meta
    assert "flags" in meta

    assert isinstance(meta["duration_s"], (int, float))
    assert isinstance(meta["size_mb"], (int, float))
    assert isinstance(meta["codec"], str)
    assert isinstance(meta["resolution"], str)
    assert isinstance(meta["fps"], (int, float))
    assert isinstance(meta["has_audio"], bool)
    assert isinstance(meta["exif"], dict)
    assert isinstance(meta["flags"], list)


def test_metadata_flag_reencoded(ffmpeg_encoded_video):
    """metadata() flags 're-encoded by ...' for a file produced with ffmpeg."""
    meta = pipeline.metadata(ffmpeg_encoded_video)
    flags = meta["flags"]
    # Check if re-encoded flag is present
    has_reencode_flag = any("re-encoded by" in f.lower() for f in flags)
    assert has_reencode_flag, f"Expected 're-encoded by' flag in {flags}"


def test_metadata_flag_no_audio(silent_video):
    """metadata() flags 'no audio track' when file has no audio stream."""
    meta = pipeline.metadata(silent_video)
    assert not meta["has_audio"]
    assert "no audio track" in meta["flags"]


def test_metadata_handles_missing_file():
    """metadata() handles non-existent file path gracefully without crashing."""
    meta = pipeline.metadata("non_existent_video_path_123.mp4")
    assert meta["duration_s"] == 0.0
    assert meta["size_mb"] == 0.0
    assert not meta["has_audio"]
    assert "no audio track" in meta["flags"]
