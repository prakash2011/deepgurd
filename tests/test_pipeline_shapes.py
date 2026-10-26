"""
Tests for DeepGuard pipeline tensor shapes and model architecture.
Verifies Section 14 automated tests requirements:
- features() returns shape (16, 2560) for a short sample video
- video_frames() and audio_frames() each return (16, 3, 224, 224)
- Video without audio does not raise and yields zero audio tensor
- Model forward pass on random tensor (2, 16, 2560) returns shape (2,)
"""

import os
import tempfile
import subprocess
import shutil
import pytest
import numpy as np
import cv2
import torch

import config
import pipeline


@pytest.fixture(scope="session")
def sample_video_with_audio(tmp_path_factory):
    """Creates a 2-second test MP4 video containing video frames and a generated audio tone."""
    tmp_dir = tmp_path_factory.mktemp("video_test")
    raw_video = tmp_dir / "raw_video.mp4"
    final_video = tmp_dir / "video_with_audio.mp4"

    # 1. Write frames via OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15.0
    w, h = 320, 240
    out = cv2.VideoWriter(str(raw_video), fourcc, fps, (w, h))
    for i in range(30):
        # Draw a synthetic colored image with a square in the center
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 8 % 255, 120, 200)
        # Add a face-like oval
        cv2.circle(frame, (160, 120), 40, (255, 255, 255), -1)
        out.write(frame)
    out.release()

    # 2. Add audio stream via ffmpeg if available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(raw_video),
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_video),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and final_video.exists():
            return final_video

    return raw_video


@pytest.fixture(scope="session")
def sample_video_no_audio(tmp_path_factory):
    """Creates a 1-second video without any audio stream."""
    tmp_dir = tmp_path_factory.mktemp("no_audio_test")
    video_path = tmp_dir / "silent_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
    for _ in range(15):
        frame = np.ones((120, 160, 3), dtype=np.uint8) * 100
        out.write(frame)
    out.release()
    return video_path


def test_video_frames_shape(sample_video_with_audio):
    """video_frames() returns ndarray of shape (16, 3, 224, 224)."""
    frames = pipeline.video_frames(sample_video_with_audio, t_steps=16)
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (16, 3, 224, 224)
    assert frames.dtype == np.float32


def test_audio_frames_shape(sample_video_with_audio):
    """audio_frames() returns ndarray of shape (16, 3, 224, 224)."""
    a_frames = pipeline.audio_frames(sample_video_with_audio, t_steps=16)
    assert isinstance(a_frames, np.ndarray)
    assert a_frames.shape == (16, 3, 224, 224)
    assert a_frames.dtype == np.float32


def test_video_without_audio(sample_video_no_audio):
    """Video without audio does not raise and yields zero audio tensor."""
    a_frames = pipeline.audio_frames(sample_video_no_audio, t_steps=16)
    assert isinstance(a_frames, np.ndarray)
    assert a_frames.shape == (16, 3, 224, 224)
    # Must be all zeros
    assert np.all(a_frames == 0.0)


def test_features_shape(sample_video_with_audio):
    """features() returns Tensor of shape (16, 2560)."""
    feats = pipeline.features(sample_video_with_audio)
    assert isinstance(feats, torch.Tensor)
    assert feats.shape == (16, 2560)


def test_model_forward_pass_shape():
    """Model forward pass on random tensor (2, 16, 2560) returns shape (2,)."""
    model = pipeline.Detector(input_dim=2560, hidden_dim=config.HIDDEN)
    model.eval()
    dummy_input = torch.randn(2, 16, 2560)
    with torch.no_grad():
        out = model(dummy_input)
        prob = model.predict_proba(dummy_input)

    assert out.shape == (2,)
    assert prob.shape == (2,)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
