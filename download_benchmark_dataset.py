"""
Downloads authentic deepfake benchmark videos from Hugging Face and structures them
into the FakeAVCeleb convention:
- RealVideo-RealAudio/<subject_id>/<filename>.mp4
- FakeVideo-FakeAudio/<subject_id>/<filename>.mp4
"""

import os
import re
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm


def download_benchmark_dataset(
    dest_dir: str = "FakeAVCeleb_real",
    num_real: int = 15,
    num_fake: int = 15,
):
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_id = "angads24/deepfake-video"
    print(f"Fetching file list from {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    real_files = [f for f in files if "dataset/real/" in f and f.endswith(".mp4")][:num_real]
    fake_files = [f for f in files if "dataset/fake/" in f and f.endswith(".mp4")][:num_fake]

    print(f"Downloading {len(real_files)} real videos and {len(fake_files)} fake videos...")

    # Download Real Videos
    for r_file in tqdm(real_files, desc="Downloading Real Videos"):
        filename = Path(r_file).name
        # Extract subject ID (e.g. 01 from 01__kitchen.mp4)
        m = re.match(r"^(\d+)", filename)
        sub_id = f"id{m.group(1).zfill(5)}" if m else "id00001"
        target_path = dest_root / "RealVideo-RealAudio" / sub_id / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            downloaded = hf_hub_download(repo_id=repo_id, filename=r_file, repo_type="dataset")
            # Copy to target path
            import shutil
            shutil.copy2(downloaded, target_path)

    # Download Fake Videos
    for f_file in tqdm(fake_files, desc="Downloading Fake Videos"):
        filename = Path(f_file).name
        # Extract subject ID (e.g. 01_02 -> id01002)
        m = re.match(r"^(\d+)_(\d+)", filename)
        if m:
            sub_id = f"id{m.group(1).zfill(3)}{m.group(2).zfill(2)}"
        else:
            sub_id = "id00999"
        target_path = dest_root / "FakeVideo-FakeAudio" / sub_id / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            downloaded = hf_hub_download(repo_id=repo_id, filename=f_file, repo_type="dataset")
            import shutil
            shutil.copy2(downloaded, target_path)

    print(f"Successfully downloaded and organized real benchmark videos into {dest_root}!")


if __name__ == "__main__":
    download_benchmark_dataset(dest_dir="FakeAVCeleb_real", num_real=15, num_fake=15)
