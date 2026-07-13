"""
Dataset downloader for Vektor benchmarks.

Downloads SIFT-128 and GloVe-100 from the ann-benchmarks data repository.
Run once before executing benchmarks:

    python scripts/download_datasets.py

Files are saved to benchmarks/datasets/ which is gitignored.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "benchmarks" / "datasets"

DATASETS = {
    "sift-128-euclidean.hdf5": {
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "expected_size_mb": 465,
    },
    "glove-100-angular.hdf5": {
        "url": "https://ann-benchmarks.com/glove-100-angular.hdf5",
        "expected_size_mb": 485,
    },
}


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:.1f}%  ({downloaded // 1_000_000}MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def verify_file(path: Path, expected_size_mb: int) -> None:
    size_mb = path.stat().st_size / 1_000_000
    if size_mb < expected_size_mb * 0.9:
        raise ValueError(
            f"{path.name}: expected ~{expected_size_mb}MB, "
            f"got {size_mb:.1f}MB — likely a partial download. Delete and retry."
        )
    print(f"  OK: {path.name} ({size_mb:.1f}MB)")


def main():
    print(f"Dataset directory: {DATASETS_DIR}\n")
    for filename, info in DATASETS.items():
        dest = DATASETS_DIR / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
            verify_file(dest, info["expected_size_mb"])
        else:
            download_file(info["url"], dest)
            verify_file(dest, info["expected_size_mb"])
    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()