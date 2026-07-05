"""
vektor.benchmark.datasets
--------------------------
Dataset loaders for benchmark runs.

Each loader returns a DatasetBundle — a typed container holding:
- train:       float32 ndarray of shape (n_train, dim)
- queries:     float32 ndarray of shape (n_queries, dim)
- ground_truth: int32 ndarray of shape (n_queries, k)
  Each row contains the indices of the k true nearest neighbours
  for that query, sorted nearest-first.
- metric:      The distance metric this dataset uses.
- name:        Human-readable dataset name for result logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks" / "datasets"


@dataclass
class DatasetBundle:
    name: str
    train: np.ndarray        # float32, shape (n_train, dim)
    queries: np.ndarray      # float32, shape (n_queries, dim)
    ground_truth: np.ndarray # int32, shape (n_queries, k)
    metric: str


# ---------------------------------------------------------------------------
# HDF5 datasets (SIFT-128, GloVe-100)
# ---------------------------------------------------------------------------

def load_hdf5_dataset(
    filename: str,
    metric: str,
    name: str,
    max_train: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> DatasetBundle:
    """
    Load a standard ann-benchmarks HDF5 dataset.

    Args:
        filename:    Filename inside benchmarks/datasets/.
        metric:      Distance metric ("euclidean" or "cosine").
        name:        Human-readable name for logging.
        max_train:   If set, load only the first max_train training vectors.
        max_queries: If set, load only the first max_queries query vectors.

    Returns:
        DatasetBundle with float32 arrays and int32 ground truth.

    Raises:
        FileNotFoundError: Dataset file not found. Run download_datasets.py.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Run: pip install h5py>=3.9")

    path = DATASETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python scripts/download_datasets.py"
        )

    with h5py.File(path, "r") as f:
        train = f["train"][:max_train].astype(np.float32)
        queries = f["test"][:max_queries].astype(np.float32)
        ground_truth = f["neighbors"][:max_queries].astype(np.int32)

    return DatasetBundle(
        name=name,
        train=train,
        queries=queries,
        ground_truth=ground_truth,
        metric=metric,
    )


def load_sift128(max_train: Optional[int] = None,
                 max_queries: Optional[int] = None) -> DatasetBundle:
    """Load SIFT-128 (L2/euclidean). Full size: 1M train, 10K queries."""
    return load_hdf5_dataset(
        filename="sift-128-euclidean.hdf5",
        metric="euclidean",
        name="SIFT-128",
        max_train=max_train,
        max_queries=max_queries,
    )


def load_glove100(max_train: Optional[int] = None,
                  max_queries: Optional[int] = None) -> DatasetBundle:
    """Load GloVe-100 (cosine/angular). Full size: 1.18M train, 10K queries."""
    return load_hdf5_dataset(
        filename="glove-100-angular.hdf5",
        metric="cosine",
        name="GloVe-100",
        max_train=max_train,
        max_queries=max_queries,
    )


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def _brute_force_ground_truth(
    train: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    """
    Compute exact ground truth using matrix operations.

    [Certain] Do NOT use the Phase 4 VectorStore brute-force loop here.
    That would be circular — a bug in Phase 4 search would make HNSW
    recall look correct when it isn't.
    This uses raw NumPy matrix math for independence.
    """
    n_queries = queries.shape[0]
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)

    for i, query in enumerate(queries):
        if metric == "euclidean":
            diffs = train - query
            dists = np.sum(diffs ** 2, axis=1)
            indices = np.argsort(dists)[:k]
        elif metric == "cosine":
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            train_norms = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-10)
            sims = train_norms @ query_norm
            indices = np.argsort(-sims)[:k]
        elif metric == "dot":
            dots = train @ query
            indices = np.argsort(-dots)[:k]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        ground_truth[i] = indices

    return ground_truth


def load_synthetic(
    n_train: int = 10_000,
    n_queries: int = 100,
    dim: int = 128,
    k: int = 10,
    metric: str = "euclidean",
    seed: int = 42,
) -> DatasetBundle:
    """
    Generate a synthetic dataset with exact brute-force ground truth.

    Args:
        n_train:   Number of training vectors.
        n_queries: Number of query vectors.
        dim:       Vector dimension.
        k:         Number of nearest neighbours in ground truth.
        metric:    Distance metric for ground truth computation.
        seed:      Random seed for reproducibility.

    Returns:
        DatasetBundle with NumPy-computed ground truth.
    """
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n_train, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    ground_truth = _brute_force_ground_truth(train, queries, k, metric)

    return DatasetBundle(
        name=f"synthetic-{dim}d-{n_train}",
        train=train,
        queries=queries,
        ground_truth=ground_truth,
        metric=metric,
    )