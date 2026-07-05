"""
vektor.benchmark.recall
------------------------
Recall@k calculator.

[Certain] Use external ground truth from HDF5 files for real datasets.
Never compute ground truth using the system under test and measure recall
against its own output — that is circular and undetectable as wrong.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(returned_ids: list[int], true_ids: list[int], k: int) -> float:
    """
    Compute recall@k for a single query.

    Args:
        returned_ids: IDs returned by the search system (up to k).
        true_ids:     True k nearest neighbour IDs (ground truth).
        k:            The k in recall@k.

    Returns:
        float in [0.0, 1.0].
    """
    if k == 0:
        return 1.0
    returned_set = set(returned_ids[:k])
    true_set = set(true_ids[:k])
    return len(returned_set & true_set) / k


def mean_recall_at_k(
    all_returned: list[list[int]],
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute mean recall@k across all queries.

    Args:
        all_returned:  List of returned ID lists, one per query.
        ground_truth:  Int array of shape (n_queries, gt_k). Each row is
                       the sorted true nearest neighbour indices.
        k:             The k in recall@k. Must be <= gt_k.

    Returns:
        float: mean recall@k across all queries.
    """
    assert len(all_returned) == len(ground_truth), (
        f"Query count mismatch: {len(all_returned)} returned vs "
        f"{len(ground_truth)} ground truth rows."
    )
    recalls = [
        recall_at_k(returned, gt_row.tolist(), k)
        for returned, gt_row in zip(all_returned, ground_truth)
    ]
    return float(np.mean(recalls))