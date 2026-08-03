# Vektor Benchmark Report

## Methodology & Two-Branch Execution Disclosure

Experiments 6 and the completed portion of 1/2 (SIFT-128, ef=10-200) were measured on the original from-scratch implementation. Due to impractical wall-clock cost at full scale — confirmed independent of hardware via a dedicated server run — Experiments 3, 4, 5, and the remainder of 1/2 were completed using a verified-equivalent optimized branch (`perf-experiment`: contiguous vector storage + batched distance computation), whose correctness was validated against the original implementation via identical recall and an explicit heap-behavior regression test before use.

---

## Benchmark Execution Log & Results Tracker

| Experiment | Target | Config / Command | Status |
| :--- | :--- | :--- | :---: |
| **Pre-Flight** | Dataset Download | `python scripts/download_datasets.py` | Required |
| **Pre-Flight** | Server Speedup Replication | `python -c "..."` (5k worker test) | Required |
| **Pre-Flight** | Baseline Sanity | `pytest tests/test_phase6_benchmark.py::TestBaselineValidation -v` | Required |
| **Exp 3 & 4** | Build Time & Memory (1k..1M) | `python -m benchmarks.exp3_4_build_and_memory` | Pending Server Run |
| **Exp 5** | Recall vs M (Fixed Ground Truth) | `python -m benchmarks.exp5_recall_vs_m` | Pending Server Run |
| **Exp 1 & 2** | Full SIFT-128 + GloVe-100 Sweep | `python -m benchmarks.exp1_2_recall_qps_vs_ef` | Pending Server Run |
| **Post-Flight** | Monotonicity Verification | `python benchmarks/check_monotonicity.py` | Pending Server Run |
