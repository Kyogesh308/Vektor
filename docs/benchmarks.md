# Vektor Benchmark Results

## Phase 2 — NumPy vs Pure Python Distance Metrics

**Hardware:** [i5-12450H (2.00 GHz), 16.0 GB]  
**Python:** 3.12.9  
**NumPy:** [2.5.0]  
**Dimension:** 1536 (OpenAI text-embedding-ada-002)  
**Trials:** 10,000 per metric  

Vektor Phase 2 Benchmark — NumPy vs Pure Python
Dimension: 1536 | Trials: 10,000

──────────────────────────────────────────────────
  Metric:      Cosine Similarity
  Dimension:   1536
  Trials:      10,000
  NumPy time:  78.01 ms total  |  7.80 µs per call
  Python time: 3617.98 ms total  |  361.80 µs per call
  Speedup:     46.4x  ← NumPy is faster

──────────────────────────────────────────────────
  Metric:      Dot Product
  Dimension:   1536
  Trials:      10,000
  NumPy time:  8.70 ms total  |  0.87 µs per call
  Python time: 775.91 ms total  |  77.59 µs per call
  Speedup:     89.2x  ← NumPy is faster

──────────────────────────────────────────────────
  Metric:      L2 Distance
  Dimension:   1536
  Trials:      10,000
  NumPy time:  32.62 ms total  |  3.26 µs per call
  Python time: 1792.48 ms total  |  179.25 µs per call
  Speedup:     55.0x  ← NumPy is faster

──────────────────────────────────────────────────

_Recorded: [3 July, 2026]_