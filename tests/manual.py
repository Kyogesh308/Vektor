# Run this manually to sanity-check before continuing
import sys
from pathlib import Path

# Adds the 'src' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vektor.hnsw.layer import layer_distribution_stats
stats = layer_distribution_stats(M=16, n_samples=10_000)
print(stats)
# Expected: layer 0 has ~9400 entries, layer 1 ~560, layer 2 ~35, etc.
# The ratio between consecutive layers should be approximately 1/M = 0.0625