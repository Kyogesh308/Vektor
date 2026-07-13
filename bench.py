# bench.py
import numpy as np
from vektor.collection import Collection
from vektor.hnsw.index import HNSWIndex

rng = np.random.default_rng(42)
vectors = rng.standard_normal((5000,128)).astype(np.float32)

col = Collection(
    name="t",
    dimension=128,
    metric="euclidean",
    m=16,
    ef_construction=200,
)

idx = HNSWIndex(col, seed=42)

for i, v in enumerate(vectors):
    idx.add(i, v)