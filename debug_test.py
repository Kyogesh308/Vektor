from tests.test_api_integration import make_vecs
from vektor import Vektor

client = Vektor()

col = client.create_collection("debuggg", dim=16, metric="cosine")

for i in range(10):
    col.insert(f"v{i}", make_vecs(1, 16, seed=i)[0])

print(col._index.size)
print(col.count())