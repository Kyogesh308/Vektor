# Vektor Release Guide

This guide details the final steps required to release the `vektor-db` package to PyPI, once all development and local packaging tasks are complete. The `dist/` directory already contains the built `.whl` and `.tar.gz` files from Phase 14.

## Prerequisites
Ensure that you have an account on both [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/), and that you have configured your API tokens (typically stored in `~/.pypirc`).

## Step 1: Upload to TestPyPI
It is best practice to test the release on TestPyPI first to ensure that everything renders correctly and dependencies resolve.

```bash
twine upload --repository testpypi dist/*
```

## Step 2: Verification from TestPyPI
Create a fresh virtual environment and verify that the package installs cleanly from TestPyPI. Because TestPyPI may not have all dependencies, we supply the main PyPI index as a fallback.

```bash
# Windows
python -m venv testpypi_test
testpypi_test\Scripts\activate

# Install from TestPyPI with PyPI fallback
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vektor-db

# Test that the library functions correctly
python -c "
from vektor import Vektor
client = Vektor()
collection = client.create_collection('documents', dim=4, metric='cosine')
collection.insert('doc1', [0.1, 0.2, 0.3, 0.4])
results = collection.search([0.1, 0.2, 0.3, 0.4], k=1)
print(results[0].id)
"

deactivate
```

## Step 3: Real PyPI Upload
Once TestPyPI verification is successful, upload the built package to the real PyPI index.
**Warning**: This action is irreversible for version `1.0.0`. You cannot upload the same version number twice, even if you delete the release on PyPI.

```bash
twine upload dist/*
```

## Step 4: Final Verification on PyPI
Test the official installation the same way a new user would.

```bash
# Windows
python -m venv final_test
final_test\Scripts\activate

# Install from PyPI
pip install vektor-db

# Quickstart test
python -c "
from vektor import Vektor
client = Vektor()
collection = client.create_collection('documents', dim=4, metric='cosine')
collection.insert('doc1', [0.1, 0.2, 0.3, 0.4], {'source': 'example'})
collection.insert('doc2', [0.5, 0.1, 0.2, 0.9], {'source': 'example'})
results = collection.search([0.1, 0.2, 0.3, 0.4], k=2)
for r in results:
    print(r.id, r.score, r.metadata)
"

deactivate
```

If the quickstart completes successfully, the PyPI release is fully complete and functional!
