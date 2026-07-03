from __future__ import annotations
"""
vektor.persistence.atomic
-------------------------
Atomic file write utility using the write-to-temp, fsync, replace pattern.

os.replace() is used over os.rename() because it is atomic on both
POSIX and Windows. os.rename() is only atomic on POSIX.
"""


import os
import tempfile
from pathlib import Path
from contextlib import contextmanager


class AtomicWriteError(Exception):
    """Raised when an atomic write operation fails."""


@contextmanager
def atomic_write(target_path: Path, mode: str = "wb"):
    """
    Context manager for atomic file writes.

    Writes to a temporary file in the same directory as target_path,
    then atomically replaces target_path on successful exit.
    If an exception occurs, the temp file is deleted and target_path
    is left completely untouched.

    Usage:
        with atomic_write(Path("data/vector.bin")) as f:
            f.write(data)

    Args:
        target_path: The final destination path.
        mode:        File open mode. Default "wb" (binary write).

    Raises:
        AtomicWriteError: If the rename step fails after a successful write.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=target_path.name + ".tmp_",
    )

    try:
        with os.fdopen(tmp_fd, mode) as tmp_file:
            yield tmp_file
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_path, target_path)

    except Exception:
        # Clean up temp file if anything went wrong
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise