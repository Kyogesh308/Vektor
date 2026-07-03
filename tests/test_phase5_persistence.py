"""
Phase 5 — Persistence Layer Tests

Critical tests:
- test_vector_survives_restart: the only test that validates the phase's purpose
- test_crash_mid_write_leaves_file_intact: validates the atomic write pattern
- test_integrity_check_detects_mismatch: validates the startup guard
"""

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from vektor.persistence.atomic import atomic_write, AtomicWriteError
from vektor.persistence.binary import (
    init_vector_bin, append_vector, read_vector,
    read_vector_bin_header, append_offset, read_offset,
    write_stub_graph_node, read_graph_node,
    HEADER_SIZE,
)
from vektor.persistence.db import (
    initialize_db, insert_collection, get_collection,
    insert_vector_record, get_all_live_vector_records,
    tombstone_vector, start_run, end_run, write_log, record_fix,
)
from vektor.persistence.manifest import (
    write_manifest, read_manifest, validate_manifest_against_db, ManifestError
)
from vektor.persistence.archive import export_collection, import_collection, ArchiveError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def db_conn(tmp_dir):
    return initialize_db(tmp_dir / "metadata.db")


@pytest.fixture
def vector_bin(tmp_dir):
    path = tmp_dir / "vector.bin"
    init_vector_bin(path, dimension=4)
    return path


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_happy_path_creates_file(self, tmp_dir):
        target = tmp_dir / "test.bin"
        with atomic_write(target) as f:
            f.write(b"hello")
        assert target.read_bytes() == b"hello"

    def test_no_temp_file_remains_on_success(self, tmp_dir):
        target = tmp_dir / "test.bin"
        with atomic_write(target) as f:
            f.write(b"data")
        tmp_files = list(tmp_dir.glob("*.tmp_*"))
        assert len(tmp_files) == 0

    def test_exception_leaves_target_untouched(self, tmp_dir):
        target = tmp_dir / "test.bin"
        target.write_bytes(b"original")
        with pytest.raises(RuntimeError):
            with atomic_write(target) as f:
                f.write(b"partial")
                raise RuntimeError("simulated crash")
        assert target.read_bytes() == b"original"

    def test_no_temp_file_remains_on_failure(self, tmp_dir):
        target = tmp_dir / "test.bin"
        with pytest.raises(RuntimeError):
            with atomic_write(target) as f:
                raise RuntimeError("crash")
        tmp_files = list(tmp_dir.glob("*.tmp_*"))
        assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# vector.bin
# ---------------------------------------------------------------------------

class TestVectorBin:

    def test_header_written_on_init(self, vector_bin):
        header = read_vector_bin_header(vector_bin)
        assert header["dimension"] == 4
        assert header["slot_count"] == 0

    def test_append_and_read_roundtrip(self, vector_bin):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        append_vector(vector_bin, slot_id=0, vector=v)
        result = read_vector(vector_bin, slot_index=0, dimension=4)
        np.testing.assert_array_almost_equal(result, v, decimal=5)

    def test_slot_count_increments(self, vector_bin):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        append_vector(vector_bin, slot_id=0, vector=v)
        append_vector(vector_bin, slot_id=1, vector=v)
        header = read_vector_bin_header(vector_bin)
        assert header["slot_count"] == 2

    def test_multiple_vectors_readable_independently(self, vector_bin):
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        append_vector(vector_bin, 0, v1)
        append_vector(vector_bin, 1, v2)
        r1 = read_vector(vector_bin, 0, 4)
        r2 = read_vector(vector_bin, 1, 4)
        np.testing.assert_array_almost_equal(r1, v1, decimal=5)
        np.testing.assert_array_almost_equal(r2, v2, decimal=5)

    def test_crash_mid_append_leaves_file_intact(self, vector_bin):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        append_vector(vector_bin, 0, v)
        original = vector_bin.read_bytes()

        with pytest.raises(RuntimeError):
            from unittest.mock import patch
            with patch("vektor.persistence.binary.atomic_write") as mock_aw:
                mock_aw.side_effect = RuntimeError("disk full")
                append_vector(vector_bin, 1, v)

        assert vector_bin.read_bytes() == original


# ---------------------------------------------------------------------------
# offsets.bin
# ---------------------------------------------------------------------------

class TestOffsetsBin:

    def test_append_and_read_roundtrip(self, tmp_dir):
        path = tmp_dir / "offsets.bin"
        append_offset(path, node_id=0, byte_offset=1024)
        node_id, byte_offset = read_offset(path, index=0)
        assert node_id == 0
        assert byte_offset == 1024

    def test_multiple_offsets_readable(self, tmp_dir):
        path = tmp_dir / "offsets.bin"
        for i in range(5):
            append_offset(path, node_id=i, byte_offset=i * 100)
        for i in range(5):
            node_id, byte_offset = read_offset(path, index=i)
            assert node_id == i
            assert byte_offset == i * 100


# ---------------------------------------------------------------------------
# graph.bin
# ---------------------------------------------------------------------------

class TestGraphBin:

    def test_stub_node_write_and_read(self, tmp_dir):
        path = tmp_dir / "graph.bin"
        byte_offset = write_stub_graph_node(path)
        node = read_graph_node(path, byte_offset)
        assert node["num_layers"] == 0
        assert node["layers"] == []


# ---------------------------------------------------------------------------
# SQLite — db.py
# ---------------------------------------------------------------------------

class TestDatabase:

    def test_initialize_creates_all_tables(self, db_conn):
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert {"collections", "vectors", "runs", "logs", "fixes"}.issubset(table_names)

    def test_wal_mode_enabled(self, db_conn):
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_insert_and_get_collection(self, db_conn):
        insert_collection(db_conn, "test", 128, "cosine", 16, 200, "0.2.0")
        row = get_collection(db_conn, "test")
        assert row["dimension"] == 128
        assert row["metric"] == "cosine"

    def test_insert_vector_record(self, db_conn):
        insert_collection(db_conn, "test", 4, "cosine", 16, 200, "0.2.0")
        insert_vector_record(db_conn, "v1", "test", slot_id=0)
        records = get_all_live_vector_records(db_conn, "test")
        assert len(records) == 1
        assert records[0]["id"] == "v1"

    def test_tombstone_excludes_from_live(self, db_conn):
        insert_collection(db_conn, "test", 4, "cosine", 16, 200, "0.2.0")
        insert_vector_record(db_conn, "v1", "test", slot_id=0)
        tombstone_vector(db_conn, "v1", "test")
        records = get_all_live_vector_records(db_conn, "test")
        assert len(records) == 0

    def test_runs_and_logs(self, db_conn):
        run_id = start_run(db_conn, "test", "insert")
        write_log(db_conn, "test message", "INFO", run_id)
        end_run(db_conn, run_id, "completed")
        run = db_conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        assert run["status"] == "completed"
        log = db_conn.execute("SELECT * FROM logs WHERE run_id=?", (run_id,)).fetchone()
        assert log["message"] == "test message"

    def test_fixes_table(self, db_conn):
        run_id = start_run(db_conn, None, "debug")
        record_fix(db_conn, run_id, "off-by-one in slot count", "incremented after write")
        fix = db_conn.execute("SELECT * FROM fixes WHERE run_id=?", (run_id,)).fetchone()
        assert "off-by-one" in fix["problem"]


# ---------------------------------------------------------------------------
# manifest.py
# ---------------------------------------------------------------------------

class TestManifest:

    def test_write_and_read_roundtrip(self, tmp_dir):
        path = tmp_dir / "collection.json"
        write_manifest(path, "docs", 512, "cosine", 16, 200)
        data = read_manifest(path)
        assert data["name"] == "docs"
        assert data["dimension"] == 512

    def test_missing_manifest_raises(self, tmp_dir):
        with pytest.raises(ManifestError):
            read_manifest(tmp_dir / "nonexistent.json")

    def test_validate_against_db_passes(self, db_conn, tmp_dir):
        insert_collection(db_conn, "docs", 512, "cosine", 16, 200, "0.2.0")
        manifest = {"dimension": 512, "metric": "cosine", "m": 16, "ef_construction": 200}
        row = get_collection(db_conn, "docs")
        validate_manifest_against_db(manifest, row)  # must not raise

    def test_validate_dimension_mismatch_raises(self, db_conn, tmp_dir):
        insert_collection(db_conn, "docs", 512, "cosine", 16, 200, "0.2.0")
        manifest = {"dimension": 999, "metric": "cosine", "m": 16, "ef_construction": 200}
        row = get_collection(db_conn, "docs")
        with pytest.raises(ManifestError):
            validate_manifest_against_db(manifest, row)


# ---------------------------------------------------------------------------
# archive.py
# ---------------------------------------------------------------------------

class TestArchive:

    def _make_collection_dir(self, base: Path, db_conn) -> Path:
        """Create a minimal valid collection directory for archive tests."""
        col_dir = base / "docs"
        col_dir.mkdir()

        insert_collection(db_conn, "docs", 4, "cosine", 16, 200, "0.2.0")
        write_manifest(col_dir / "collection.json", "docs", 4, "cosine", 16, 200)

        v_path = col_dir / "vector.bin"
        init_vector_bin(v_path, dimension=4)
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        append_vector(v_path, 0, v)

        o_path = col_dir / "offsets.bin"
        byte_offset = write_stub_graph_node(col_dir / "graph.bin")
        append_offset(o_path, 0, byte_offset)

        import shutil
        shutil.copy(base / "metadata.db", col_dir / "metadata.db")
        return col_dir

    def test_export_creates_vdb(self, tmp_dir, db_conn):
        (tmp_dir / "metadata.db")  # already created by db_conn
        col_dir = self._make_collection_dir(tmp_dir, db_conn)
        out = tmp_dir / "docs.vdb"
        export_collection(col_dir, out)
        assert out.exists()
        import zipfile
        assert zipfile.is_zipfile(out)

    def test_export_missing_files_raises(self, tmp_dir):
        col_dir = tmp_dir / "empty"
        col_dir.mkdir()
        with pytest.raises(ArchiveError):
            export_collection(col_dir, tmp_dir / "out.vdb")


# ---------------------------------------------------------------------------
# Restart simulation — the only test that validates Phase 5's purpose
# ---------------------------------------------------------------------------

class TestVectorSurvivesRestart:

    def test_vector_readable_after_process_restart(self, tmp_dir):
        """
        Simulate process restart by:
        1. Writing vectors to disk
        2. Closing all handles (simulating process exit)
        3. Re-reading from disk cold (simulating process start)
        4. Verifying data integrity
        """
        db_path = tmp_dir / "metadata.db"
        v_path = tmp_dir / "vector.bin"

        # --- "First process" ---
        conn = initialize_db(db_path)
        insert_collection(conn, "docs", 4, "cosine", 16, 200, "0.2.0")
        init_vector_bin(v_path, dimension=4)

        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        append_vector(v_path, slot_id=0, vector=original)
        insert_vector_record(conn, "vec_001", "docs", slot_id=0)
        conn.close()
        # All handles closed — simulates process exit

        # --- "Second process" ---
        conn2 = initialize_db(db_path)
        records = get_all_live_vector_records(conn2, "docs")
        assert len(records) == 1

        slot_id = records[0]["slot_id"]
        recovered = read_vector(v_path, slot_index=slot_id, dimension=4)
        np.testing.assert_allclose(recovered, original, rtol=1e-5)
        conn2.close()

    def test_tombstoned_vector_not_returned_after_restart(self, tmp_dir):
        db_path = tmp_dir / "metadata.db"

        conn = initialize_db(db_path)
        insert_collection(conn, "docs", 4, "cosine", 16, 200, "0.2.0")
        insert_vector_record(conn, "vec_001", "docs", slot_id=0)
        tombstone_vector(conn, "vec_001", "docs")
        conn.close()

        conn2 = initialize_db(db_path)
        records = get_all_live_vector_records(conn2, "docs")
        assert len(records) == 0
        conn2.close()