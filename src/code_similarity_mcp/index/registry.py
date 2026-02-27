"""SQLite-backed method registry + FAISS vector index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class MethodRegistry:
    """Persistent store for method metadata (SQLite) and embeddings (FAISS)."""

    def __init__(self, index_dir: str | Path) -> None:
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "methods.db"
        self._faiss_path = self._dir / "index.faiss"
        self._id_map_path = self._dir / "id_map.json"
        self._chunks_faiss_path = self._dir / "chunks.faiss"
        self._chunk_id_map_path = self._dir / "chunk_id_map.json"

        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()
        self._init_chunks_table()
        self._faiss_index, self._id_map = self._load_or_create_index()
        self._chunks_index, self._chunk_id_map = self._load_or_create_chunk_index()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS methods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                return_type TEXT,
                body_code TEXT NOT NULL,
                normalized_code TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                dependencies TEXT NOT NULL,
                faiss_pos INTEGER,
                ast_fingerprint TEXT NOT NULL DEFAULT '[]'
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_file ON methods(file_path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON methods(code_hash)")
        # Migrate existing databases that pre-date the ast_fingerprint column
        existing_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(methods)").fetchall()
        }
        if "ast_fingerprint" not in existing_cols:
            self._conn.execute(
                "ALTER TABLE methods ADD COLUMN ast_fingerprint TEXT NOT NULL DEFAULT '[]'"
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _load_or_create_index(self):
        if self._faiss_path.exists() and self._id_map_path.exists():
            index = faiss.read_index(str(self._faiss_path))
            id_map: dict[int, int] = {
                int(k): v for k, v in json.loads(self._id_map_path.read_text()).items()
            }
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            id_map: dict[int, int] = {}  # faiss_pos -> method db id
        return index, id_map

    def _save_index(self) -> None:
        faiss.write_index(self._faiss_index, str(self._faiss_path))
        self._id_map_path.write_text(json.dumps(self._id_map))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_method(self, method_info, embedding: np.ndarray) -> int:
        """Insert a method and its embedding. Returns the DB row id."""
        cur = self._conn.execute(
            """INSERT INTO methods
               (file_path, language, name, parameters, return_type, body_code,
                normalized_code, code_hash, start_line, end_line, dependencies,
                faiss_pos, ast_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                method_info.file_path,
                method_info.language,
                method_info.name,
                json.dumps(method_info.parameters),
                method_info.return_type,
                method_info.body_code,
                method_info.normalized_code,
                method_info.code_hash,
                method_info.start_line,
                method_info.end_line,
                json.dumps(method_info.dependencies),
                None,  # faiss_pos updated below
                json.dumps(getattr(method_info, "ast_fingerprint", [])),
            ),
        )
        db_id = cur.lastrowid

        # Add to FAISS
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss_pos = self._faiss_index.ntotal
        self._faiss_index.add(vec)
        self._id_map[faiss_pos] = db_id

        self._conn.execute("UPDATE methods SET faiss_pos=? WHERE id=?", (faiss_pos, db_id))
        self._conn.commit()
        self._save_index()
        return db_id

    def delete_by_id(self, db_id: int) -> bool:
        """Remove a single method by its DB id. Returns True if it existed."""
        cur = self._conn.execute(
            "SELECT faiss_pos FROM methods WHERE id=?", (db_id,)
        )
        row = cur.fetchone()
        if row is None:
            return False

        faiss_pos = row[0]
        self._conn.execute("DELETE FROM methods WHERE id=?", (db_id,))
        self._conn.commit()

        if faiss_pos is not None:
            self._id_map.pop(faiss_pos, None)
        self._save_index()
        return True

    def _delete_chunks_for_methods(self, method_ids: list[int]) -> int:
        """Delete all chunks for the given method IDs.

        Updates *_chunk_id_map* in memory but does **not** commit or save the
        chunk FAISS index — the caller is responsible for both.  Returns the
        number of chunk rows removed.
        """
        if not method_ids:
            return 0
        placeholders = ",".join("?" * len(method_ids))
        cur = self._conn.execute(
            f"SELECT faiss_pos FROM chunks WHERE function_id IN ({placeholders})",
            method_ids,
        )
        chunk_rows = cur.fetchall()
        if not chunk_rows:
            return 0

        self._conn.execute(
            f"DELETE FROM chunks WHERE function_id IN ({placeholders})",
            method_ids,
        )
        for (faiss_pos,) in chunk_rows:
            if faiss_pos is not None:
                self._chunk_id_map.pop(faiss_pos, None)
        return len(chunk_rows)

    def delete_by_file(self, file_path: str) -> int:
        """Remove all methods for a file and cascade-delete their chunks.

        Returns the number of method rows removed.
        """
        cur = self._conn.execute(
            "SELECT id, faiss_pos FROM methods WHERE file_path=?", (file_path,)
        )
        rows = cur.fetchall()
        if not rows:
            return 0

        method_ids = [row[0] for row in rows]

        # Cascade: remove chunks that belong to these methods before deleting
        # the methods themselves so that re-indexing a file never leaves orphaned
        # chunk records referencing function IDs that no longer exist.
        chunks_removed = self._delete_chunks_for_methods(method_ids)

        self._conn.execute("DELETE FROM methods WHERE file_path=?", (file_path,))
        self._conn.commit()

        # Update method FAISS id map
        positions = {row[1] for row in rows if row[1] is not None}
        for pos in positions:
            self._id_map.pop(pos, None)
        self._save_index()

        # Persist chunk FAISS id map only if anything was actually removed
        if chunks_removed > 0:
            self._save_chunk_index()

        return len(rows)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        allowed_ids: set[int] | None = None,
    ) -> list[dict]:
        """Return top-k similar methods by embedding similarity.

        If *allowed_ids* is provided, only methods whose DB id is in that set
        are returned.  Pass an empty set to short-circuit (returns []).
        """
        if self._faiss_index.ntotal == 0:
            return []
        if allowed_ids is not None and not allowed_ids:
            return []
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._faiss_index.ntotal)
        scores, positions = self._faiss_index.search(vec, k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos not in self._id_map:
                continue
            db_id = self._id_map[pos]
            if allowed_ids is not None and db_id not in allowed_ids:
                continue
            row = self._get_method_by_id(db_id)
            if row:
                row["embedding_score"] = float(score)
                results.append(row)
        return results

    def filter_by_criteria(
        self, language: str, param_count: int, loc: int
    ) -> set[int]:
        """Return DB IDs matching: same language, param count ±1, LOC ±30%.

        Language and LOC bounds are applied via SQL; param count is checked
        in Python because parameters are stored as a JSON array.
        """
        if loc > 0:
            loc_min = int(loc * 0.7)
            loc_max = int(loc / 0.7) + 1
            cur = self._conn.execute(
                """SELECT id, parameters FROM methods
                   WHERE language=?
                   AND (end_line - start_line + 1) BETWEEN ? AND ?""",
                (language, loc_min, loc_max),
            )
        else:
            cur = self._conn.execute(
                "SELECT id, parameters FROM methods WHERE language=?",
                (language,),
            )

        result: set[int] = set()
        for row_id, params_json in cur.fetchall():
            params = json.loads(params_json)
            if abs(len(params) - param_count) <= 1:
                result.add(row_id)
        return result

    def get_by_file(self, file_path: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM methods WHERE file_path=?", (file_path,)
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def _get_method_by_id(self, db_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM methods WHERE id=?", (db_id,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> dict:
        cols = [d[0] for d in self._conn.execute("SELECT * FROM methods LIMIT 0").description]
        d = dict(zip(cols, row))
        d["parameters"] = json.loads(d["parameters"])
        d["dependencies"] = json.loads(d["dependencies"])
        d["ast_fingerprint"] = json.loads(d.get("ast_fingerprint") or "[]")
        return d

    def get_all_methods(self) -> list[dict]:
        """Return all methods stored in the database."""
        cur = self._conn.execute("SELECT * FROM methods")
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_embedding(self, faiss_pos: int) -> np.ndarray | None:
        """Reconstruct the stored embedding vector for a given FAISS position."""
        if faiss_pos is None or faiss_pos < 0 or faiss_pos >= self._faiss_index.ntotal:
            return None
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        self._faiss_index.reconstruct(faiss_pos, vec)
        return vec

    def stats(self) -> dict:
        cur = self._conn.execute("SELECT COUNT(*) FROM methods")
        count = cur.fetchone()[0]
        cur2 = self._conn.execute("SELECT COUNT(DISTINCT file_path) FROM methods")
        files = cur2.fetchone()[0]
        return {"methods": count, "files": files, "faiss_total": self._faiss_index.ntotal}

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Chunk storage — SQLite table + dedicated FAISS index
    # ------------------------------------------------------------------

    def _init_chunks_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_id INTEGER NOT NULL REFERENCES methods(id),
                chunk_index INTEGER NOT NULL,
                statement_start INTEGER NOT NULL,
                statement_end INTEGER NOT NULL,
                statement_indices TEXT NOT NULL,
                function_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                depends_on_chunks TEXT NOT NULL,
                depended_on_by_chunks TEXT NOT NULL,
                normalized_code TEXT NOT NULL DEFAULT '',
                code_hash TEXT NOT NULL DEFAULT '',
                dependency_links TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                faiss_pos INTEGER
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_function ON chunks(function_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)"
        )
        # Migrate existing databases that pre-date the new schema columns.
        existing_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        for col_name, col_def in [
            ("normalized_code", "TEXT NOT NULL DEFAULT ''"),
            ("code_hash", "TEXT NOT NULL DEFAULT ''"),
            ("dependency_links", "TEXT NOT NULL DEFAULT '[]'"),
            ("created_at", "TEXT NOT NULL DEFAULT (datetime('now'))"),
        ]:
            if col_name not in existing_cols:
                self._conn.execute(
                    f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}"
                )
        self._conn.commit()

    def _load_or_create_chunk_index(self):
        if self._chunks_faiss_path.exists() and self._chunk_id_map_path.exists():
            index = faiss.read_index(str(self._chunks_faiss_path))
            id_map: dict[int, int] = {
                int(k): v
                for k, v in json.loads(self._chunk_id_map_path.read_text()).items()
            }
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            id_map = {}
        return index, id_map

    def _save_chunk_index(self) -> None:
        faiss.write_index(self._chunks_index, str(self._chunks_faiss_path))
        self._chunk_id_map_path.write_text(json.dumps(self._chunk_id_map))

    def add_chunk(
        self,
        chunk_info,
        embedding: np.ndarray,
        normalized_code: str = "",
    ) -> int:
        """Insert a chunk and its embedding. Returns the chunk DB row id.

        Args:
            chunk_info: A :class:`~code_similarity_mcp.parser.base.ChunkInfo`
                instance describing the chunk.
            embedding: Pre-computed L2-normalised embedding vector (shape
                ``(384,)``, dtype ``float32``).
            normalized_code: Normalized source text for this chunk (as produced
                by ``embed_chunks`` with ``return_texts=True``).  Used to store
                ``normalized_code`` and to compute ``code_hash``.  Defaults to
                ``""`` when not supplied (e.g. for legacy callers).
        """
        code_hash = (
            hashlib.sha256(normalized_code.encode()).hexdigest()
            if normalized_code
            else ""
        )
        dependency_links = json.dumps(chunk_info.depends_on_chunks)

        cur = self._conn.execute(
            """INSERT INTO chunks
               (function_id, chunk_index, statement_start, statement_end,
                statement_indices, function_name, file_path,
                depends_on_chunks, depended_on_by_chunks,
                normalized_code, code_hash, dependency_links,
                faiss_pos)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk_info.function_id,
                chunk_info.chunk_index,
                chunk_info.statement_start,
                chunk_info.statement_end,
                json.dumps(chunk_info.statement_indices),
                chunk_info.function_name,
                chunk_info.file_path,
                json.dumps(chunk_info.depends_on_chunks),
                json.dumps(chunk_info.depended_on_by_chunks),
                normalized_code,
                code_hash,
                dependency_links,
                None,  # faiss_pos set below
            ),
        )
        chunk_db_id = cur.lastrowid

        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss_pos = self._chunks_index.ntotal
        self._chunks_index.add(vec)
        self._chunk_id_map[faiss_pos] = chunk_db_id

        self._conn.execute(
            "UPDATE chunks SET faiss_pos=? WHERE id=?", (faiss_pos, chunk_db_id)
        )
        self._conn.commit()
        self._save_chunk_index()
        return chunk_db_id

    def delete_chunk_by_id(self, chunk_id: int) -> bool:
        """Remove a single chunk by its DB id.

        Removes the entry from *_chunk_id_map* and persists the updated chunk
        FAISS index to disk.  The FAISS vector at the former position remains
        in the index (FAISS does not support in-place deletion) but the orphaned
        position is no longer reachable via *_chunk_id_map*, so it will never
        appear in search results.

        Returns:
            ``True`` if the chunk existed and was removed; ``False`` otherwise.
        """
        cur = self._conn.execute(
            "SELECT faiss_pos FROM chunks WHERE id=?", (chunk_id,)
        )
        row = cur.fetchone()
        if row is None:
            return False

        faiss_pos = row[0]
        self._conn.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
        self._conn.commit()

        if faiss_pos is not None:
            self._chunk_id_map.pop(faiss_pos, None)
        self._save_chunk_index()
        return True

    def delete_chunks_by_function(self, function_id: int) -> int:
        """Remove all chunks for a function. Returns count removed."""
        cur = self._conn.execute(
            "SELECT id, faiss_pos FROM chunks WHERE function_id=?", (function_id,)
        )
        rows = cur.fetchall()
        if not rows:
            return 0

        self._conn.execute("DELETE FROM chunks WHERE function_id=?", (function_id,))
        self._conn.commit()

        positions = {row[1] for row in rows if row[1] is not None}
        for pos in positions:
            self._chunk_id_map.pop(pos, None)
        self._save_chunk_index()
        return len(rows)

    def get_chunks_by_function(self, function_id: int) -> list[dict]:
        """Return all stored chunks for a function."""
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE function_id=?", (function_id,)
        )
        return [self._chunk_row_to_dict(r) for r in cur.fetchall()]

    def get_chunks_by_file(self, file_path: str) -> list[dict]:
        """Return all stored chunks for a file."""
        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE file_path=?", (file_path,)
        )
        return [self._chunk_row_to_dict(r) for r in cur.fetchall()]

    def _chunk_row_to_dict(self, row) -> dict:
        cols = [
            d[0]
            for d in self._conn.execute("SELECT * FROM chunks LIMIT 0").description
        ]
        d = dict(zip(cols, row))
        d["statement_indices"] = json.loads(d["statement_indices"])
        d["depends_on_chunks"] = json.loads(d["depends_on_chunks"])
        d["depended_on_by_chunks"] = json.loads(d["depended_on_by_chunks"])
        d["dependency_links"] = json.loads(d.get("dependency_links") or "[]")
        return d

    def get_chunk_count(self) -> int:
        """Return total number of stored chunks."""
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]

    def get_chunk_by_id(self, chunk_id: int) -> dict | None:
        """Return a single chunk by its DB id, or None if not found."""
        cur = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        row = cur.fetchone()
        return self._chunk_row_to_dict(row) if row else None

    def get_chunk_embedding(self, faiss_pos: int) -> np.ndarray | None:
        """Reconstruct the stored chunk embedding vector for a given FAISS position."""
        if faiss_pos is None or faiss_pos < 0 or faiss_pos >= self._chunks_index.ntotal:
            return None
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        self._chunks_index.reconstruct(faiss_pos, vec)
        return vec

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        allowed_chunk_ids: set[int] | None = None,
    ) -> list[dict]:
        """Return top-k similar chunks by embedding similarity.

        If *allowed_chunk_ids* is provided, only chunks whose DB id is in that
        set are returned.  Pass an empty set to short-circuit (returns []).
        """
        if self._chunks_index.ntotal == 0:
            return []
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return []
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._chunks_index.ntotal)
        scores, positions = self._chunks_index.search(vec, k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos not in self._chunk_id_map:
                continue
            chunk_db_id = self._chunk_id_map[pos]
            if allowed_chunk_ids is not None and chunk_db_id not in allowed_chunk_ids:
                continue
            row = self.get_chunk_by_id(chunk_db_id)
            if row:
                row["embedding_score"] = float(score)
                results.append(row)
        return results
