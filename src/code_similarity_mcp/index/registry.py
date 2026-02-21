"""SQLite-backed method registry + FAISS vector index."""

from __future__ import annotations

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

        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()
        self._faiss_index, self._id_map = self._load_or_create_index()

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

    def delete_by_file(self, file_path: str) -> int:
        """Remove all methods for a file. Returns count removed."""
        cur = self._conn.execute(
            "SELECT id, faiss_pos FROM methods WHERE file_path=?", (file_path,)
        )
        rows = cur.fetchall()
        if not rows:
            return 0

        self._conn.execute("DELETE FROM methods WHERE file_path=?", (file_path,))
        self._conn.commit()

        # Mark FAISS positions as invalid in the id_map
        positions = {row[1] for row in rows if row[1] is not None}
        for pos in positions:
            self._id_map.pop(pos, None)
        self._save_index()
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
