"""
MemoryStore — SQLite-backed persistent memory for OmicVerse Jarvis.

Inspired by metabot's MemoryStorage design:
  • Hierarchical folders (path-based)
  • Markdown documents with tags
  • FTS5 full-text search across title / content / tags
  • Channel + session metadata for provenance
  • Pure stdlib (sqlite3) — no external deps

Database file: ``~/.ovjarvis/memory.db`` (default, configurable).

Usage::

    from omicverse.jarvis.memory import MemoryStore

    store = MemoryStore()
    doc = store.create_document(
        title="UMAP analysis of PBMC",
        content="Performed UMAP with n_neighbors=15 ...",
        tags=["umap", "pbmc"],
        channel="telegram",
        session_id="9d3d0d20c3b7690b",
    )
    results = store.search("UMAP PBMC")
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryFolder:
    id: str
    name: str
    parent_id: Optional[str]
    path: str
    created_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "path": self.path,
            "created_at": self.created_at,
        }


@dataclass
class MemoryDocument:
    id: str
    title: str
    folder_id: str
    folder_path: str
    content: str
    tags: List[str]
    channel: str        # which channel created this ("telegram", "web", ...)
    session_id: str     # stable web session_id (from WebSessionBridge)
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "folder_id": self.folder_id,
            "folder_path": self.folder_path,
            "content": self.content,
            "tags": self.tags,
            "channel": self.channel,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class SearchResult:
    id: str
    title: str
    folder_path: str
    snippet: str
    tags: List[str]
    channel: str
    session_id: str
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "folder_path": self.folder_path,
            "snippet": self.snippet,
            "tags": self.tags,
            "channel": self.channel,
            "session_id": self.session_id,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".ovjarvis", "memory.db")

# Each statement is a separate string to avoid mis-splitting on semicolons
# inside trigger bodies.
_DDL_STMTS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    """CREATE TABLE IF NOT EXISTS folders (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        parent_id  TEXT REFERENCES folders(id),
        path       TEXT UNIQUE NOT NULL,
        created_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        folder_id   TEXT NOT NULL DEFAULT 'root' REFERENCES folders(id) ON DELETE CASCADE,
        folder_path TEXT NOT NULL DEFAULT '/',
        content     TEXT DEFAULT '',
        tags        TEXT DEFAULT '[]',
        channel     TEXT DEFAULT '',
        session_id  TEXT DEFAULT '',
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        title,
        content,
        tags,
        doc_id UNINDEXED
    )""",
    # Triggers — kept as separate statements to avoid semicolon-split issues
    """CREATE TRIGGER IF NOT EXISTS docs_fts_insert AFTER INSERT ON documents
       BEGIN
         INSERT INTO documents_fts(title, content, tags, doc_id)
         VALUES (new.title, new.content, new.tags, new.id);
       END""",
    """CREATE TRIGGER IF NOT EXISTS docs_fts_update AFTER UPDATE ON documents
       BEGIN
         DELETE FROM documents_fts WHERE doc_id = old.id;
         INSERT INTO documents_fts(title, content, tags, doc_id)
         VALUES (new.title, new.content, new.tags, new.id);
       END""",
    """CREATE TRIGGER IF NOT EXISTS docs_fts_delete AFTER DELETE ON documents
       BEGIN
         DELETE FROM documents_fts WHERE doc_id = old.id;
       END""",
]


def _seed_root(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO folders(id, name, parent_id, path, created_at) "
        "VALUES ('root', 'root', NULL, '/', ?)",
        (time.time(),),
    )
    conn.commit()


def _escape_fts5(query: str) -> str:
    """Escape a user query for FTS5 (wrap each token in double quotes)."""
    tokens = query.split()
    return " ".join(f'"{t.replace(chr(34), "")}"' for t in tokens if t)


class MemoryStore:
    """Persistent memory store backed by SQLite with FTS5 search.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created on first use.
    """

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        for stmt in _DDL_STMTS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()
        _seed_root(self._conn)

    # ------------------------------------------------------------------
    # Folders
    # ------------------------------------------------------------------

    def create_folder(
        self,
        name: str,
        parent_id: str = "root",
    ) -> MemoryFolder:
        with self._lock:
            parent = self._conn.execute(
                "SELECT path FROM folders WHERE id = ?", (parent_id,)
            ).fetchone()
            parent_path = parent["path"] if parent else "/"
            path = parent_path.rstrip("/") + "/" + name.strip("/")
            fid = str(uuid.uuid4())
            now = time.time()
            self._conn.execute(
                "INSERT INTO folders(id, name, parent_id, path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (fid, name, parent_id, path, now),
            )
            self._conn.commit()
            return MemoryFolder(id=fid, name=name, parent_id=parent_id, path=path, created_at=now)

    def list_folders(self) -> List[MemoryFolder]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, parent_id, path, created_at FROM folders ORDER BY path"
            ).fetchall()
        return [
            MemoryFolder(
                id=r["id"],
                name=r["name"],
                parent_id=r["parent_id"],
                path=r["path"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_folder_tree(self) -> List[dict]:
        """Return folders as a flat list ordered by path depth."""
        return [f.to_dict() for f in self.list_folders()]

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def create_document(
        self,
        title: str,
        content: str = "",
        tags: Optional[List[str]] = None,
        folder_id: str = "root",
        channel: str = "",
        session_id: str = "",
    ) -> MemoryDocument:
        with self._lock:
            folder = self._conn.execute(
                "SELECT path FROM folders WHERE id = ?", (folder_id,)
            ).fetchone()
            folder_path = folder["path"] if folder else "/"
            did = str(uuid.uuid4())
            now = time.time()
            tags_json = json.dumps(tags or [])
            self._conn.execute(
                "INSERT INTO documents(id, title, folder_id, folder_path, content, tags, "
                "channel, session_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (did, title, folder_id, folder_path, content, tags_json,
                 channel, session_id, now, now),
            )
            self._conn.commit()
        return MemoryDocument(
            id=did,
            title=title,
            folder_id=folder_id,
            folder_path=folder_path,
            content=content,
            tags=tags or [],
            channel=channel,
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )

    def update_document(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
    ) -> Optional[MemoryDocument]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return None
            new_title = title if title is not None else row["title"]
            new_content = content if content is not None else row["content"]
            new_tags = json.dumps(tags) if tags is not None else row["tags"]
            new_folder_id = folder_id if folder_id is not None else row["folder_id"]
            folder = self._conn.execute(
                "SELECT path FROM folders WHERE id = ?", (new_folder_id,)
            ).fetchone()
            new_folder_path = folder["path"] if folder else "/"
            now = time.time()
            self._conn.execute(
                "UPDATE documents SET title=?, content=?, tags=?, folder_id=?, "
                "folder_path=?, updated_at=? WHERE id=?",
                (new_title, new_content, new_tags, new_folder_id,
                 new_folder_path, now, doc_id),
            )
            self._conn.commit()
        return MemoryDocument(
            id=doc_id,
            title=new_title,
            folder_id=new_folder_id,
            folder_path=new_folder_path,
            content=new_content,
            tags=json.loads(new_tags) if isinstance(new_tags, str) else new_tags,
            channel=row["channel"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=now,
        )

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM documents WHERE id = ?", (doc_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_document(self, doc_id: str) -> Optional[MemoryDocument]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_doc(row)

    def list_documents(
        self,
        folder_id: Optional[str] = None,
        channel: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryDocument]:
        clauses, params = [], []
        if folder_id:
            clauses.append("folder_id = ?")
            params.append(folder_id)
        if channel:
            clauses.append("channel = ?")
            params.append(channel)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM documents {where} "
                f"ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def search(
        self,
        query: str,
        limit: int = 20,
        channel: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Full-text search across title, content, and tags."""
        if not query.strip():
            return []
        fts_q = _escape_fts5(query)
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.id, d.title, d.folder_path, d.tags, d.channel, d.session_id, "
                "d.updated_at, "
                "snippet(documents_fts, 1, '**', '**', '...', 24) AS snippet "
                "FROM documents_fts "
                "JOIN documents d ON documents_fts.doc_id = d.id "
                "WHERE documents_fts MATCH ? "
                + ("AND d.channel = ? " if channel else "")
                + ("AND d.session_id = ? " if session_id else "")
                + "ORDER BY rank LIMIT ?",
                [fts_q]
                + ([channel] if channel else [])
                + ([session_id] if session_id else [])
                + [limit],
            ).fetchall()
        results = []
        for r in rows:
            results.append(
                SearchResult(
                    id=r["id"],
                    title=r["title"],
                    folder_path=r["folder_path"],
                    snippet=r["snippet"] or "",
                    tags=json.loads(r["tags"] or "[]"),
                    channel=r["channel"],
                    session_id=r["session_id"],
                    updated_at=r["updated_at"],
                )
            )
        return results

    def stats(self) -> dict:
        with self._lock:
            doc_count = self._conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            folder_count = self._conn.execute(
                "SELECT COUNT(*) FROM folders"
            ).fetchone()[0]
        return {"document_count": doc_count, "folder_count": folder_count}

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> MemoryDocument:
        tags_raw = row["tags"] or "[]"
        try:
            tags = json.loads(tags_raw)
        except (json.JSONDecodeError, TypeError):
            tags = []
        return MemoryDocument(
            id=row["id"],
            title=row["title"],
            folder_id=row["folder_id"],
            folder_path=row["folder_path"],
            content=row["content"] or "",
            tags=tags,
            channel=row["channel"] or "",
            session_id=row["session_id"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
