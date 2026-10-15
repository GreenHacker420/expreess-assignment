"""SQLite database service for documents registry, query logs, and user feedback."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DatabaseService(Protocol):
    """Protocol for application persistence."""

    def record_document(
        self, document_id: str, source: str, title: str, chunk_count: int, content_hash: str
    ) -> None: ...

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]: ...

    def get_document_by_source(self, source: str) -> Optional[Dict[str, Any]]: ...

    def list_documents(self) -> List[Dict[str, Any]]: ...

    def count_documents(self) -> int: ...

    def delete_document(self, document_id: str) -> bool: ...

    def record_query(
        self,
        query_id: str,
        question: str,
        answer: str,
        status: str,
        retry_count: int,
        retrieval_attempts: int,
        citations: List[Dict[str, Any]],
    ) -> None: ...

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]: ...

    def record_feedback(
        self, query_id: str, rating: str, comment: Optional[str] = None
    ) -> str: ...


class SQLiteDatabaseService:
    """Thread-safe SQLite service for single-process local storage."""

    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path
        parent = Path(db_path).parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    retrieval_attempts INTEGER NOT NULL,
                    citations TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (query_id) REFERENCES query_logs (query_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback (query_id);
                """
            )

    def record_document(
        self, document_id: str, source: str, title: str, chunk_count: int, content_hash: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, source, title, chunk_count, content_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    chunk_count = excluded.chunk_count,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at;
                """,
                (document_id, source, title, chunk_count, content_hash, now, now),
            )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?;", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_document_by_source(self, source: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source = ?;", (source,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT document_id, source, title, chunk_count, updated_at FROM documents ORDER BY updated_at DESC;"
            ).fetchall()
            return [dict(r) for r in rows]

    def count_documents(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM documents;").fetchone()
            return row["total"] if row else 0

    def delete_document(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?;", (document_id,))
            return cur.rowcount > 0

    def record_query(
        self,
        query_id: str,
        question: str,
        answer: str,
        status: str,
        retry_count: int,
        retrieval_attempts: int,
        citations: List[Dict[str, Any]],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (query_id, question, answer, status, retry_count, retrieval_attempts, citations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (query_id, question, answer, status, retry_count, retrieval_attempts, citations_json, now),
            )

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM query_logs WHERE query_id = ?;", (query_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            try:
                data["citations"] = json.loads(data["citations"])
            except Exception:
                data["citations"] = []
            return data

    def record_feedback(
        self, query_id: str, rating: str, comment: Optional[str] = None
    ) -> str:
        # Validate query exists
        if not self.get_query(query_id):
            raise KeyError(f"Query {query_id} does not exist.")
        feedback_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (feedback_id, query_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (feedback_id, query_id, rating, comment, now),
            )
        return feedback_id


class MockDatabaseService:
    """In-memory mock database for testing."""

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.query_logs: Dict[str, Dict[str, Any]] = {}
        self.feedback: Dict[str, Dict[str, Any]] = {}

    def record_document(
        self, document_id: str, source: str, title: str, chunk_count: int, content_hash: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.documents[document_id] = {
            "document_id": document_id,
            "source": source,
            "title": title,
            "chunk_count": chunk_count,
            "content_hash": content_hash,
            "created_at": now,
            "updated_at": now,
        }

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(document_id)

    def get_document_by_source(self, source: str) -> Optional[Dict[str, Any]]:
        for doc in self.documents.values():
            if doc["source"] == source:
                return doc
        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def count_documents(self) -> int:
        return len(self.documents)

    def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    def record_query(
        self,
        query_id: str,
        question: str,
        answer: str,
        status: str,
        retry_count: int,
        retrieval_attempts: int,
        citations: List[Dict[str, Any]],
    ) -> None:
        self.query_logs[query_id] = {
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "status": status,
            "retry_count": retry_count,
            "retrieval_attempts": retrieval_attempts,
            "citations": citations,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        return self.query_logs.get(query_id)

    def record_feedback(
        self, query_id: str, rating: str, comment: Optional[str] = None
    ) -> str:
        if query_id not in self.query_logs:
            raise KeyError(f"Query {query_id} does not exist.")
        feedback_id = str(uuid.uuid4())
        self.feedback[feedback_id] = {
            "feedback_id": feedback_id,
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return feedback_id
