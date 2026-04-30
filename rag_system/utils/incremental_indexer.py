"""
Incremental Indexing System for LocalGPT

Implements intelligent document change detection and incremental indexing
to avoid re-processing unchanged documents and re-embedding the entire corpus.
"""

import logging
import os
import hashlib
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for tracking document state"""
    file_path: str
    file_hash: str
    modification_time: float
    size: int
    last_indexed: Optional[float] = None
    chunk_count: int = 0
    index_id: Optional[str] = None

class IncrementalIndexer:
    """
    Manages incremental indexing by tracking document changes and
    only re-processing modified or new documents.
    """

    def __init__(self, db_path: str = "backend/chat_data.db", index_store_path: str = "index_store"):
        """
        Initialize incremental indexer.

        Args:
            db_path: Path to the SQLite database
            index_store_path: Path to the index store directory
        """
        self.db_path = db_path
        self.index_store_path = Path(index_store_path)
        self.index_store_path.mkdir(exist_ok=True)

        # Initialize database tables
        self._init_database()

    def _init_database(self):
        """Initialize database tables for incremental indexing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Document metadata tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_metadata (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                modification_time REAL NOT NULL,
                file_size INTEGER NOT NULL,
                last_indexed REAL,
                chunk_count INTEGER DEFAULT 0,
                index_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # Index operations log for rollback/debugging
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                file_path TEXT,
                chunk_count INTEGER,
                timestamp REAL DEFAULT (strftime('%s', 'now')),
                status TEXT DEFAULT 'completed'
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_meta_index_id ON document_metadata(index_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_meta_last_indexed ON document_metadata(last_indexed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_index_ops_index_id ON index_operations(index_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_index_ops_timestamp ON index_operations(timestamp)')

        conn.commit()
        conn.close()

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_file_metadata(self, file_path: str) -> DocumentMetadata:
        """Get current file metadata"""
        stat = os.stat(file_path)
        file_hash = self.calculate_file_hash(file_path)

        return DocumentMetadata(
            file_path=file_path,
            file_hash=file_hash,
            modification_time=stat.st_mtime,
            size=stat.st_size
        )

    def get_stored_metadata(self, file_path: str) -> Optional[DocumentMetadata]:
        """Get stored metadata from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT file_hash, modification_time, file_size, last_indexed, chunk_count, index_id
            FROM document_metadata
            WHERE file_path = ?
        ''', (file_path,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return DocumentMetadata(
                file_path=file_path,
                file_hash=row[0],
                modification_time=row[1],
                size=row[2],
                last_indexed=row[3],
                chunk_count=row[4],
                index_id=row[5]
            )
        return None

    def has_document_changed(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if document has changed since last indexing.

        Returns:
            (has_changed, reason)
        """
        if not os.path.exists(file_path):
            return False, "file does not exist"

        current_meta = self.get_file_metadata(file_path)
        stored_meta = self.get_stored_metadata(file_path)

        if stored_meta is None:
            return True, "new document"

        # Check for changes
        if current_meta.file_hash != stored_meta.file_hash:
            return True, "content changed"

        if current_meta.modification_time != stored_meta.modification_time:
            return True, "modification time changed"

        if current_meta.size != stored_meta.size:
            return True, "file size changed"

        return False, None

    def detect_changes(self, file_paths: List[str]) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Detect which documents have changed.

        Returns:
            Dict[file_path, (has_changed, reason)]
        """
        changes = {}
        for file_path in file_paths:
            has_changed, reason = self.has_document_changed(file_path)
            changes[file_path] = (has_changed, reason)

        return changes

    def update_document_metadata(self, file_path: str, index_id: str,
                               chunk_count: int, operation: str = "index"):
        """Update document metadata after indexing"""
        current_meta = self.get_file_metadata(file_path)
        now = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update or insert metadata
        cursor.execute('''
            INSERT OR REPLACE INTO document_metadata
            (file_path, file_hash, modification_time, file_size, last_indexed, chunk_count, index_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_path,
            current_meta.file_hash,
            current_meta.modification_time,
            current_meta.size,
            now,
            chunk_count,
            index_id,
            now
        ))

        # Log the operation
        cursor.execute('''
            INSERT INTO index_operations (index_id, operation, file_path, chunk_count, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (index_id, operation, file_path, chunk_count, now))

        conn.commit()
        conn.close()

    def get_incremental_file_list(self, file_paths: List[str],
                                force_reindex: bool = False) -> Tuple[List[str], List[str]]:
        """
        Get lists of files that need indexing vs those that are unchanged.

        Args:
            file_paths: All file paths to consider
            force_reindex: If True, reindex all files regardless of changes

        Returns:
            (files_to_index, unchanged_files)
        """
        if force_reindex:
            return file_paths, []

        changes = self.detect_changes(file_paths)
        files_to_index = []
        unchanged_files = []

        for file_path, (has_changed, reason) in changes.items():
            if has_changed:
                files_to_index.append(file_path)
                logger.info("file_needs_indexing file_path=%s reason=%s", file_path, reason)
            else:
                unchanged_files.append(file_path)
                logger.debug("file_unchanged file_path=%s", file_path)

        return files_to_index, unchanged_files

    def get_index_stats(self, index_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if index_id:
            # Stats for specific index
            cursor.execute('''
                SELECT COUNT(*), SUM(chunk_count), SUM(file_size), MAX(last_indexed)
                FROM document_metadata
                WHERE index_id = ?
            ''', (index_id,))
        else:
            # Global stats
            cursor.execute('''
                SELECT COUNT(*), SUM(chunk_count), SUM(file_size), MAX(last_indexed)
                FROM document_metadata
            ''')

        row = cursor.fetchone()

        # Get operation history
        if index_id:
            cursor.execute('''
                SELECT operation, COUNT(*), SUM(chunk_count)
                FROM index_operations
                WHERE index_id = ?
                GROUP BY operation
            ''', (index_id,))
        else:
            cursor.execute('''
                SELECT operation, COUNT(*), SUM(chunk_count)
                FROM index_operations
                GROUP BY operation
            ''')

        operations = cursor.fetchall()
        conn.close()

        return {
            'total_documents': row[0] or 0,
            'total_chunks': row[1] or 0,
            'total_size_bytes': row[2] or 0,
            'last_indexed': row[3],
            'operations': {
                op[0]: {'count': op[1], 'chunks': op[2] or 0}
                for op in operations
            }
        }

    def cleanup_orphaned_metadata(self, existing_files: List[str]) -> int:
        """
        Remove metadata for files that no longer exist.

        Returns:
            Number of entries cleaned up
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all tracked files
        cursor.execute('SELECT file_path FROM document_metadata')
        tracked_files = {row[0] for row in cursor.fetchall()}

        # Find files that no longer exist
        existing_files_set = set(existing_files)
        orphaned_files = tracked_files - existing_files_set

        if orphaned_files:
            # Remove orphaned metadata
            cursor.executemany('''
                DELETE FROM document_metadata WHERE file_path = ?
            ''', [(f,) for f in orphaned_files])

            # Log cleanup operations
            now = time.time()
            cursor.executemany('''
                INSERT INTO index_operations (index_id, operation, file_path, timestamp, status)
                VALUES (?, 'cleanup', ?, ?, 'completed')
            ''', [('system', f, now) for f in orphaned_files])

            conn.commit()

        conn.close()
        return len(orphaned_files)

    def reset_index(self, index_id: str):
        """Reset all metadata for an index (useful for forced reindexing)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Remove metadata for this index
        cursor.execute('DELETE FROM document_metadata WHERE index_id = ?', (index_id,))

        # Log the reset operation
        now = time.time()
        cursor.execute('''
            INSERT INTO index_operations (index_id, operation, timestamp, status)
            VALUES (?, 'reset', ?, 'completed')
        ''', (index_id, now))

        conn.commit()
        conn.close()

        logger.info("incremental_index_reset index_id=%s", index_id)
