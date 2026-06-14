"""
Persistent Job Tracking for Resumable Indexing

Provides:
- Per-stage progress tracking
- Resumable indexing from crash points
- Detailed audit trail of what happened
- Automatic crash recovery
"""

import sqlite3
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class PipelineStage(str, Enum):
    """Pipeline stages in order"""
    CONVERSION = "conversion"
    CHUNKING = "chunking"
    OVERVIEW = "overview"
    ENRICHMENT = "enrichment"
    EMBEDDING = "embedding"
    STORAGE = "storage"


STAGE_ORDER = [
    PipelineStage.CONVERSION,
    PipelineStage.CHUNKING,
    PipelineStage.OVERVIEW,
    PipelineStage.ENRICHMENT,
    PipelineStage.EMBEDDING,
    PipelineStage.STORAGE,
]


class JobProgressTracker:
    """Tracks and persists indexing job progress"""

    def __init__(self, db_path: str = "backend/chat_data.db"):
        self.db_path = db_path

    def _get_conn(self):
        """Open a fresh per-call connection.

        Persistent connections in DELETE journal mode hold a write lock for the
        lifetime of the process, blocking all other writers.  A fresh connection
        is cheaper than debugging lock timeouts and is safe in CPython where the
        connection is closed immediately when the local reference drops.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ========================================================================
    # JOB-LEVEL OPERATIONS
    # ========================================================================

    def mark_job_resuming(self, job_id: str) -> Dict[str, Any]:
        """
        Mark a job as resuming (for crash recovery).
        Returns what needs to be reprocessed.
        """
        db = self._get_conn()
        cursor = db.cursor()

        try:
            # Get job info
            cursor.execute("SELECT status, updated_at FROM index_jobs WHERE id = ?", (job_id,))
            job = cursor.fetchone()

            if not job:
                return {"error": "Job not found", "job_id": job_id}

            # Update job to queued; the backend runner will move it to running.
            cursor.execute(
                """
                UPDATE index_jobs
                SET status = 'queued', stage = 'queued', message = 'Queued for resume after crash', updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), job_id),
            )

            # Get files that need reprocessing
            cursor.execute(
                """
                SELECT id, filename FROM index_job_files
                WHERE job_id = ? AND status != 'done'
                ORDER BY filename
                """,
                (job_id,),
            )

            files_to_retry = [dict(row) for row in cursor.fetchall()]

            db.commit()

            return {
                "job_id": job_id,
                "status": "resuming",
                "files_to_retry": files_to_retry,
                "total_files": len(files_to_retry),
            }

        except Exception as e:
            db.rollback()
            return {"error": str(e), "job_id": job_id}

    def get_job_timeline(self, job_id: str) -> Dict[str, Any]:
        """Get complete timeline of events for a job"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            # Get job info
            cursor.execute(
                """
                SELECT id, status, progress, message, error, created_at, updated_at, finished_at
                FROM index_jobs WHERE id = ?
                """,
                (job_id,),
            )

            job = cursor.fetchone()
            if not job:
                return {"error": "Job not found"}

            # Get files and their stage progress
            cursor.execute(
                """
                SELECT 
                    f.id, f.filename, f.status, f.attempt_count,
                    COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as completed_stages,
                    COUNT(s.id) as total_stages
                FROM index_job_files f
                LEFT JOIN index_job_file_stages s ON f.id = s.file_id
                WHERE f.job_id = ?
                GROUP BY f.id
                ORDER BY f.filename
                """,
                (job_id,),
            )

            files_timeline = [dict(row) for row in cursor.fetchall()]

            # Get stage details for each file
            for file_entry in files_timeline:
                cursor.execute(
                    """
                    SELECT stage_name, status, started_at, finished_at, duration_seconds, error
                    FROM index_job_file_stages
                    WHERE file_id = ?
                    ORDER BY stage_name
                    """,
                    (file_entry["id"],),
                )

                file_entry["stages"] = [dict(row) for row in cursor.fetchall()]

            return {
                "job_id": job_id,
                "job_status": job["status"],
                "progress": job["progress"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "finished_at": job["finished_at"],
                "files": files_timeline,
                "total_files": len(files_timeline),
                "completed_files": sum(1 for f in files_timeline if f["status"] == "done"),
                "failed_files": sum(1 for f in files_timeline if f["status"] == "failed"),
                "pending_files": sum(1 for f in files_timeline if f["status"] == "pending"),
            }

        except Exception as e:
            return {"error": str(e), "job_id": job_id}

    # ========================================================================
    # FILE-LEVEL OPERATIONS
    # ========================================================================

    def get_or_create_file_record(
        self,
        job_id: str,
        index_id: str,
        stored_path: str,
        filename: Optional[str] = None,
    ) -> Optional[int]:
        """Return the index_job_files id for a path, creating it if needed."""
        db = self._get_conn()
        cursor = db.cursor()
        filename = filename or Path(stored_path).name

        try:
            cursor.execute(
                """
                SELECT id FROM index_job_files
                WHERE job_id = ? AND (stored_path = ? OR filename = ?)
                ORDER BY id
                LIMIT 1
                """,
                (job_id, stored_path, filename),
            )
            row = cursor.fetchone()
            if row:
                file_id = int(row["id"])
                cursor.execute(
                    """
                    UPDATE index_job_files
                    SET status = 'in_progress', started_at = COALESCE(started_at, ?), updated_at = ?
                    WHERE id = ? AND status IN ('pending', 'failed', 'in_progress')
                    """,
                    (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), file_id),
                )
                db.commit()
                return file_id

            now = datetime.utcnow().isoformat()
            cursor.execute(
                """
                INSERT INTO index_job_files (
                    job_id, index_id, stored_path, filename, status, started_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'in_progress', ?, ?)
                """,
                (job_id, index_id, stored_path, filename, now, now),
            )
            db.commit()
            return int(cursor.lastrowid)

        except Exception:
            db.rollback()
            return None

    def should_skip_stage(self, file_id: int, stage_name: str) -> bool:
        """Check if a stage has already been completed for this file"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            cursor.execute(
                """
                SELECT status FROM index_job_file_stages
                WHERE file_id = ? AND stage_name = ?
                """,
                (file_id, stage_name),
            )

            result = cursor.fetchone()
            if not result:
                return False

            return result["status"] == "completed"

        except Exception:
            return False

    def start_stage(self, file_id: int, job_id: str, stage_name: str) -> None:
        """Mark a stage as starting"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            now = datetime.utcnow().isoformat()
            cursor.execute(
                """
                UPDATE index_job_file_stages
                SET status = 'in_progress', started_at = ?, finished_at = NULL,
                    duration_seconds = NULL, error = NULL
                WHERE file_id = ? AND stage_name = ?
                """,
                (now, file_id, stage_name),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO index_job_file_stages
                    (file_id, job_id, stage_name, status, started_at)
                    VALUES (?, ?, ?, 'in_progress', ?)
                    """,
                    (file_id, job_id, stage_name, now),
                )

            db.commit()

        except Exception as e:
            db.rollback()
            raise e

    def complete_stage(
        self, file_id: int, stage_name: str, output_hash: Optional[str] = None
    ) -> None:
        """Mark a stage as completed"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            now = datetime.utcnow()

            # Get start time to calculate duration
            cursor.execute(
                """
                SELECT started_at FROM index_job_file_stages
                WHERE file_id = ? AND stage_name = ?
                """,
                (file_id, stage_name),
            )

            result = cursor.fetchone()
            duration = None

            if result and result["started_at"]:
                started = datetime.fromisoformat(result["started_at"])
                duration = (now - started).total_seconds()

            cursor.execute(
                """
                UPDATE index_job_file_stages
                SET status = 'completed', finished_at = ?, duration_seconds = ?, output_hash = ?
                WHERE file_id = ? AND stage_name = ?
                """,
                (now.isoformat(), duration, output_hash, file_id, stage_name),
            )

            db.commit()

        except Exception as e:
            db.rollback()
            raise e

    def fail_stage(self, file_id: int, stage_name: str, error: str) -> None:
        """Mark a stage as failed"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            now = datetime.utcnow()
            cursor.execute(
                """
                SELECT started_at FROM index_job_file_stages
                WHERE file_id = ? AND stage_name = ?
                """,
                (file_id, stage_name),
            )
            result = cursor.fetchone()
            duration = None
            if result and result["started_at"]:
                started = datetime.fromisoformat(result["started_at"])
                duration = (now - started).total_seconds()

            cursor.execute(
                """
                UPDATE index_job_file_stages
                SET status = 'failed', finished_at = ?, duration_seconds = ?, error = ?
                WHERE file_id = ? AND stage_name = ?
                """,
                (now.isoformat(), duration, error, file_id, stage_name),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO index_job_file_stages
                    (file_id, job_id, stage_name, status, finished_at, error)
                    SELECT ?, job_id, ?, 'failed', ?, ?
                    FROM index_job_files
                    WHERE id = ?
                    """,
                    (file_id, stage_name, now.isoformat(), error, file_id),
                )

            db.commit()

        except Exception as e:
            db.rollback()
            raise e

    # ========================================================================
    # FILE STATUS UPDATES
    # ========================================================================

    def get_file_progress(self, file_id: int) -> Dict[str, Any]:
        """Get current progress for a file across all stages"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            cursor.execute(
                """
                SELECT filename, status, attempt_count, chunks_generated
                FROM index_job_files WHERE id = ?
                """,
                (file_id,),
            )

            file_info = cursor.fetchone()

            if not file_info:
                return {"error": "File not found"}

            # Get stage progress
            cursor.execute(
                """
                SELECT stage_name, status, duration_seconds
                FROM index_job_file_stages
                WHERE file_id = ?
                ORDER BY stage_name
                """,
                (file_id,),
            )

            stages = [dict(row) for row in cursor.fetchall()]

            completed_stages = sum(1 for s in stages if s["status"] == "completed")
            total_stages = len(stages)

            return {
                "file_id": file_id,
                "filename": file_info["filename"],
                "overall_status": file_info["status"],
                "attempt_count": file_info["attempt_count"],
                "chunks_generated": file_info["chunks_generated"],
                "stages": stages,
                "completed_stages": completed_stages,
                "total_stages": total_stages,
                "progress_percent": int((completed_stages / total_stages * 100) if total_stages > 0 else 0),
            }

        except Exception as e:
            return {"error": str(e)}

    def mark_file_failed(self, file_id: int, error: str, error_code: str = "unknown") -> None:
        """Mark a file as failed and increment attempt count"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            cursor.execute(
                """
                UPDATE index_job_files
                SET status = 'failed', error = ?, last_error_code = ?,
                    attempt_count = attempt_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (error, error_code, datetime.utcnow().isoformat(), file_id),
            )

            db.commit()

        except Exception as e:
            db.rollback()
            raise e

    def mark_file_done(self, file_id: int, chunks_generated: int = 0) -> None:
        """Mark a file as successfully completed"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            cursor.execute(
                """
                UPDATE index_job_files
                SET status = 'done', chunks_generated = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (chunks_generated, datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), file_id),
            )

            db.commit()

        except Exception as e:
            db.rollback()
            raise e

    # ========================================================================
    # RECOVERY & DIAGNOSTICS
    # ========================================================================

    def recover_stale_jobs(self, older_than_minutes: int = 5) -> Dict[str, Any]:
        """
        Find jobs that have been building for too long without updates (likely crashed).
        Mark them as paused so they can be resumed.
        """
        db = self._get_conn()
        cursor = db.cursor()

        try:
            from datetime import timedelta

            stale_threshold = datetime.utcnow() - timedelta(minutes=older_than_minutes)
            stale_time = stale_threshold.isoformat()

            # Find stale jobs. The backend uses "running"; older docs/code used "building".
            cursor.execute(
                """
                SELECT id, index_id, created_at, updated_at
                FROM index_jobs
                WHERE status IN ('building', 'running') AND updated_at < ?
                ORDER BY updated_at ASC
                """,
                (stale_time,),
            )

            stale_jobs = cursor.fetchall()
            recovered = 0

            for job in stale_jobs:
                job_id = job["id"]

                # Mark as paused (not failed, so it can be resumed)
                cursor.execute(
                    """
                    UPDATE index_jobs
                    SET status = 'paused', message = 'Auto-recovered from stale state', updated_at = ?
                    WHERE id = ?
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )
                cursor.execute(
                    """
                    UPDATE index_job_files
                    SET status = 'pending', stage = NULL, updated_at = ?
                    WHERE job_id = ? AND status IN ('processing', 'in_progress')
                    """,
                    (datetime.utcnow().isoformat(), job_id),
                )
                cursor.execute(
                    """
                    UPDATE index_job_file_stages
                    SET status = 'pending', finished_at = NULL,
                        duration_seconds = NULL
                    WHERE job_id = ? AND status = 'in_progress'
                    """,
                    (job_id,),
                )

                recovered += 1

            db.commit()

            return {
                "found": len(stale_jobs),
                "recovered": recovered,
                "jobs": [dict(job) for job in stale_jobs],
            }

        except Exception as e:
            db.rollback()
            return {"error": str(e), "found": 0, "recovered": 0}

    def get_job_statistics(self, job_id: str) -> Dict[str, Any]:
        """Get statistics about a job's execution"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            # Overall stats
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as completed_files,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_files,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_files,
                    SUM(chunks_generated) as total_chunks,
                    MAX(attempt_count) as max_attempts,
                    AVG(attempt_count) as avg_attempts
                FROM index_job_files
                WHERE job_id = ?
                """,
                (job_id,),
            )

            stats = dict(cursor.fetchone() or {})

            # Stage timing stats
            cursor.execute(
                """
                SELECT 
                    stage_name,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM index_job_file_stages
                WHERE job_id = ?
                GROUP BY stage_name
                ORDER BY stage_name
                """,
                (job_id,),
            )

            stage_stats = [dict(row) for row in cursor.fetchall()]

            return {
                "job_id": job_id,
                "overall": stats,
                "by_stage": stage_stats,
            }

        except Exception as e:
            return {"error": str(e)}

    def export_audit_trail(self, job_id: str) -> List[Dict[str, Any]]:
        """Export complete audit trail for a job (for diagnostics)"""
        db = self._get_conn()
        cursor = db.cursor()

        try:
            cursor.execute(
                """
                SELECT 
                    f.filename,
                    s.stage_name,
                    s.status,
                    s.started_at,
                    s.finished_at,
                    s.duration_seconds,
                    s.error
                FROM index_job_file_stages s
                JOIN index_job_files f ON s.file_id = f.id
                WHERE s.job_id = ?
                ORDER BY f.filename, s.stage_name
                """,
                (job_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            return [{"error": str(e)}]
