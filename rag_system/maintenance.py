"""
System Maintenance Tools for LocalGPT

Provides utilities for:
1. Repairing stuck builds
2. Removing orphan uploaded files
3. Deleting broken indexes
4. Rebuilding failed files only
5. Listing index health
6. Exporting diagnostics bundle
"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MaintenanceTools:
    """Maintenance operations for LocalGPT system"""

    def __init__(
        self,
        db_path: str = "backend/chat_data.db",
        project_root: str = ".",
        lancedb_path: str = "lancedb",
        uploads_path: str = "shared_uploads",
        index_store_path: str = "index_store",
    ):
        self.db_path = db_path
        self.project_root = Path(project_root).resolve()
        self.lancedb_path = self.project_root / lancedb_path
        self.uploads_path = self.project_root / uploads_path
        self.index_store_path = self.project_root / index_store_path

    def _canonical_project_path(self, stored_path: str) -> Path:
        """Normalize absolute and legacy project-relative database paths."""
        path = Path(stored_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve(strict=False)

    @staticmethod
    def _lancedb_table_names(conn) -> List[str]:
        if hasattr(conn, "list_tables"):
            raw = conn.list_tables()
            if not isinstance(raw, dict):
                try:
                    raw = dict(raw)
                except (TypeError, ValueError):
                    raw = getattr(raw, "tables", raw)
            if isinstance(raw, dict):
                raw = raw.get("tables", [])
            return [item.name if hasattr(item, "name") else str(item) for item in raw]
        if hasattr(conn, "table_names"):
            return list(conn.table_names(limit=10_000))
        return []

    def _get_db(self):
        """Open a fresh per-call connection (avoids persistent write-lock holders)."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None

    # ========================================================================
    # 1. REPAIR STUCK BUILDS
    # ========================================================================

    def repair_stuck_builds(self, older_than_minutes: int = 30) -> Dict[str, Any]:
        """
        Find and repair stuck/stale build jobs.

        Args:
            older_than_minutes: Consider jobs older than this as stuck

        Returns:
            Report of repaired jobs
        """
        db = self._get_db()
        report: Dict[str, Any] = {
            "found": 0,
            "repaired": 0,
            "stuck_jobs": [],
            "errors": [],
        }

        try:
            cursor = db.cursor()
            stale_threshold = datetime.utcnow() - timedelta(minutes=older_than_minutes)
            stale_time = stale_threshold.isoformat()

            # Find stuck jobs (still in 'building' status after threshold)
            cursor.execute(
                """
                SELECT id, index_id, status, created_at, updated_at
                FROM index_jobs
                WHERE status IN ('building', 'queued')
                AND updated_at < ?
                ORDER BY updated_at ASC
            """,
                (stale_time,),
            )

            stuck_jobs = cursor.fetchall()
            report["found"] = len(stuck_jobs)

            for job in stuck_jobs:
                job_id = job["id"]
                index_id = job["index_id"]

                try:
                    # Mark job as failed with diagnosis
                    cursor.execute(
                        """
                        UPDATE index_jobs
                        SET status = 'failed',
                            error = 'Automatically recovered: Job was stuck',
                            updated_at = ?
                        WHERE id = ?
                    """,
                        (datetime.utcnow().isoformat(), job_id),
                    )

                    # Mark pending files as failed
                    cursor.execute(
                        """
                        UPDATE index_job_files
                        SET status = 'failed',
                            error = 'Recovered from stuck job'
                        WHERE job_id = ? AND status = 'pending'
                    """,
                        (job_id,),
                    )

                    db.commit()
                    report["repaired"] += 1
                    report["stuck_jobs"].append(
                        {
                            "job_id": job_id,
                            "index_id": index_id,
                            "status": "recovered",
                            "was_stuck_since": job["updated_at"],
                        }
                    )
                    logger.info(f"Recovered stuck job {job_id}")

                except Exception as e:
                    report["errors"].append(f"Failed to repair job {job_id}: {str(e)}")
                    db.rollback()

        except Exception as e:
            report["errors"].append(f"Database error: {str(e)}")
            logger.error(f"Error repairing stuck builds: {e}")

        return report

    # ========================================================================
    # 2. REMOVE ORPHAN UPLOADED FILES
    # ========================================================================

    def remove_orphan_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Find and remove uploaded files not associated with any index.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Report of orphan files found and removed
        """
        report: Dict[str, Any] = {
            "total_scanned": 0,
            "orphans_found": 0,
            "orphans_removed": 0,
            "total_freed_bytes": 0,
            "orphan_files": [],
            "errors": [],
        }

        try:
            db = self._get_db()
            cursor = db.cursor()

            # Get all files referenced in index_documents
            cursor.execute("SELECT stored_path FROM index_documents")
            referenced_files = {
                self._canonical_project_path(row["stored_path"])
                for row in cursor.fetchall()
                if row["stored_path"]
            }

            # Scan uploads directory
            if not self.uploads_path.exists():
                return report

            for file_path in self.uploads_path.glob("*"):
                if not file_path.is_file():
                    continue

                report["total_scanned"] += 1
                canonical_path = file_path.resolve(strict=False)
                relative_path = str(file_path.relative_to(self.project_root))

                # Check if file is referenced
                if canonical_path not in referenced_files:
                    report["orphans_found"] += 1
                    file_size = file_path.stat().st_size

                    report["orphan_files"].append(
                        {
                            "path": relative_path,
                            "size": file_size,
                            "size_str": self._format_bytes(file_size),
                            "modified": datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat(),
                        }
                    )

                    if not dry_run:
                        try:
                            file_path.unlink()
                            report["orphans_removed"] += 1
                            report["total_freed_bytes"] += file_size
                            logger.info(f"Removed orphan file: {relative_path}")
                        except Exception as e:
                            report["errors"].append(
                                f"Failed to delete {relative_path}: {str(e)}"
                            )

        except Exception as e:
            report["errors"].append(f"Error scanning uploads: {str(e)}")
            logger.error(f"Error removing orphan files: {e}")

        report["dry_run"] = dry_run
        return report

    # ========================================================================
    # 3. DELETE BROKEN INDEXES
    # ========================================================================

    def delete_broken_indexes(
        self, dry_run: bool = True, health_status: str = "unhealthy"
    ) -> Dict[str, Any]:
        """
        Find and delete broken/unhealthy indexes and associated data.

        Args:
            dry_run: If True, only report what would be deleted
            health_status: 'unhealthy', 'warning', or 'unhealthy_only'

        Returns:
            Report of deleted indexes
        """
        report: Dict[str, Any] = {
            "total_indexes": 0,
            "broken_found": 0,
            "deleted": 0,
            "total_freed_bytes": 0,
            "deleted_indexes": [],
            "errors": [],
        }

        try:
            db = self._get_db()
            cursor = db.cursor()

            # Get all indexes
            cursor.execute("SELECT id, name, metadata, vector_table_name FROM indexes")
            all_indexes = cursor.fetchall()
            report["total_indexes"] = len(all_indexes)

            for index_row in all_indexes:
                index_id = index_row["id"]
                index_name = index_row["name"]
                metadata = json.loads(index_row["metadata"] or "{}")
                metadata_status = metadata.get("status")

                # Determine if broken
                is_broken = False
                if health_status == "unhealthy_only":
                    is_broken = metadata_status in {"failed", "empty"}
                else:
                    is_broken = metadata_status in {"failed", "empty", "incomplete"}

                if not is_broken:
                    continue

                report["broken_found"] += 1

                # Calculate size
                freed_bytes = self._estimate_index_size(index_id)

                if dry_run:
                    report["deleted_indexes"].append(
                        {
                            "index_id": index_id,
                            "name": index_name,
                            "status": metadata_status,
                            "size_bytes": freed_bytes,
                            "size_str": self._format_bytes(freed_bytes),
                        }
                    )
                else:
                    try:
                        self._delete_index_completely(
                            index_id,
                            index_row["vector_table_name"],
                            cursor,
                            db,
                        )
                        report["deleted"] += 1
                        report["total_freed_bytes"] += freed_bytes
                        report["deleted_indexes"].append(
                            {
                                "index_id": index_id,
                                "name": index_name,
                                "status": "deleted",
                            }
                        )
                        logger.info(f"Deleted broken index {index_id}: {index_name}")
                    except Exception as e:
                        report["errors"].append(
                            f"Failed to delete index {index_id}: {str(e)}"
                        )
                        db.rollback()

        except Exception as e:
            report["errors"].append(f"Error scanning indexes: {str(e)}")
            logger.error(f"Error deleting broken indexes: {e}")

        report["dry_run"] = dry_run
        return report

    # ========================================================================
    # 4. REBUILD FAILED FILES ONLY
    # ========================================================================

    def get_failed_files_for_index(self, index_id: str) -> Dict[str, Any]:
        """
        Get list of files that failed in the latest build job.

        Args:
            index_id: ID of the index to check

        Returns:
            List of failed files with paths and errors
        """
        report: Dict[str, Any] = {
            "index_id": index_id,
            "failed_files": [],
            "total_failed": 0,
            "error": None,
        }

        try:
            db = self._get_db()
            cursor = db.cursor()

            # Get latest job
            cursor.execute(
                """
                SELECT id FROM index_jobs
                WHERE index_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (index_id,),
            )

            latest_job = cursor.fetchone()
            if not latest_job:
                report["error"] = "No build job found for this index"
                return report

            job_id = latest_job["id"]

            # Get failed files
            cursor.execute(
                """
                SELECT stored_path, filename, error, chunks_generated
                FROM index_job_files
                WHERE job_id = ? AND status = 'failed'
                ORDER BY filename ASC
            """,
                (job_id,),
            )

            failed_rows = cursor.fetchall()
            report["total_failed"] = len(failed_rows)

            for row in failed_rows:
                report["failed_files"].append(
                    {
                        "path": row["stored_path"],
                        "filename": row["filename"],
                        "error": row["error"],
                        "chunks_attempted": row["chunks_generated"],
                    }
                )

        except Exception as e:
            report["error"] = str(e)
            logger.error(f"Error getting failed files: {e}")

        return report

    def rebuild_failed_files_only(
        self, index_id: str, force: bool = False
    ) -> Dict[str, Any]:
        """
        Mark failed files to be rebuilt on next indexing job.

        Args:
            index_id: ID of the index
            force: If True, force rebuild even if not actually failed

        Returns:
            Report of prepared rebuild
        """
        report: Dict[str, Any] = {
            "index_id": index_id,
            "files_prepared": 0,
            "job_id": None,
            "error": None,
        }

        try:
            db = self._get_db()
            cursor = db.cursor()

            # Get latest job to find failed files
            cursor.execute(
                """
                SELECT id FROM index_jobs
                WHERE index_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (index_id,),
            )

            latest_job = cursor.fetchone()
            if not latest_job:
                report["error"] = "No build job found"
                return report

            job_id = latest_job["id"]

            if not force:
                # Get failed files
                cursor.execute(
                    """
                    SELECT COUNT(*) as count FROM index_job_files
                    WHERE job_id = ? AND status = 'failed'
                """,
                    (job_id,),
                )
                count = cursor.fetchone()["count"]

                if count == 0:
                    report["error"] = (
                        "No failed files found. Use force=True to rebuild all"
                    )
                    return report

            # Reset only failed/pending files unless the caller explicitly
            # requests a full rebuild.
            if force:
                cursor.execute(
                    """
                    UPDATE index_job_files
                    SET status = 'pending', error = NULL
                    WHERE job_id = ?
                """,
                    (job_id,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE index_job_files
                    SET status = 'pending', error = NULL
                    WHERE job_id = ? AND status IN ('failed', 'pending')
                """,
                    (job_id,),
                )

            report["files_prepared"] = cursor.rowcount

            # Preparation does not launch a worker. Leave the job paused so
            # the normal resume endpoint can start it without presenting a
            # phantom in-progress build.
            cursor.execute(
                """
                UPDATE index_jobs
                SET status = 'paused', error = NULL, updated_at = ?
                WHERE id = ?
            """,
                (datetime.utcnow().isoformat(), job_id),
            )

            db.commit()
            report["job_id"] = job_id

        except Exception as e:
            report["error"] = str(e)
            db.rollback()
            logger.error(f"Error preparing failed files rebuild: {e}")

        return report

    # ========================================================================
    # 5. LIST INDEX HEALTH (detailed diagnostic)
    # ========================================================================

    def get_index_health_report(self, index_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed health report for one or all indexes.

        Args:
            index_id: Specific index, or None for all

        Returns:
            Detailed health diagnostics
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "indexes": [],
            "summary": {"total": 0, "healthy": 0, "warning": 0, "unhealthy": 0},
        }

        try:
            db = self._get_db()
            cursor = db.cursor()
            vector_db = None
            vector_table_names = None
            if self.lancedb_path.exists():
                try:
                    import lancedb

                    vector_db = lancedb.connect(  # type: ignore[attr-defined]
                        str(self.lancedb_path)
                    )
                    vector_table_names = set(self._lancedb_table_names(vector_db))
                except Exception as exc:
                    report["vector_store_error"] = str(exc)

            if index_id:
                cursor.execute(
                    "SELECT id, name, metadata, vector_table_name FROM indexes WHERE id = ?",
                    (index_id,),
                )
            else:
                cursor.execute(
                    "SELECT id, name, metadata, vector_table_name FROM indexes ORDER BY created_at DESC"
                )

            indexes = cursor.fetchall()

            for idx_row in indexes:
                idx_id = idx_row["id"]
                metadata = json.loads(idx_row["metadata"] or "{}")

                # Get vector table info
                cursor.execute(
                    """
                    SELECT COUNT(*) as doc_count FROM index_documents WHERE index_id = ?
                """,
                    (idx_id,),
                )
                doc_count = cursor.fetchone()["doc_count"]

                # Get latest job status
                cursor.execute(
                    """
                    SELECT status, error, created_at FROM index_jobs
                    WHERE index_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (idx_id,),
                )
                latest_job = cursor.fetchone()

                table_name = idx_row["vector_table_name"]
                table_exists = None
                vector_rows = None
                if table_name and vector_table_names is not None:
                    table_exists = table_name in vector_table_names
                    if table_exists and vector_db is not None:
                        try:
                            vector_rows = vector_db.open_table(table_name).count_rows()
                        except Exception:
                            table_exists = False

                # Determine health
                metadata_status = metadata.get("status")
                health = (
                    "unhealthy"
                    if metadata_status in {"failed", "empty"}
                    else (
                        "warning"
                        if metadata_status in {"incomplete", "building"}
                        else "healthy"
                    )
                )
                latest_job_status = latest_job["status"] if latest_job else None
                if latest_job_status == "failed":
                    health = "unhealthy"
                elif (
                    latest_job_status in {"building", "queued", "paused"}
                    and health == "healthy"
                ):
                    health = "warning"
                if table_exists is False or vector_rows == 0:
                    health = "unhealthy"

                health_data = {
                    "index_id": idx_id,
                    "name": idx_row["name"],
                    "health": health,
                    "status": metadata_status,
                    "documents": doc_count,
                    "latest_job": {
                        "status": latest_job_status,
                        "error": latest_job["error"] if latest_job else None,
                        "created_at": latest_job["created_at"] if latest_job else None,
                    },
                    "metadata": {
                        "created_at": metadata.get("created_at"),
                        "chunk_size": metadata.get("chunk_size"),
                        "embedding_model": metadata.get("embedding_model"),
                        "enable_enrich": metadata.get("enable_enrich"),
                        "vector_table": table_name,
                    },
                    "vector_store": {
                        "table": table_name,
                        "exists": table_exists,
                        "rows": vector_rows,
                    },
                }

                report["indexes"].append(health_data)
                report["summary"]["total"] += 1
                report["summary"][health] += 1

        except Exception as e:
            report["error"] = str(e)
            logger.error(f"Error getting index health: {e}")

        return report

    # ========================================================================
    # 6. EXPORT DIAGNOSTICS BUNDLE
    # ========================================================================

    def export_diagnostics_bundle(
        self,
        output_path: Optional[str] = None,
        include_logs: bool = True,
        include_config: bool = True,
    ) -> Dict[str, Any]:
        """
        Export complete diagnostics bundle (logs, configs, state).

        Args:
            output_path: Path to save bundle, or None for auto-generated
            include_logs: Include log files
            include_config: Include configuration

        Returns:
            Report of exported bundle
        """
        report: Dict[str, Any] = {
            "bundle_path": None,
            "files_included": 0,
            "total_size": 0,
            "sections": {"indexes": 0, "jobs": 0, "logs": 0, "config": 0},
            "errors": [],
        }

        try:
            # Create bundle directory
            if output_path is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_path = str(self.project_root / f"diagnostics_bundle_{timestamp}")

            bundle_dir = Path(output_path)
            bundle_dir.mkdir(parents=True, exist_ok=True)

            db = self._get_db()
            cursor = db.cursor()

            # 1. Export index diagnostics
            try:
                cursor.execute("SELECT id, name, metadata FROM indexes")
                indexes = cursor.fetchall()
                indexes_data = [dict(idx) for idx in indexes]

                with open(bundle_dir / "indexes.json", "w") as f:
                    json.dump(indexes_data, f, indent=2)
                report["sections"]["indexes"] = len(indexes_data)
                report["files_included"] += 1
            except Exception as e:
                report["errors"].append(f"Failed to export indexes: {str(e)}")

            # 2. Export job history
            try:
                cursor.execute("""
                    SELECT id, index_id, status, stage, progress, message,
                           created_at, updated_at
                    FROM index_jobs
                    ORDER BY created_at DESC
                """)
                jobs = cursor.fetchall()
                jobs_data = [dict(job) for job in jobs]

                with open(bundle_dir / "jobs.json", "w") as f:
                    json.dump(jobs_data, f, indent=2)
                report["sections"]["jobs"] = len(jobs_data)
                report["files_included"] += 1
            except Exception as e:
                report["errors"].append(f"Failed to export jobs: {str(e)}")

            # 3. Export logs
            if include_logs:
                try:
                    logs_dir = bundle_dir / "logs"
                    logs_dir.mkdir(exist_ok=True)

                    src_logs = self.project_root / "logs"
                    if src_logs.exists():
                        for log_file in src_logs.glob("*.log"):
                            try:
                                shutil.copy2(log_file, logs_dir / log_file.name)
                                report["sections"]["logs"] += 1
                                report["files_included"] += 1
                            except Exception as e:
                                report["errors"].append(
                                    f"Failed to copy log {log_file}: {str(e)}"
                                )
                except Exception as e:
                    report["errors"].append(f"Failed to export logs: {str(e)}")

            # 4. Export configuration
            if include_config:
                try:
                    config_files = [
                        "pyproject.toml",
                    ]

                    for config_file in config_files:
                        config_path = self.project_root / config_file
                        if config_path.exists():
                            try:
                                shutil.copy2(config_path, bundle_dir / config_file)
                                report["sections"]["config"] += 1
                                report["files_included"] += 1
                            except Exception:
                                pass  # Skip if permission denied
                except Exception as e:
                    report["errors"].append(f"Failed to export config: {str(e)}")

            # 5. Create manifest
            try:
                manifest = {
                    "created_at": datetime.utcnow().isoformat(),
                    "system": {
                        "python_version": os.sys.version,  # type: ignore[attr-defined]
                        "platform": os.sys.platform,  # type: ignore[attr-defined]
                    },
                    "contents": report["sections"],
                    "errors": report["errors"],
                }
                with open(bundle_dir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)
                report["files_included"] += 1
            except Exception as e:
                report["errors"].append(f"Failed to create manifest: {str(e)}")

            # Calculate total size
            for file_path in bundle_dir.glob("**/*"):
                if file_path.is_file():
                    report["total_size"] += file_path.stat().st_size

            report["bundle_path"] = str(bundle_dir)
            logger.info(f"Diagnostics bundle exported to {bundle_dir}")

        except Exception as e:
            report["errors"].append(f"Failed to create bundle: {str(e)}")
            logger.error(f"Error exporting diagnostics: {e}")

        return report

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _estimate_index_size(self, index_id: str) -> int:
        """Estimate total size of index data"""
        total_size = 0

        try:
            # LanceDB table size
            table_pattern = f"text_pages_{index_id}*"
            if self.lancedb_path.exists():
                for table_dir in self.lancedb_path.glob(table_pattern):
                    if table_dir.is_dir():
                        total_size += self._dir_size(table_dir)

            # Overview file
            overview_file = self.index_store_path / "overviews" / f"{index_id}.jsonl"
            if overview_file.exists():
                total_size += overview_file.stat().st_size

            # Cache files
            cache_dir = self.index_store_path / "chunk_cache"
            if cache_dir.exists():
                for cache_file in cache_dir.glob(f"*{index_id}*"):
                    total_size += cache_file.stat().st_size

        except Exception as e:
            logger.warning(f"Could not estimate size for {index_id}: {e}")

        return total_size

    def _delete_index_completely(
        self,
        index_id: str,
        vector_table_name: Optional[str],
        cursor,
        db,
    ) -> None:
        """Delete generated artifacts and then remove the index records."""
        document_rows = cursor.execute(
            "SELECT stored_path FROM index_documents WHERE index_id = ?",
            (index_id,),
        ).fetchall()

        # Drop the exact configured table and its late-chunk companion.
        if self.lancedb_path.exists():
            import lancedb

            conn = lancedb.connect(str(self.lancedb_path))  # type: ignore[attr-defined]
            table_names = set(self._lancedb_table_names(conn))

            base_table = vector_table_name or f"text_pages_{index_id}"
            for table_name in (base_table, f"{base_table}_lc"):
                if table_name in table_names:
                    conn.drop_table(table_name)

        # Delete only files owned by the upload directory and not referenced
        # by another index.
        for row in document_rows:
            stored_path = row["stored_path"]
            if not stored_path:
                continue
            file_path = self._canonical_project_path(stored_path)
            try:
                file_path.relative_to(self.uploads_path.resolve(strict=False))
            except ValueError:
                continue
            other_rows = cursor.execute(
                """
                SELECT stored_path
                FROM index_documents
                WHERE index_id != ?
                """,
                (index_id,),
            ).fetchall()
            is_shared = any(
                row["stored_path"]
                and self._canonical_project_path(row["stored_path"]) == file_path
                for row in other_rows
            )
            if not is_shared and file_path.is_file():
                file_path.unlink()

        overview_file = self.index_store_path / "overviews" / f"{index_id}.jsonl"
        overview_file.unlink(missing_ok=True)

        cache_dir = self.index_store_path / "chunk_cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob(f"*{index_id}*"):
                if cache_file.is_file():
                    cache_file.unlink()

        # Delete from sessions_indexes
        cursor.execute("DELETE FROM session_indexes WHERE index_id = ?", (index_id,))

        # Delete job files
        cursor.execute("DELETE FROM index_job_files WHERE index_id = ?", (index_id,))

        # Delete jobs
        cursor.execute("DELETE FROM index_jobs WHERE index_id = ?", (index_id,))

        # Delete documents
        cursor.execute("DELETE FROM index_documents WHERE index_id = ?", (index_id,))

        # Delete index
        cursor.execute("DELETE FROM indexes WHERE id = ?", (index_id,))

        db.commit()

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Format bytes to human readable"""
        value: float = size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"

    @staticmethod
    def _dir_size(path: Path) -> int:
        """Calculate directory size"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    # ========================================================================
    # 7. SQLITE VACUUM
    # ========================================================================

    def vacuum_database(self) -> Dict[str, Any]:
        """Run SQLite VACUUM if fragmentation exceeds 10%, free unused pages."""
        db = self._get_db()
        if db is None:
            return {"error": "Database connection unavailable"}
        try:
            cursor = db.cursor()
            freelist = cursor.execute("PRAGMA freelist_count").fetchone()[0]
            page_count = cursor.execute("PRAGMA page_count").fetchone()[0]
            fragmentation_pct = (
                round((freelist / page_count) * 100, 1) if page_count else 0
            )
            page_size = cursor.execute("PRAGMA page_size").fetchone()[0]
            freed_kb = round((freelist * page_size) / 1024, 1)

            if fragmentation_pct >= 10:
                cursor.execute("VACUUM")
                cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.commit()
                vacuumed = True
                logger.info(
                    f"SQLite VACUUM completed; freed ~{freed_kb} KB ({fragmentation_pct}% fragmentation)"
                )
            else:
                vacuumed = False
                logger.info(
                    f"SQLite fragmentation {fragmentation_pct}% < 10%; skipping VACUUM"
                )

            return {
                "fragmentation_pct": fragmentation_pct,
                "freed_kb": freed_kb,
                "vacuumed": vacuumed,
            }
        except Exception as e:
            logger.error(f"VACUUM failed: {e}")
            return {"error": str(e)}

    # ========================================================================
    # 8. LANCEDB ORPHAN TABLE SWEEP
    # ========================================================================

    def remove_orphan_lancedb_tables(self, dry_run: bool = True) -> Dict[str, Any]:
        """Drop LanceDB tables that have no matching index record in SQLite.

        Args:
            dry_run: When True (default), only report; do not drop anything.
        """
        result: Dict[str, Any] = {
            "orphans": [],
            "dropped": [],
            "dry_run": dry_run,
            "error": None,
        }

        # Gather known index IDs from database
        db = self._get_db()
        if db is None:
            result["error"] = "Database connection unavailable"
            return result
        try:
            rows = db.execute("SELECT id FROM indexes").fetchall()
            known_ids = {row["id"] for row in rows}
        except Exception as e:
            result["error"] = f"Could not query indexes table: {e}"
            return result

        # Gather LanceDB table names
        try:
            import lancedb
        except ImportError:
            result["error"] = "lancedb package not available"
            return result

        try:
            conn = lancedb.connect(str(self.lancedb_path))  # type: ignore[attr-defined]
            all_tables = self._lancedb_table_names(conn)
        except Exception as e:
            result["error"] = (
                f"Could not connect to LanceDB at {self.lancedb_path}: {e}"
            )
            return result

        # Identify orphans: text_pages_{id} tables whose id isn't in known_ids
        prefix = "text_pages_"
        for table_name in all_tables:
            if not table_name.startswith(prefix):
                continue
            index_id = table_name[len(prefix) :]
            # Strip lc suffix if present
            index_id = index_id.removesuffix("_lc").removesuffix("_lc_v3")
            if index_id not in known_ids:
                result["orphans"].append(table_name)
                if not dry_run:
                    try:
                        conn.drop_table(table_name)
                        result["dropped"].append(table_name)
                        logger.info(f"Dropped orphan LanceDB table: {table_name}")
                    except Exception as drop_err:
                        logger.warning(
                            f"Failed to drop orphan table {table_name}: {drop_err}"
                        )

        return result
