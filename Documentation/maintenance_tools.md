# 🛠️ LocalGPT Maintenance Tools

A comprehensive suite of tools for maintaining and repairing the LocalGPT indexing system. These tools help you:

1. **Repair stuck builds** - Recover from failed indexing jobs
2. **Remove orphan files** - Clean up unused uploaded files
3. **Delete broken indexes** - Remove corrupted or incomplete indexes
4. **Rebuild failed files** - Reprocess only files that failed
5. **List index health** - View detailed diagnostics for all indexes
6. **Export diagnostics** - Collect logs and state for troubleshooting

## Quick Start

### Via CLI (Recommended)

```bash
# Repair stuck builds
python maintenance_cli.py repair-stuck-builds

# See orphan files (dry run)
python maintenance_cli.py remove-orphan-files --mode dry_run

# Actually remove orphan files
python maintenance_cli.py remove-orphan-files --mode execute

# Check all indexes health
python maintenance_cli.py list-health

# Export diagnostics for support
python maintenance_cli.py export-diagnostics
```

### Via REST API

```bash
# Repair stuck builds
curl -X POST http://localhost:8000/maintenance/repair-stuck-builds

# Remove orphan files (dry run)
curl -X POST http://localhost:8000/maintenance/remove-orphan-files?dry_run=true

# List index health
curl http://localhost:8000/maintenance/index-health

# Export diagnostics
curl -X POST http://localhost:8000/maintenance/export-diagnostics
```

## Detailed Commands

### 1. Repair Stuck Builds

**Problem**: Indexing jobs get stuck in "building" or "queued" status and never complete.

**Solution**: Mark stale jobs as failed so they can be retried.

```bash
# CLI - repair jobs older than 30 minutes (default)
python maintenance_cli.py repair-stuck-builds

# CLI - repair jobs older than 60 minutes
python maintenance_cli.py repair-stuck-builds --older-than 60

# API
curl -X POST http://localhost:8000/maintenance/repair-stuck-builds?older_than_minutes=30
```

**What it does**:
- Finds indexing jobs that haven't updated for N minutes
- Marks them as `failed` with diagnosis message
- Marks any pending files in those jobs as failed
- Allows jobs to be retried from the frontend

**Output**:
```json
{
  "found": 2,
  "repaired": 2,
  "stuck_jobs": [
    {
      "job_id": "abc123",
      "index_id": "idx_456",
      "status": "recovered",
      "was_stuck_since": "2025-05-08T10:30:00"
    }
  ],
  "errors": []
}
```

---

### 2. Remove Orphan Files

**Problem**: Uploaded files accumulate that aren't associated with any index (storage waste).

**Solution**: Find and delete unreferenced files.

```bash
# CLI - preview what would be deleted (safe)
python maintenance_cli.py remove-orphan-files --mode dry_run

# CLI - actually delete orphan files
python maintenance_cli.py remove-orphan-files --mode execute

# API - dry run
curl -X POST http://localhost:8000/maintenance/remove-orphan-files?dry_run=true

# API - execute
curl -X POST http://localhost:8000/maintenance/remove-orphan-files?dry_run=false
```

**What it does**:
- Scans the `shared_uploads/` directory
- Checks which files are referenced in the database
- Reports on orphan files (not referenced by any index)
- Optionally deletes them and reports freed space

**Output** (dry run):
```json
{
  "total_scanned": 15,
  "orphans_found": 3,
  "orphans_removed": 0,
  "total_freed_bytes": 0,
  "dry_run": true,
  "orphan_files": [
    {
      "path": "shared_uploads/old_file_123.pdf",
      "size": 5242880,
      "size_str": "5.0 MB",
      "modified": "2025-04-10T14:30:00"
    }
  ]
}
```

---

### 3. Delete Broken Indexes

**Problem**: Indexes become corrupted or get stuck in failed/incomplete state.

**Solution**: Remove unhealthy indexes and reclaim space.

```bash
# CLI - preview broken indexes
python maintenance_cli.py delete-broken-indexes --mode dry_run

# CLI - delete only unhealthy indexes (failed/empty)
python maintenance_cli.py delete-broken-indexes --mode execute --health-status unhealthy_only

# CLI - delete unhealthy + warning indexes
python maintenance_cli.py delete-broken-indexes --mode execute --health-status unhealthy

# API
curl -X POST http://localhost:8000/maintenance/delete-broken-indexes?dry_run=true&health_status=unhealthy
```

**What it does**:
- Finds indexes with `status` = failed, empty, or incomplete
- Estimates and reports the size of each index
- Optionally deletes:
  - Vector table from LanceDB
  - Overview files
  - Document records
  - Job history
  - Cached chunks
  - Database records
- Reports freed space

**Output** (dry run):
```json
{
  "total_indexes": 10,
  "broken_found": 2,
  "deleted": 0,
  "dry_run": true,
  "deleted_indexes": [
    {
      "index_id": "idx_corrupted_123",
      "name": "Old Index",
      "status": "failed",
      "size_bytes": 104857600,
      "size_str": "100.0 MB"
    }
  ]
}
```

---

### 4. Rebuild Failed Files Only

**Problem**: Some files fail to index (conversion error, timeout, etc.), but you want to retry without re-processing successful files.

**Solution**: Mark failed files for retry while skipping successful ones.

```bash
# CLI - see which files failed in the latest build
python maintenance_cli.py get-failed-files idx_abc123

# CLI - prepare failed files for rebuild
python maintenance_cli.py rebuild-failed-files idx_abc123

# CLI - force rebuild all files (even if no failures)
python maintenance_cli.py rebuild-failed-files idx_abc123 --force

# API - get failed files
curl http://localhost:8000/maintenance/failed-files/idx_abc123

# API - rebuild failed files
curl -X POST http://localhost:8000/maintenance/rebuild-failed-files/idx_abc123?force=false
```

**What it does**:
- Gets the latest indexing job for the index
- Identifies files with status `failed`
- Resets them to `pending` for the next job
- Keeps successful files unchanged
- Optionally force-rebuilds all files

**Output** (get-failed-files):
```json
{
  "index_id": "idx_abc123",
  "failed_files": [
    {
      "path": "shared_uploads/problematic.pdf",
      "filename": "problematic.pdf",
      "error": "PDF conversion timeout",
      "chunks_attempted": 0
    }
  ],
  "total_failed": 1
}
```

**Output** (rebuild):
```json
{
  "index_id": "idx_abc123",
  "files_prepared": 1,
  "job_id": "job_xyz789",
  "error": null
}
```

---

### 5. List Index Health

**Problem**: Need to understand the state of all indexes at a glance.

**Solution**: View detailed diagnostics for each index.

```bash
# CLI - all indexes
python maintenance_cli.py list-health

# CLI - specific index
python maintenance_cli.py list-health idx_abc123

# API - all indexes
curl http://localhost:8000/maintenance/index-health

# API - specific index
curl http://localhost:8000/maintenance/index-health?index_id=idx_abc123
```

**What it does**:
- Queries all indexes from the database
- Checks each index's metadata status
- Gets document count
- Reports latest job status
- Provides health summary (healthy/warning/unhealthy)

**Output**:
```json
{
  "timestamp": "2025-05-08T15:30:00.123456",
  "indexes": [
    {
      "index_id": "idx_prod_2025",
      "name": "Production Index",
      "health": "healthy",
      "status": "completed",
      "documents": 145,
      "latest_job": {
        "status": "completed",
        "error": null,
        "created_at": "2025-05-08T14:00:00"
      },
      "metadata": {
        "created_at": "2025-05-01T10:00:00",
        "chunk_size": 512,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "enable_enrich": true,
        "vector_table": "text_pages_idx_prod_2025"
      }
    },
    {
      "index_id": "idx_experiment",
      "name": "Experiment Index",
      "health": "unhealthy",
      "status": "failed",
      "documents": 23,
      "latest_job": {
        "status": "failed",
        "error": "Vector table not found",
        "created_at": "2025-05-08T13:00:00"
      }
    }
  ],
  "summary": {
    "total": 2,
    "healthy": 1,
    "warning": 0,
    "unhealthy": 1
  }
}
```

---

### 6. Export Diagnostics Bundle

**Problem**: Something is wrong and you need to provide logs and state to support.

**Solution**: Export everything in one directory.

```bash
# CLI - default (includes logs and config)
python maintenance_cli.py export-diagnostics

# CLI - specific output path
python maintenance_cli.py export-diagnostics --output /tmp/diagnostics

# CLI - exclude logs
python maintenance_cli.py export-diagnostics --no-logs

# API
curl -X POST http://localhost:8000/maintenance/export-diagnostics

# API - specific path
curl -X POST "http://localhost:8000/maintenance/export-diagnostics?output_path=/tmp/diagnostics"
```

**What it does**:
- Creates a timestamped directory (e.g., `diagnostics_bundle_20250508_153045`)
- Exports:
  - `indexes.json` - all index records
  - `jobs.json` - job history
  - `logs/` - application logs
  - `pyproject.toml`, `.env`, `docker.env` - config files
  - `manifest.json` - metadata about the bundle
- Reports total size and file count

**Output**:
```json
{
  "bundle_path": "/path/to/diagnostics_bundle_20250508_153045",
  "files_included": 12,
  "total_size": 5242880,
  "sections": {
    "indexes": 10,
    "jobs": 45,
    "logs": 8,
    "config": 3
  },
  "errors": []
}
```

**Bundle structure**:
```
diagnostics_bundle_20250508_153045/
├── indexes.json          # All indexes
├── jobs.json            # All build jobs
├── manifest.json        # Bundle metadata
├── pyproject.toml
├── docker.env
└── logs/
    ├── indexing.log
    ├── retrieval.log
    └── system.log
```

---

## API Reference

### POST /maintenance/repair-stuck-builds

**Query Parameters**:
- `older_than_minutes` (int, default=30) - Minutes to consider a job stuck

**Response**: Repair report with recovered jobs

### POST /maintenance/remove-orphan-files

**Query Parameters**:
- `dry_run` (bool, default=true) - If false, actually delete files

**Response**: Report of found/deleted orphan files

### POST /maintenance/delete-broken-indexes

**Query Parameters**:
- `dry_run` (bool, default=true) - If false, actually delete indexes
- `health_status` (str, default="unhealthy") - "unhealthy", "warning", or "unhealthy_only"

**Response**: Report of found/deleted broken indexes

### GET /maintenance/failed-files/{index_id}

**Response**: List of files that failed in latest build job

### POST /maintenance/rebuild-failed-files/{index_id}

**Query Parameters**:
- `force` (bool, default=false) - Force rebuild even if no failures

**Response**: Confirmation and job ID

### GET /maintenance/index-health

**Query Parameters**:
- `index_id` (str, optional) - Specific index or null for all

**Response**: Health report for index(es)

### POST /maintenance/export-diagnostics

**Query Parameters**:
- `output_path` (str, optional) - Custom output path
- `include_logs` (bool, default=true) - Include log files
- `include_config` (bool, default=true) - Include config files

**Response**: Report of created bundle with path

---

## Common Use Cases

### Troubleshooting a Failed Index

1. Check health:
   ```bash
   python maintenance_cli.py list-health idx_problematic
   ```

2. See what failed:
   ```bash
   python maintenance_cli.py get-failed-files idx_problematic
   ```

3. Fix the underlying issue (re-upload missing files, etc.)

4. Rebuild failed files only:
   ```bash
   python maintenance_cli.py rebuild-failed-files idx_problematic
   ```

### Cleaning Up After Testing

1. See what's orphaned:
   ```bash
   python maintenance_cli.py remove-orphan-files --mode dry_run
   python maintenance_cli.py delete-broken-indexes --mode dry_run
   ```

2. Remove if it looks right:
   ```bash
   python maintenance_cli.py remove-orphan-files --mode execute
   python maintenance_cli.py delete-broken-indexes --mode execute
   ```

### Regular Maintenance Schedule

Run periodically (e.g., weekly):

```bash
# Repair any stuck jobs
python maintenance_cli.py repair-stuck-builds --older-than 120

# Remove orphan files
python maintenance_cli.py remove-orphan-files --mode execute

# Check health
python maintenance_cli.py list-health
```

### Collecting Diagnostics for Support

```bash
python maintenance_cli.py export-diagnostics --output ~/localgpt_diagnostics

# Share the directory
zip -r ~/localgpt_diagnostics.zip ~/localgpt_diagnostics
```

---

## Safety Considerations

All operations that modify data have a **dry-run mode** (enabled by default):

- ✅ **Repair stuck builds** - Always safe, just marks jobs as failed
- ✅ **List health** - Read-only, always safe
- ✅ **Get failed files** - Read-only, always safe  
- ⚠️ **Remove orphan files** - Use `--mode dry_run` first to preview
- ⚠️ **Delete broken indexes** - Use `--mode dry_run` first to preview
- ⚠️ **Rebuild failed files** - Resets job state, use with care

**Best practice**: Always run with `--mode dry_run` or `dry_run=true` first!

---

## Troubleshooting

### "Maintenance tools not available"

Ensure the RAG system modules are in the Python path:

```bash
# Works if you're in the project root
python maintenance_cli.py list-health

# Might fail if in a subdirectory - use absolute path
/path/to/localGPT/maintenance_cli.py list-health
```

### Commands timing out

Large operations (many indexes, large exports) might timeout. Either:

1. Run on smaller datasets:
   ```bash
   python maintenance_cli.py list-health idx_specific_id
   ```

2. Increase API timeout (if using REST API):
   ```bash
   curl --max-time 120 http://localhost:8000/maintenance/...
   ```

### Database locked error

The maintenance tools try not to conflict with running indexing jobs, but if you see database lock errors:

1. Wait for any running indexing jobs to complete
2. Try again

---

## Implementation Details

The maintenance tools are implemented in `rag_system/maintenance.py` and exposed via:

1. **CLI**: `maintenance_cli.py` - Direct command-line interface
2. **REST API**: Backend `/maintenance/*` endpoints
3. **Python Module**: `from rag_system.maintenance import MaintenanceTools`

For custom integrations, import and use directly:

```python
from rag_system.maintenance import MaintenanceTools

tools = MaintenanceTools()
report = tools.repair_stuck_builds()
print(f"Repaired {report['repaired']} jobs")
```

---

## Next Steps

After using these maintenance tools:

1. **Monitor indexes** - Use `list-health` periodically
2. **Implement automation** - Add to cron jobs or scheduled tasks
3. **Set up alerts** - Monitor unhealthy indexes and repair automatically
4. **Document issues** - Export diagnostics for recurring problems

For production systems, consider:

- Running `repair-stuck-builds` daily
- Running `remove-orphan-files` weekly
- Exporting diagnostics monthly for analysis
