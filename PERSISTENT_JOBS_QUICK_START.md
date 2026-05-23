# 🚀 Persistent Indexing Jobs - Quick Start

**Status**: ✅ Pipeline Integrated (UI progress display pending)

## What's New?

Jobs no longer disappear when the backend crashes. All progress is saved to the database with per-stage tracking.

## Try It Now

### 1. Check Job Progress
```bash
# See real-time file-by-file status
curl http://localhost:8000/index-jobs/my-job-123/file-status

# Response shows:
# - Which files completed
# - Which failed (with error)
# - What stage each is on
```

### 2. Resume After Crash
```bash
# Backend automatically recovers stale jobs on startup
# Then resume manually via API:
curl -X POST http://localhost:8000/index-jobs/my-job-123/resume

# Job is queued again and the pipeline skips completed files
# Completed conversion/chunking work is reused when the chunk cache is available
```

### 3. View Performance Report
```bash
# Which stages take longest?
curl http://localhost:8000/index-jobs/my-job-123/statistics

# Response shows:
# - Average time per stage
# - Which files retried and how many times
# - Bottleneck identification
```

### 4. Get Complete Timeline
```bash
# Every stage, every file, every event
curl http://localhost:8000/index-jobs/my-job-123/timeline

# Response shows:
# - When each file started/finished
# - Each stage's duration
# - Any errors and when they occurred
```

### 5. Export Audit Trail
```bash
# For debugging and compliance
curl http://localhost:8000/index-jobs/my-job-123/audit-trail > trail.json

# JSON with all events:
# - filename, stage_name, status, timestamps, error
```

## Auto-Recovery on Startup

No action needed - it's automatic:

```
Backend crashes during indexing
↓
You restart the backend
↓
Startup hook auto-detects stale jobs
↓
Marks them as 'paused' (not failed!)
↓
You can resume them via the API
```

## Current Status

| Component | Status |
|-----------|--------|
| Database schema | ✅ Complete |
| JobProgressTracker class | ✅ Complete |
| REST API endpoints | ✅ Complete |
| Auto-recovery on startup | ✅ Complete |
| Pipeline integration | ✅ Complete |
| UI progress display | ⏳ Future |

## What's Tracked

For each file:
- **Conversion** - PDF to markdown
- **Chunking** - Split into pieces
- **Overview** - Summarize (optional)
- **Enrichment** - Add context (optional)
- **Embedding** - Generate vectors
- **Storage** - Write to database

Each stage has:
- ✅/❌ Status
- Start/end time
- Duration
- Error (if failed)
- Hash of output (for dedup)

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/index-jobs/{id}/resume` | POST | Resume paused job |
| `/index-jobs/{id}/timeline` | GET | View all events |
| `/index-jobs/{id}/file-status` | GET | Per-file breakdown |
| `/index-jobs/{id}/statistics` | GET | Performance metrics |
| `/index-jobs/{id}/audit-trail` | GET | All events (CSV format) |
| `/index-jobs/recover-stale` | POST | Manual recovery |

## Example: Complete Flow

```bash
# 1. Start indexing (via UI or API)
# Progress is saved to DB

# 2. Backend crashes mid-job
# All progress is safely persisted

# 3. Restart backend
# Auto-recovery runs (you see: "✅ Auto-recovered 1 stale job")

# 4. Check what happened
curl http://localhost:8000/index-jobs/abc123/timeline
# Response shows exactly where it stopped

# 5. Resume
curl -X POST http://localhost:8000/index-jobs/abc123/resume
# Job continues from last incomplete stage

# 6. Monitor progress
curl http://localhost:8000/index-jobs/abc123/file-status
# Shows current progress file-by-file
```

## Reliability Improvements

| Scenario | Before | After |
|----------|--------|-------|
| Backend crashes | Job lost ❌ | Job resumed ✅ |
| Stuck build | Stuck forever | Detected & marked paused |
| Failed file | Unclear why | Clear error + error code |
| Performance tuning | Guesswork | Exact metrics per stage |
| Compliance audit | Log files only | Complete event timeline |

## Next Steps

### Soon (Phase 3)
- UI display of job timeline
- Resume button in interface
- Per-file error display

### Future (Phase 4)
- Automatic retry with backoff
- Parallel file processing
- Resource limits per stage
- Persist enriched chunks/embeddings so resumes can skip every completed stage exactly

## Need Help?

### Check Job Status
```bash
curl http://localhost:8000/index-jobs/{job_id}/file-status | jq .
```

### View Error Details
```bash
curl http://localhost:8000/index-jobs/{job_id}/timeline | jq '.files[] | select(.overall_status=="failed")'
```

### Export Complete History
```bash
curl http://localhost:8000/index-jobs/{job_id}/audit-trail > history.json
cat history.json | jq -r '.events[] | "\(.filename) - \(.stage_name): \(.status)"'
```

### Force Recovery (if auto-recovery doesn't run)
```bash
curl -X POST http://localhost:8000/index-jobs/recover-stale?older_than_minutes=10
```

---

**TL;DR**: Jobs are now resilient to crashes. They auto-recover on startup and can be resumed from exactly where they left off.
