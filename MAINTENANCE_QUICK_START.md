# 🛠️ Maintenance Tools - Quick Reference

## One-Minute Setup

All tools are ready to use immediately:

```bash
# Check current state
python maintenance_cli.py list-health

# Or use the simple wrapper
./maintain list-health
```

## Common Commands

```bash
# ✅ Daily maintenance (safe)
./maintain repair-stuck-builds --older-than 120
./maintain remove-orphan-files --dry-run    # Preview first!
./maintain remove-orphan-files --execute    # Then execute
./maintain list-health

# 🔍 Troubleshoot a failed index
./maintain get-failed-files my_index_id
./maintain rebuild-failed-files my_index_id

# 📦 Collect diagnostics for support
./maintain export-diagnostics

# 🗑️ Clean up test/broken indexes
./maintain delete-broken-indexes --dry-run     # Preview first!
./maintain delete-broken-indexes --execute     # Then execute
```

## REST API Examples

```bash
# Health check
curl http://localhost:8000/maintenance/index-health

# See what's broken
curl http://localhost:8000/maintenance/index-health?index_id=problematic_idx

# Repair stuck jobs
curl -X POST http://localhost:8000/maintenance/repair-stuck-builds

# Preview orphan files
curl -X POST http://localhost:8000/maintenance/remove-orphan-files?dry_run=true

# Actually remove them
curl -X POST http://localhost:8000/maintenance/remove-orphan-files?dry_run=false

# Export diagnostics
curl -X POST http://localhost:8000/maintenance/export-diagnostics -o diagnostics.zip
```

## Python Usage

```python
from rag_system.maintenance import MaintenanceTools

tools = MaintenanceTools()

# Get health report
report = tools.get_index_health_report()
print(f"Total indexes: {report['summary']['total']}")
print(f"Healthy: {report['summary']['healthy']}")

# Repair stuck builds
result = tools.repair_stuck_builds()
print(f"Recovered {result['repaired']} jobs")

# Get failed files
failed = tools.get_failed_files_for_index("my_index")
for file_info in failed["failed_files"]:
    print(f"Failed: {file_info['filename']} - {file_info['error']}")

# Prepare to rebuild them
result = tools.rebuild_failed_files_only("my_index")
print(f"Prepared {result['files_prepared']} files for rebuild")
```

## What Each Tool Does

| Tool | Purpose | Destructive? | Safe Default |
|------|---------|--------------|--------------|
| `repair-stuck-builds` | Recover stalled indexing jobs | No | Always safe |
| `remove-orphan-files` | Delete unused uploads | Yes | Dry-run only |
| `delete-broken-indexes` | Remove corrupted indexes | Yes | Dry-run only |
| `get-failed-files` | List failed files | No | Always safe |
| `rebuild-failed-files` | Prep failed files for retry | No | Always safe |
| `list-health` | View index diagnostics | No | Always safe |
| `export-diagnostics` | Create support bundle | No | Always safe |

## Safety Rules

1. ✅ **Always use dry-run first**: `--dry-run` or `--mode dry_run` (default)
2. ✅ **Review the report** before executing
3. ✅ **Read-only operations are always safe**: `list-health`, `get-failed-files`, `export-diagnostics`
4. ✅ **Never destructive by accident**: Dry-run is the default
5. ✅ **Check before deletion**: All operations show what they'll affect

## Real-World Examples

### "My index failed to build"

```bash
# See what went wrong
./maintain get-failed-files my_index

# Check overall health
./maintain list-health my_index

# After fixing the problem, retry
./maintain rebuild-failed-files my_index
```

### "I'm running out of storage"

```bash
# See what's taking space
./maintain list-health

# Preview what can be removed
./maintain remove-orphan-files --dry-run
./maintain delete-broken-indexes --dry-run

# Remove if it looks good
./maintain remove-orphan-files --execute
./maintain delete-broken-indexes --execute
```

### "Need to give support diagnostics"

```bash
# Collect everything
./maintain export-diagnostics

# Find the bundle
ls -lah diagnostics_bundle_*

# Share the directory
zip -r diagnostics.zip diagnostics_bundle_20250508_153045/
```

### "Weekly maintenance routine"

```bash
#!/bin/bash
# save as: weekly_maintenance.sh

echo "🛠️ Weekly maintenance..."

# Repair any stuck jobs
./maintain repair-stuck-builds --older-than 120

# Clean up orphans
./maintain remove-orphan-files --execute

# Check health
./maintain list-health > health_report_$(date +%Y%m%d).json

echo "✅ Weekly maintenance complete"
```

Then add to crontab:
```bash
crontab -e
# Add: 0 2 * * 0 /path/to/localGPT/weekly_maintenance.sh
```

## Troubleshooting

**"Command not found"**
- Make sure you're in the project root: `cd /path/to/localGPT`
- Or use full path: `python /path/to/localGPT/maintenance_cli.py list-health`

**"Database locked"**
- Wait for any running indexing jobs to complete
- Try again

**"Maintenance tools not available"**
- Backend not started, or Python environment missing RAG modules
- Try: `source .venv/bin/activate` before starting backend

## File Locations

Key files created:

```
localGPT/
├── maintenance_cli.py           # Python CLI (can run standalone)
├── maintain                     # Bash wrapper (easiest to use)
├── rag_system/
│   └── maintenance.py           # Core implementation
├── backend/server.py            # REST API endpoints added
└── Documentation/
    └── maintenance_tools.md     # Full documentation
```

## Output Locations

Results and reports saved to:

- **Orphan files**: Scan `shared_uploads/`
- **Broken indexes**: Check `lancedb/` and `index_store/`
- **Diagnostics**: Saved as `diagnostics_bundle_YYYYMMDD_HHMMSS/`
- **Logs**: Collected from `logs/`

## Next Steps

1. **Right now**: Run `./maintain list-health` to see current state
2. **This week**: Run `./maintain repair-stuck-builds`
3. **Monthly**: Run `./maintain export-diagnostics` and review
4. **Ongoing**: Add weekly maintenance to cron job

## Get Help

- **Commands**: `./maintain help`
- **Full guide**: `Documentation/maintenance_tools.md`
- **Implementation details**: `MAINTENANCE_SUMMARY.md`

---

**Ready to use. No additional setup needed.**
