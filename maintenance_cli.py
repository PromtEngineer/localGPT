#!/usr/bin/env python3
"""
LocalGPT Maintenance CLI Tool

Usage:
    python maintenance_cli.py repair-stuck-builds [--older-than MINUTES]
    python maintenance_cli.py remove-orphan-files [--dry-run|--execute]
    python maintenance_cli.py delete-broken-indexes [--dry-run|--execute]
    python maintenance_cli.py get-failed-files INDEX_ID
    python maintenance_cli.py rebuild-failed-files INDEX_ID [--force]
    python maintenance_cli.py list-health [INDEX_ID]
    python maintenance_cli.py export-diagnostics [--output PATH]
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.maintenance import MaintenanceTools


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_report(report: dict, title: str = "Report"):
    """Pretty print a report"""
    print_section(title)
    print(json.dumps(report, indent=2, default=str))


def cmd_repair_stuck_builds(args):
    """Repair stuck builds"""
    tools = MaintenanceTools()
    report = tools.repair_stuck_builds(args.older_than)
    
    print_report(report, "Repair Stuck Builds")
    
    if report["repaired"] > 0:
        print(f"\n✅ Successfully recovered {report['repaired']} stuck job(s)")
    else:
        print(f"\n✓ No stuck jobs found")
    
    if report["errors"]:
        print(f"\n⚠️ Errors encountered:")
        for error in report["errors"]:
            print(f"  - {error}")


def cmd_remove_orphan_files(args):
    """Remove orphan uploaded files"""
    tools = MaintenanceTools()
    
    is_dry_run = args.mode == "dry_run"
    report = tools.remove_orphan_files(dry_run=is_dry_run)
    
    title = "Remove Orphan Files (DRY RUN)" if is_dry_run else "Remove Orphan Files"
    print_report(report, title)
    
    if report["orphans_found"] == 0:
        print(f"\n✓ No orphan files found")
    else:
        print(f"\nFound {report['orphans_found']} orphan file(s)")
        if report["orphan_files"]:
            print("\nFiles:")
            for file_info in report["orphan_files"]:
                print(f"  - {file_info['path']} ({file_info['size_str']})")
        
        if not is_dry_run and report["orphans_removed"] > 0:
            freed = report["total_freed_bytes"]
            freed_mb = freed / (1024 * 1024)
            print(f"\n✅ Removed {report['orphans_removed']} file(s), freed {freed_mb:.1f} MB")


def cmd_delete_broken_indexes(args):
    """Delete broken indexes"""
    tools = MaintenanceTools()
    
    is_dry_run = args.mode == "dry_run"
    report = tools.delete_broken_indexes(dry_run=is_dry_run, health_status=args.health_status)
    
    title = "Delete Broken Indexes (DRY RUN)" if is_dry_run else "Delete Broken Indexes"
    print_report(report, title)
    
    if report["broken_found"] == 0:
        print(f"\n✓ No broken indexes found")
    else:
        print(f"\nFound {report['broken_found']} broken index(es)")
        for idx in report["deleted_indexes"]:
            size_str = idx.get("size_str", "N/A")
            print(f"  - {idx['name']} ({idx['index_id']}) - {size_str}")
        
        if not is_dry_run and report["deleted"] > 0:
            freed = report["total_freed_bytes"]
            freed_mb = freed / (1024 * 1024)
            print(f"\n✅ Deleted {report['deleted']} index(es), freed {freed_mb:.1f} MB")


def cmd_get_failed_files(args):
    """Get files that failed in the latest build"""
    tools = MaintenanceTools()
    report = tools.get_failed_files_for_index(args.index_id)
    
    if report.get("error"):
        print(f"\n❌ Error: {report['error']}")
        return
    
    print_section(f"Failed Files for Index: {args.index_id}")
    
    if report["total_failed"] == 0:
        print("✓ No failed files found")
    else:
        print(f"Found {report['total_failed']} failed file(s):\n")
        for file_info in report["failed_files"]:
            print(f"  📄 {file_info['filename']}")
            print(f"     Path: {file_info['path']}")
            print(f"     Error: {file_info['error']}")
            print()


def cmd_rebuild_failed_files(args):
    """Rebuild failed files"""
    tools = MaintenanceTools()
    report = tools.rebuild_failed_files_only(args.index_id, args.force)
    
    if report.get("error"):
        print(f"\n❌ Error: {report['error']}")
        return
    
    print_section(f"Rebuild Failed Files: {args.index_id}")
    print(f"✅ Prepared {report['files_prepared']} file(s) for rebuild")
    print(f"Job ID: {report['job_id']}")
    print("\nThese files will be reprocessed on the next indexing job.")


def cmd_list_health(args):
    """List index health"""
    tools = MaintenanceTools()
    report = tools.get_index_health_report(args.index_id)
    
    print_section("Index Health Report")
    
    summary = report.get("summary", {})
    print(f"Summary:")
    print(f"  Total: {summary.get('total', 0)}")
    print(f"  Healthy: {summary.get('healthy', 0)}")
    print(f"  Warning: {summary.get('warning', 0)}")
    print(f"  Unhealthy: {summary.get('unhealthy', 0)}")
    
    indexes = report.get("indexes", [])
    if not indexes:
        print("\nNo indexes found")
        return
    
    print(f"\nIndexes ({len(indexes)}):")
    for idx in indexes:
        health = idx["health"]
        health_icon = "✅" if health == "healthy" else "⚠️" if health == "warning" else "❌"
        print(f"\n  {health_icon} {idx['name']} ({idx['index_id']})")
        print(f"      Status: {idx['status']}")
        print(f"      Documents: {idx['documents']}")
        if idx["latest_job"]:
            print(f"      Latest Job: {idx['latest_job']['status']}")


def cmd_export_diagnostics(args):
    """Export diagnostics bundle"""
    tools = MaintenanceTools()
    report = tools.export_diagnostics_bundle(
        output_path=args.output,
        include_logs=not args.no_logs,
        include_config=not args.no_config
    )
    
    print_section("Export Diagnostics Bundle")
    
    if report.get("bundle_path"):
        print(f"✅ Bundle created successfully")
        print(f"   Path: {report['bundle_path']}")
        print(f"   Files: {report['files_included']}")
        print(f"   Size: {report['total_size'] / (1024*1024):.1f} MB")
        print(f"\n   Contents:")
        for section, count in report.get("sections", {}).items():
            if count > 0:
                print(f"     - {section}: {count}")
    else:
        print(f"❌ Failed to create bundle")
    
    if report.get("errors"):
        print(f"\n⚠️ Errors encountered:")
        for error in report["errors"]:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="LocalGPT Maintenance Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Maintenance command")
    
    # repair-stuck-builds
    sp = subparsers.add_parser("repair-stuck-builds", help="Repair stuck build jobs")
    sp.add_argument("--older-than", type=int, default=30, help="Minutes (default: 30)")
    sp.set_defaults(func=cmd_repair_stuck_builds)
    
    # remove-orphan-files
    sp = subparsers.add_parser("remove-orphan-files", help="Remove orphan uploaded files")
    sp.add_argument("--mode", default="dry_run", choices=["dry_run", "execute"],
                    help="Run mode (default: dry_run)")
    sp.set_defaults(func=cmd_remove_orphan_files)
    
    # delete-broken-indexes
    sp = subparsers.add_parser("delete-broken-indexes", help="Delete broken indexes")
    sp.add_argument("--mode", default="dry_run", choices=["dry_run", "execute"],
                    help="Run mode (default: dry_run)")
    sp.add_argument("--health-status", default="unhealthy",
                    choices=["unhealthy", "warning", "unhealthy_only"],
                    help="Health status filter")
    sp.set_defaults(func=cmd_delete_broken_indexes)
    
    # get-failed-files
    sp = subparsers.add_parser("get-failed-files", help="Get files that failed in latest build")
    sp.add_argument("index_id", help="Index ID")
    sp.set_defaults(func=cmd_get_failed_files)
    
    # rebuild-failed-files
    sp = subparsers.add_parser("rebuild-failed-files", help="Rebuild failed files")
    sp.add_argument("index_id", help="Index ID")
    sp.add_argument("--force", action="store_true", help="Force rebuild even if no failures")
    sp.set_defaults(func=cmd_rebuild_failed_files)
    
    # list-health
    sp = subparsers.add_parser("list-health", help="List index health")
    sp.add_argument("index_id", nargs="?", help="Specific index (optional)")
    sp.set_defaults(func=cmd_list_health)
    
    # export-diagnostics
    sp = subparsers.add_parser("export-diagnostics", help="Export diagnostics bundle")
    sp.add_argument("--output", help="Output path")
    sp.add_argument("--no-logs", action="store_true", help="Skip log files")
    sp.add_argument("--no-config", action="store_true", help="Skip config files")
    sp.set_defaults(func=cmd_export_diagnostics)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
