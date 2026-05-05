#!/usr/bin/env python3
"""
Test script for Incremental Indexing functionality
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
from rag_system.utils.incremental_indexer import IncrementalIndexer

def create_test_files(test_dir: str) -> list:
    """Create test files for incremental indexing"""
    files = []

    # Create test file 1
    file1 = os.path.join(test_dir, "test1.txt")
    with open(file1, 'w') as f:
        f.write("This is test file 1 content.")
    files.append(file1)

    # Create test file 2
    file2 = os.path.join(test_dir, "test2.txt")
    with open(file2, 'w') as f:
        f.write("This is test file 2 content.")
    files.append(file2)

    return files

def test_incremental_indexer():
    """Test the incremental indexer functionality"""
    print("🧪 Testing Incremental Indexer...")

    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test.db")

    try:
        # Initialize indexer
        indexer = IncrementalIndexer(db_path, test_dir)

        # Create test files
        test_files = create_test_files(test_dir)

        # Test 1: New files should be detected as changed
        print("\n📝 Test 1: Detecting new files")
        changes = indexer.detect_changes(test_files, "test_index")
        assert all(changes[f][0] for f in test_files), "New files should be detected as changed"
        print("✅ New files correctly detected as changed")

        # Test 2: Index the files
        print("\n📝 Test 2: Indexing files")
        for i, file_path in enumerate(test_files):
            indexer.update_document_metadata(file_path, "test_index", 10 + i, "index")

        # Test 3: Unchanged files should not be detected as changed
        print("\n📝 Test 3: Detecting unchanged files")
        changes = indexer.detect_changes(test_files, "test_index")
        assert all(not changes[f][0] for f in test_files), "Unchanged files should not be detected as changed"
        print("✅ Unchanged files correctly detected as unchanged")

        # Test 4: Modified file should be detected as changed
        print("\n📝 Test 4: Detecting modified files")
        time.sleep(1)  # Ensure modification time changes
        with open(test_files[0], 'w') as f:
            f.write("This is modified test file 1 content.")

        changes = indexer.detect_changes(test_files, "test_index")
        assert changes[test_files[0]][0], "Modified file should be detected as changed"
        assert not changes[test_files[1]][0], "Unmodified file should not be detected as changed"
        print("✅ Modified files correctly detected")

        # Test 5: Incremental file list
        print("\n📝 Test 5: Incremental file list generation")
        files_to_index, unchanged_files = indexer.get_incremental_file_list(test_files, "test_index")
        assert test_files[0] in files_to_index, "Modified file should be in files_to_index"
        assert test_files[1] in unchanged_files, "Unmodified file should be in unchanged_files"
        print("✅ Incremental file list generation works")

        # Test 6: Index stats
        print("\n📝 Test 6: Index statistics")
        stats = indexer.get_index_stats("test_index")
        assert stats['total_documents'] == 2, "Should have 2 documents"
        assert stats['total_chunks'] == 21, "Should have 21 total chunks"
        print("✅ Index statistics work")

        # Test 7: Same files in a different index should be treated as new
        print("\n📝 Test 7: Index-scoped change detection")
        changes = indexer.detect_changes(test_files, "other_index")
        assert all(changes[f][0] for f in test_files), "Files should be new in another index"
        print("✅ Index-scoped metadata works")

        print("\n🎉 All incremental indexer tests passed!")

    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_force_reindex():
    """Test force reindex functionality"""
    print("🧪 Testing Force Reindex...")

    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test.db")

    try:
        indexer = IncrementalIndexer(db_path, test_dir)
        test_files = create_test_files(test_dir)

        # Index files initially
        for file_path in test_files:
            indexer.update_document_metadata(file_path, "test_index", 5, "index")

        # Force reindex should return all files
        files_to_index, unchanged_files = indexer.get_incremental_file_list(test_files, "test_index", force_reindex=True)
        assert len(files_to_index) == len(test_files), "Force reindex should return all files"
        assert len(unchanged_files) == 0, "Force reindex should have no unchanged files"
        print("✅ Force reindex works")

    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_incremental_indexer()
    test_force_reindex()
    print("🎉 All incremental indexing tests passed!")
