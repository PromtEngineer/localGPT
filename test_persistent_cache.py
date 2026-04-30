#!/usr/bin/env python3
"""
Test script for PersistentCache implementation
"""

import numpy as np
import os
import tempfile
import shutil
from rag_system.utils.persistent_cache import PersistentCache

def test_file_based_cache():
    """Test file-based persistent cache"""
    print("🧪 Testing file-based persistent cache...")

    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()

    try:
        # Initialize cache with session scope
        cache = PersistentCache(cache_dir=test_dir, max_size=10, semantic_threshold=0.95, cache_scope="session")

        # Test basic storage and retrieval
        query1 = "What is machine learning?"
        embedding1 = np.random.rand(384)  # Mock embedding
        result1 = {"answer": "Machine learning is a subset of AI", "sources": []}

        cache.store(query1, "rag", result1, embedding1, "session1")

        # Test exact retrieval
        retrieved = cache.retrieve(query1, "rag", embedding1, "session1")
        assert retrieved == result1, "Exact retrieval failed"
        print("✅ Exact retrieval works")

        # Test semantic similarity (similar query)
        query2 = "What is ML?"  # Similar but different wording
        embedding2 = embedding1 + np.random.rand(384) * 0.1  # Slightly different embedding
        result2 = {"answer": "ML is machine learning", "sources": []}

        cache.store(query2, "rag", result2, embedding2, "session1")

        # Should find semantic match for very similar embedding
        embedding_similar = embedding1 + np.random.rand(384) * 0.05  # Very similar
        retrieved_similar = cache.retrieve("What is machine learning exactly?", "rag", embedding_similar, "session1")
        assert retrieved_similar is not None, "Semantic retrieval failed"
        print("✅ Semantic similarity retrieval works")

        # Test session scoping
        retrieved_wrong_session = cache.retrieve(query1, "rag", embedding1, "session2")
        assert retrieved_wrong_session is None, "Session scoping failed"
        print("✅ Session scoping works")

        # Test cache stats
        stats = cache.stats()
        assert stats['backend'] == 'file', "Backend detection failed"
        assert stats['total_entries'] > 0, "Stats reporting failed"
        print("✅ Cache stats work")

        # Test persistence across restarts
        cache2 = PersistentCache(cache_dir=test_dir, max_size=10, semantic_threshold=0.95)
        retrieved_persistent = cache2.retrieve(query1, "rag", embedding1, "session1")
        assert retrieved_persistent == result1, "Persistence failed"
        print("✅ Persistence across restarts works")

        print("🎉 File-based cache tests passed!")

    finally:
        # Clean up
        shutil.rmtree(test_dir)

def test_cache_limits():
    """Test cache size limits and eviction"""
    print("🧪 Testing cache size limits...")

    test_dir = tempfile.mkdtemp()

    try:
        cache = PersistentCache(cache_dir=test_dir, max_size=3, semantic_threshold=0.95)

        # Fill cache beyond limit
        for i in range(5):
            query = f"Query {i}"
            embedding = np.random.rand(384)
            result = {"answer": f"Answer {i}", "sources": []}
            cache.store(query, "rag", result, embedding)

        # Should only have max_size entries
        assert len(cache) <= 3, "Size limit not enforced"
        print("✅ Cache size limits work")

    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_file_based_cache()
    test_cache_limits()
    print("🎉 All persistent cache tests passed!")