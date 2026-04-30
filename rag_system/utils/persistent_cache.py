"""
Persistent Caching System for LocalGPT

Implements Redis/file-based persistent caching to replace the in-memory TTL cache.
Provides semantic similarity matching for query caching with persistence across restarts.
"""

import json
import logging
import os
import time
import hashlib
import pickle
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

class PersistentCache:
    """
    Persistent cache that uses Redis if available, otherwise falls back to file-based storage.
    Maintains semantic similarity for query caching with persistence across restarts.
    """

    def __init__(self, cache_dir: str = "cache", redis_url: str = "redis://localhost:6379",
                 max_size: int = 1000, semantic_threshold: float = 0.98,
                 cache_scope: str = "global"):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory for file-based cache storage
            redis_url: Redis connection URL
            max_size: Maximum number of cache entries
            semantic_threshold: Similarity threshold for semantic matching
            cache_scope: 'global' or 'session' - whether cache is shared across sessions
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.redis_url = redis_url
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold
        self.cache_scope = cache_scope

        # Initialize Redis connection if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()  # Test connection
                logger.info("redis_cache_initialized")
                self.use_redis = True
            except (redis.ConnectionError, redis.ResponseError) as e:
                logger.warning("redis_connection_failed redis_url=%s error=%s", redis_url, e)
                self.use_redis = False
        else:
            logger.warning("redis_not_available")
            self.use_redis = False

        # In-memory index for fast semantic search (embeddings are expensive to load)
        self._embedding_index: Dict[str, np.ndarray] = {}
        self._metadata_index: Dict[str, Dict[str, Any]] = {}

        # Load existing cache on startup
        self._load_cache_index()

    def _get_cache_key(self, query: str, query_type: str, session_id: Optional[str] = None) -> str:
        """Generate a unique cache key"""
        if self.cache_scope == "session" and session_id:
            key_base = f"{session_id}:{query_type}:{query.strip().lower()}"
        else:
            key_base = f"{query_type}:{query.strip().lower()}"

        # Use hash to avoid key length issues
        return hashlib.md5(key_base.encode()).hexdigest()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes"""
        return pickle.dumps(embedding)

    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array"""
        return pickle.loads(data)

    def _save_to_file(self, key: str, data: Dict[str, Any]):
        """Save cache entry to file"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convert numpy array to list for JSON serialization
                serializable_data = data.copy()
                if 'embedding' in serializable_data and isinstance(serializable_data['embedding'], np.ndarray):
                    serializable_data['embedding'] = serializable_data['embedding'].tolist()

                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("cache_save_failed key=%s error=%s", key, e)

    def _load_from_file(self, key: str) -> Optional[Dict[str, Any]]:
        """Load cache entry from file"""
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert embedding list back to numpy array
            if 'embedding' in data and isinstance(data['embedding'], list):
                data['embedding'] = np.array(data['embedding'])

            # Check TTL
            if 'timestamp' in data:
                age = time.time() - data['timestamp']
                if age > 300:  # 5 minutes TTL (same as original)
                    cache_file.unlink()  # Delete expired entry
                    return None

            return data
        except Exception as e:
            logger.error("cache_load_failed key=%s error=%s", key, e)
            return None

    def _redis_key(self, key: str) -> str:
        """Get Redis key with namespace"""
        return f"localgpt_cache:{key}"

    def _save_to_redis(self, key: str, data: Dict[str, Any]):
        """Save cache entry to Redis"""
        try:
            redis_key = self._redis_key(key)
            # Store metadata as JSON
            metadata = data.copy()
            embedding_bytes = None

            if 'embedding' in metadata:
                embedding_bytes = self._serialize_embedding(metadata['embedding'])
                metadata['embedding'] = None  # Don't store in JSON

            # Set TTL to 5 minutes (300 seconds)
            self.redis_client.setex(f"{redis_key}:metadata", 300, json.dumps(metadata))
            if embedding_bytes:
                self.redis_client.setex(f"{redis_key}:embedding", 300, embedding_bytes)

        except Exception as e:
            logger.error("redis_save_failed key=%s error=%s", key, e)

    def _load_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """Load cache entry from Redis"""
        try:
            redis_key = self._redis_key(key)

            # Load metadata
            metadata_json = self.redis_client.get(f"{redis_key}:metadata")
            if not metadata_json:
                return None

            metadata = json.loads(metadata_json)

            # Load embedding if present
            embedding_bytes = self.redis_client.get(f"{redis_key}:embedding")
            if embedding_bytes and metadata.get('embedding') is None:
                metadata['embedding'] = self._deserialize_embedding(embedding_bytes)

            return metadata

        except Exception as e:
            logger.error("redis_load_failed key=%s error=%s", key, e)
            return None

    def _load_cache_index(self):
        """Load cache index for fast semantic search"""
        logger.info("loading_persistent_cache_index")

        if self.use_redis:
            # For Redis, we can't efficiently scan all keys, so we'll load on demand
            logger.info("redis_cache_index_loaded_on_demand")
            return

        # For file-based cache, scan directory and load embeddings into memory
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check if expired
                if 'timestamp' in data:
                    age = time.time() - data['timestamp']
                    if age > 300:  # 5 minutes
                        cache_file.unlink()
                        continue

                key = cache_file.stem

                # Load embedding into memory index
                if 'embedding' in data and isinstance(data['embedding'], list):
                    self._embedding_index[key] = np.array(data['embedding'])
                    self._metadata_index[key] = {
                        'session_id': data.get('session_id'),
                        'timestamp': data.get('timestamp', 0),
                        'query_type': data.get('query_type', 'unknown')
                    }
                    count += 1

            except Exception as e:
                logger.error("cache_file_load_error cache_file=%s error=%s", cache_file, e)

        logger.info("file_cache_index_loaded count=%s", count)

    def store(self, query: str, query_type: str, result: Dict[str, Any],
              embedding: Optional[np.ndarray] = None, session_id: Optional[str] = None):
        """
        Store a result in the persistent cache.

        Args:
            query: The original query string
            query_type: Type of query (e.g., 'rag', 'direct')
            result: The result to cache
            embedding: Query embedding for semantic matching
            session_id: Session ID for session-scoped caching
        """
        cache_key = self._get_cache_key(query, query_type, session_id)

        cache_data = {
            'query': query,
            'query_type': query_type,
            'result': result,
            'timestamp': time.time(),
            'session_id': session_id
        }

        if embedding is not None:
            cache_data['embedding'] = embedding

        # Store in persistent storage
        if self.use_redis:
            self._save_to_redis(cache_key, cache_data)
        else:
            self._save_to_file(cache_key, cache_data)

        # Update in-memory index
        if embedding is not None:
            self._embedding_index[cache_key] = embedding.copy()
            self._metadata_index[cache_key] = {
                'session_id': session_id,
                'timestamp': cache_data['timestamp'],
                'query_type': query_type
            }

        # Enforce size limit
        self._enforce_size_limit()

    def retrieve(self, query: str, query_type: str, embedding: Optional[np.ndarray] = None,
                session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a result from the cache using exact match or semantic similarity.

        Args:
            query: The query string
            query_type: Type of query
            embedding: Query embedding for semantic matching
            session_id: Session ID for session-scoped caching

        Returns:
            Cached result if found, None otherwise
        """
        # First try exact match
        cache_key = self._get_cache_key(query, query_type, session_id)

        result = self._load_cache_entry(cache_key)
        if result:
            logger.info("exact_cache_hit query=%s", query[:50])
            return result['result']

        # If we have an embedding, try semantic matching
        if embedding is not None:
            semantic_result = self._find_semantic_match(embedding, session_id)
            if semantic_result:
                return semantic_result

        return None

    def _load_cache_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a cache entry by key"""
        if self.use_redis:
            return self._load_from_redis(key)
        else:
            return self._load_from_file(key)

    def _find_semantic_match(self, query_embedding: np.ndarray,
                           session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar cached queries.

        Returns:
            Cached result if similarity threshold met, None otherwise
        """
        best_match = None
        best_similarity = 0.0

        # Search through in-memory embedding index
        for cache_key, cached_embedding in self._embedding_index.items():
            # Check session scope
            if self.cache_scope == "session" and session_id:
                cached_session = self._metadata_index.get(cache_key, {}).get('session_id')
                if cached_session != session_id:
                    continue

            try:
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity >= self.semantic_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_key
            except (ValueError, TypeError):
                continue

        if best_match:
            # Load the full result from persistent storage
            cached_data = self._load_cache_entry(best_match)
            if cached_data:
                logger.info("semantic_cache_hit similarity=%s", round(best_similarity, 3))
                return cached_data['result']

        return None

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if v1.shape != v2.shape:
            return 0.0

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing oldest entries"""
        if len(self._embedding_index) <= self.max_size:
            return

        # Remove oldest entries
        entries = [(k, self._metadata_index[k]['timestamp']) for k in self._embedding_index.keys()]
        entries.sort(key=lambda x: x[1])  # Sort by timestamp

        to_remove = len(entries) - self.max_size
        for i in range(to_remove):
            key, _ = entries[i]
            self._remove_entry(key)

    def _remove_entry(self, key: str):
        """Remove a cache entry"""
        # Remove from in-memory index
        if key in self._embedding_index:
            del self._embedding_index[key]
        if key in self._metadata_index:
            del self._metadata_index[key]

        # Remove from persistent storage
        if self.use_redis:
            try:
                redis_key = self._redis_key(key)
                self.redis_client.delete(f"{redis_key}:metadata", f"{redis_key}:embedding")
            except Exception as e:
                logger.error("redis_remove_failed key=%s error=%s", key, e)
        else:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()

    def clear(self):
        """Clear all cache entries"""
        self._embedding_index.clear()
        self._metadata_index.clear()

        if self.use_redis:
            try:
                # Delete all keys with our namespace
                keys = self.redis_client.keys("localgpt_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error("redis_cache_clear_failed error=%s", e)
        else:
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

        logger.info("cache_cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'backend': 'redis' if self.use_redis else 'file',
            'total_entries': len(self._embedding_index),
            'max_size': self.max_size,
            'cache_dir': str(self.cache_dir) if not self.use_redis else None,
            'semantic_threshold': self.semantic_threshold,
            'cache_scope': self.cache_scope
        }

    def __len__(self) -> int:
        """Return number of cached entries"""
        return len(self._embedding_index)
