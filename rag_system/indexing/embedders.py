# from rag_system.indexing.representations import BM25Generator
import lancedb
import os
import pyarrow as pa
from typing import Any, Dict, List, Optional
import numpy as np
import json

# ---------------------------------------------------------------------------
# Per-table embedder identity + vector-normalization marker
# ---------------------------------------------------------------------------
# Two facts have to travel with a LanceDB table, because neither can be
# recovered from the vectors themselves:
#
#   1. WHICH embedding model wrote them. The vector-width check below cannot
#      catch a swap between two same-width models (harrier-oss-v1-0.6b and
#      Qwen3-Embedding-0.6B are both 1024-dim), and appending one model's
#      vectors to the other's table silently produces nonsense rankings.
#   2. WHETHER they are L2-normalized. Both model cards specify cosine
#      similarity, but LanceDB's default metric is L2; L2 ordering equals
#      cosine ordering only when every vector is unit length. Normalizing
#      invalidates vectors written before this existed, so it is recorded
#      per table rather than assumed globally.
#
# Primary store: Arrow schema metadata on the table, which lancedb 0.36.0
# round-trips through create_table/open_table (verified on this tree). If a
# LanceDB version ever drops it, a sidecar JSON is written next to the database
# instead — under <db_path>/table_meta/<table>.json, *not* a single global
# directory, because different indexes legitimately reuse the same table name
# in different database directories (the eval harness does exactly that).
#
# A table with neither marker is a legacy table: its embedder is unknown and
# its vectors are unnormalized. It keeps working, unnormalized, with a warning.

_META_MODEL_KEY = b"localgpt_embedding_model"
_META_NORMALIZED_KEY = b"localgpt_normalized"
_SIDECAR_DIRNAME = "table_meta"


class EmbedderMismatchError(RuntimeError):
    """A table was written by a different embedding model than the configured one."""


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Unit-length copy of *vector*; returned unchanged when that is impossible."""
    array = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(array).all():
        # Leave it alone so the NaN/Inf reporting downstream stays accurate.
        return array
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _sidecar_path(db_path: str, table_name: str) -> str:
    return os.path.join(db_path, _SIDECAR_DIRNAME, f"{table_name}.json")


def _read_sidecar(db_path: Optional[str], table_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not db_path or not table_name:
        return None
    path = _sidecar_path(db_path, table_name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_sidecar(db_path: str, table_name: str, model_name: str, normalized: bool) -> None:
    path = _sidecar_path(db_path, table_name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"embedding_model": model_name, "normalized": bool(normalized)}, fh, indent=2)
    except OSError as e:
        print(f"⚠️  Could not write table marker {path}: {e}")


def table_schema_metadata(model_name: str, normalized: bool) -> Dict[bytes, bytes]:
    return {
        _META_MODEL_KEY: model_name.encode("utf-8"),
        _META_NORMALIZED_KEY: (b"true" if normalized else b"false"),
    }


def read_table_marker(tbl, db_path: Optional[str] = None,
                      table_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The embedder identity recorded for *tbl*, or None for a legacy table."""
    metadata = getattr(getattr(tbl, "schema", None), "metadata", None) or {}
    raw_model = metadata.get(_META_MODEL_KEY)
    if raw_model:
        raw_norm = metadata.get(_META_NORMALIZED_KEY, b"false")
        return {
            "embedding_model": raw_model.decode("utf-8"),
            "normalized": raw_norm.decode("utf-8").lower() == "true",
            "source": "lancedb schema metadata",
        }
    sidecar = _read_sidecar(db_path, table_name)
    if sidecar and sidecar.get("embedding_model"):
        return {
            "embedding_model": sidecar["embedding_model"],
            "normalized": bool(sidecar.get("normalized")),
            "source": _sidecar_path(db_path, table_name),
        }
    return None


def assert_embedder_matches(table_name: str, marker: Dict[str, Any], configured_model: str) -> None:
    """Raise when *marker* names a different embedder than *configured_model*."""
    recorded = marker.get("embedding_model")
    if not recorded or not configured_model or recorded == configured_model:
        return
    raise EmbedderMismatchError(
        f"Table '{table_name}' was built with embedding model '{recorded}' but the "
        f"pipeline is configured for '{configured_model}'. The two produce vectors in "
        f"different spaces (a matching vector width does not make them compatible), so "
        f"any result from this table would be meaningless. Rebuild the index with "
        f"'{configured_model}', or set EMBEDDING_MODEL='{recorded}' to keep using it."
    )


def legacy_table_warning(table_name: str) -> str:
    return (
        f"⚠️  Table '{table_name}' carries no embedder marker — it was built before "
        f"localGPT recorded one. Its embedding model cannot be verified and its vectors "
        f"are unnormalized, so scores use legacy unnormalized vectors; a rebuilt index "
        f"is recommended."
    )


class LanceDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        print(f"LanceDB connection established at: {db_path}")

    def get_table(self, table_name: str):
        return self.db.open_table(table_name)

    def create_table(self, table_name: str, schema: pa.Schema, mode: str = "overwrite"):
        print(f"Creating table '{table_name}' with mode '{mode}'...")
        return self.db.create_table(table_name, schema=schema, mode=mode)

class VectorIndexer:
    """
    Handles the indexing of vector embeddings and rich metadata into LanceDB.
    The 'text' field is the content that gets embedded (which can be enriched).
    The original, clean text is stored in the metadata.
    """
    def __init__(self, db_manager: LanceDBManager):
        self.db_manager = db_manager

    @staticmethod
    def _table_vector_dim(tbl) -> int | None:
        """Vector width of an existing LanceDB table, or None if it can't be read."""
        try:
            field = tbl.schema.field("vector")
        except (KeyError, AttributeError):
            return None
        return getattr(field.type, "list_size", None)

    def index(self, table_name: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray,
              embedding_model: Optional[str] = None):
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be the same.")
        if not chunks:
            print("No chunks to index.")
            return

        # Dimensionality always comes from the vectors the loaded model produced.
        vector_dim = int(embeddings[0].shape[0])

        db = self.db_manager.db  # underlying LanceDB connection
        db_path = getattr(self.db_manager, "db_path", None)
        table_exists = bool(hasattr(db, "table_names") and table_name in db.table_names())

        # ------------------------------------------------------------------
        # Decide, before touching the vectors, which table this is:
        #   new table      -> record the embedder, write normalized vectors
        #   marked table   -> embedder must match; follow the table's own flag
        #   legacy table   -> unknown embedder, unnormalized; warn and comply
        # ------------------------------------------------------------------
        existing_tbl = None
        normalize = bool(embedding_model)  # can't claim an identity we weren't given
        if table_exists:
            existing_tbl = self.db_manager.get_table(table_name)
            existing_dim = self._table_vector_dim(existing_tbl)
            if existing_dim is not None and existing_dim != vector_dim:
                raise ValueError(
                    f"Table '{table_name}' stores {existing_dim}-dim vectors but the current "
                    f"embedding model produced {vector_dim}-dim vectors. Changing the embedding "
                    f"model requires rebuilding the index."
                )
            marker = read_table_marker(existing_tbl, db_path, table_name)
            if marker is None:
                print(legacy_table_warning(table_name))
                normalize = False
            else:
                if embedding_model:
                    assert_embedder_matches(table_name, marker, embedding_model)
                normalize = bool(marker["normalized"])

        if normalize:
            embeddings = [l2_normalize(v) for v in embeddings]

        # The schema stores the text that was used for the embedding (potentially enriched)
        # and the full metadata object as a JSON string.
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("text", pa.string(), nullable=False),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string())
        ], metadata=table_schema_metadata(embedding_model, normalize) if embedding_model else None)

        data = []
        skipped_count = 0
        
        for chunk, vector in zip(chunks, embeddings):
            # Check for NaN values in the vector
            if np.isnan(vector).any():
                print(f"⚠️ Skipping chunk '{chunk.get('chunk_id', 'unknown')}' due to NaN values in embedding")
                skipped_count += 1
                continue
                
            # Check for infinite values in the vector
            if np.isinf(vector).any():
                print(f"⚠️ Skipping chunk '{chunk.get('chunk_id', 'unknown')}' due to infinite values in embedding")
                skipped_count += 1
                continue
            
            # Ensure original_text is in metadata if not already present
            if 'original_text' not in chunk['metadata']:
                chunk['metadata']['original_text'] = chunk['text']

            # Extract document_id and chunk_index for top-level storage
            doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", -1)

            # Defensive check for text content to ensure it's a non-empty string
            text_content = chunk.get('text', '')
            if not text_content or not isinstance(text_content, str):
                text_content = ""

            data.append({
                "vector": vector.tolist(),
                "text": text_content,
                "chunk_id": chunk['chunk_id'],
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "metadata": json.dumps(chunk)
            })

        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} chunks due to invalid embeddings (NaN or infinite values)")
        
        if not data:
            print("❌ No valid embeddings to index after filtering out NaN/infinite values")
            return

        # Incremental indexing: append to existing table if present, otherwise create it
        if existing_tbl is not None:
            tbl = existing_tbl
            print(f"Appending {len(data)} vectors to existing table '{table_name}'.")
        else:
            print(f"Creating table '{table_name}' (new) and adding {len(data)} vectors...")
            tbl = self.db_manager.create_table(table_name, schema=schema, mode="create")
            if embedding_model:
                # Trust nothing: re-read the marker off the created table. If this
                # LanceDB build dropped the Arrow schema metadata, fall back to the
                # sidecar so the guard still has something to compare against.
                if read_table_marker(self.db_manager.get_table(table_name)) is None and db_path:
                    _write_sidecar(db_path, table_name, embedding_model, normalize)
                print(f"🔖 Table '{table_name}' marked: embedder='{embedding_model}', "
                      f"normalized={str(normalize).lower()}.")

        # Add data with NaN handling configuration
        try:
            tbl.add(data, on_bad_vectors='drop')
            print(f"✅ Indexed {len(data)} vectors into table '{table_name}'.")
        except Exception as e:
            print(f"❌ Failed to add data to table: {e}")
            # Fallback: try with fill strategy
            try:
                print("🔄 Retrying with NaN fill strategy...")
                tbl.add(data, on_bad_vectors='fill', fill_value=0.0)
                print(f"✅ Indexed {len(data)} vectors into table '{table_name}' (with NaN fill).")
            except Exception as e2:
                print(f"❌ Failed to add data even with NaN fill: {e2}")
                raise

# BM25Indexer is no longer needed as we are moving to LanceDB's native FTS.
# class BM25Indexer:
#     ...

if __name__ == '__main__':
    print("embedders.py updated for contextual enrichment.")
    
    # This chunk has been "enriched". The 'text' field contains the context.
    enriched_chunk = {
        'chunk_id': 'doc1_0', 
        'text': 'Context: Discusses animals.\n\n---\n\nOriginal: The cat sat on the mat.', 
        'metadata': {
            'original_text': 'The cat sat on the mat.',
            'contextual_summary': 'Discusses animals.',
            'document_id': 'doc1', 
            'title': 'Pet Stories'
        }
    }
    sample_embeddings = np.random.rand(1, 128).astype('float32')

    DB_PATH = "./rag_system/index_store/lancedb"
    db_manager = LanceDBManager(db_path=DB_PATH)
    vector_indexer = VectorIndexer(db_manager=db_manager)

    vector_indexer.index(
        table_name="enriched_text_embeddings", 
        chunks=[enriched_chunk], 
        embeddings=sample_embeddings
    )
    
    try:
        tbl = db_manager.get_table("enriched_text_embeddings")
        df = tbl.limit(1).to_pandas()
        df['metadata'] = df['metadata'].apply(json.loads)
        print("\n--- Verification ---")
        print("Embedded Text:", df['text'].iloc[0])
        print("Original Text from Metadata:", df['metadata'].iloc[0]['original_text'])
    except Exception as e:
        print(f"Could not verify LanceDB table. Error: {e}")
