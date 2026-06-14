# from rag_system.indexing.representations import BM25Generator
import json
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa

import lancedb


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

    def index(
        self,
        table_name: str,
        chunks: List[Dict[str, Any]],
        embeddings,
        metadata_schema: List[Dict[str, Any]] | None = None,
    ):
        # Drop None placeholders left by OOM-killed embedding batches.
        none_count = sum(1 for e in embeddings if e is None)
        if none_count > 0:
            print(
                f"⚠️ {none_count} chunks have no embedding (OOM batch failure); skipping them"
            )
            pairs = [
                (c, e)
                for c, e in zip(chunks, embeddings, strict=False)
                if e is not None
            ]
            chunks = [p[0] for p in pairs]
            embeddings = [p[1] for p in pairs]

        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be the same.")
        if not chunks:
            print("No chunks to index.")
            return

        vector_dim = embeddings[0].shape[0]

        # The schema stores the text that was used for the embedding (potentially enriched)
        # and the full metadata object as a JSON string.
        fields = [
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("text", pa.string(), nullable=False),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string()),
        ]

        # Typed custom-metadata columns (meta_*) for query-time filtering.
        # Chunks carry the same flattened column set per build (None when a
        # file is untagged) so the Arrow schema stays stable across files.
        _SCHEMA_ARROW_TYPES = {
            "string": pa.string(),
            "boolean": pa.bool_(),
            "integer": pa.int64(),
            "float": pa.float64(),
        }
        _VALUE_ARROW_TYPES = {
            str: pa.string(),
            bool: pa.bool_(),
            int: pa.int64(),
            float: pa.float64(),
        }
        meta_columns: dict = {}
        for field in metadata_schema or []:
            meta_columns[f"meta_{field['name']}"] = _SCHEMA_ARROW_TYPES[field["type"]]
        for c in chunks:
            for col, val in (c.get("_meta_columns") or {}).items():
                if col not in meta_columns and val is not None:
                    meta_columns[col] = _VALUE_ARROW_TYPES.get(type(val), pa.string())
            if c.get("_meta_columns"):
                for col in c["_meta_columns"]:
                    meta_columns.setdefault(col, pa.string())
        for col, pa_type in meta_columns.items():
            fields.append(pa.field(col, pa_type, nullable=True))

        schema = pa.schema(fields)

        data = []
        skipped_count = 0

        for chunk, vector in zip(chunks, embeddings, strict=False):
            # Check for NaN values in the vector
            if np.isnan(vector).any():
                print(
                    f"⚠️ Skipping chunk '{chunk.get('chunk_id', 'unknown')}' due to NaN values in embedding"
                )
                skipped_count += 1
                continue

            # Check for infinite values in the vector
            if np.isinf(vector).any():
                print(
                    f"⚠️ Skipping chunk '{chunk.get('chunk_id', 'unknown')}' due to infinite values in embedding"
                )
                skipped_count += 1
                continue

            # Ensure original_text is in metadata if not already present
            if "original_text" not in chunk["metadata"]:
                chunk["metadata"]["original_text"] = chunk["text"]

            # Extract document_id and chunk_index for top-level storage
            doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", -1)

            # Defensive check for text content to ensure it's a non-empty string
            text_content = chunk.get("text", "")
            if not text_content or not isinstance(text_content, str):
                text_content = ""

            row = {
                "vector": vector.tolist(),
                "text": text_content,
                "chunk_id": chunk["chunk_id"],
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "metadata": json.dumps(
                    {k: v for k, v in chunk.items() if k != "_meta_columns"}
                ),
            }
            if meta_columns:
                chunk_meta = chunk.get("_meta_columns") or {}
                for col in meta_columns:
                    row[col] = chunk_meta.get(col)
            data.append(row)

        if skipped_count > 0:
            print(
                f"⚠️ Skipped {skipped_count} chunks due to invalid embeddings (NaN or infinite values)"
            )

        if not data:
            print(
                "❌ No valid embeddings to index after filtering out NaN/infinite values"
            )
            return

        # Incremental indexing: append to existing table if present, otherwise create it
        db = self.db_manager.db  # underlying LanceDB connection

        if hasattr(db, "table_names") and table_name in db.table_names(limit=10_000):
            tbl = self.db_manager.get_table(table_name)
            print(f"Appending {len(data)} vectors to existing table '{table_name}'.")
        else:
            print(
                f"Creating table '{table_name}' (new) and adding {len(data)} vectors..."
            )
            tbl = self.db_manager.create_table(table_name, schema=schema, mode="create")

        # Add data with NaN handling configuration
        try:
            tbl.add(data, on_bad_vectors="drop")
            print(f"✅ Indexed {len(data)} vectors into table '{table_name}'.")
        except Exception as e:
            print(f"❌ Failed to add data to table: {e}")
            # Fallback: try with fill strategy
            try:
                print("🔄 Retrying with NaN fill strategy...")
                tbl.add(data, on_bad_vectors="fill", fill_value=0.0)
                print(
                    f"✅ Indexed {len(data)} vectors into table '{table_name}' (with NaN fill)."
                )
            except Exception as e2:
                print(f"❌ Failed to add data even with NaN fill: {e2}")
                raise

        # Build an IVF-PQ ANN index for large tables (≥5000 rows) to speed up queries
        try:
            total_rows = tbl.count_rows() if hasattr(tbl, "count_rows") else len(data)
            if total_rows >= 5000:
                tbl.create_index(
                    metric="cosine", num_partitions=256, num_sub_vectors=96
                )
                print(
                    f"✅ IVF-PQ ANN index built for '{table_name}' ({total_rows} rows)."
                )
        except Exception as ann_err:
            # ANN index is optional; log and continue
            print(f"⚠️ Could not build ANN index for '{table_name}': {ann_err}")


# BM25Indexer is no longer needed as we are moving to LanceDB's native FTS.
# class BM25Indexer:
#     ...

if __name__ == "__main__":
    print("embedders.py updated for contextual enrichment.")

    # This chunk has been "enriched". The 'text' field contains the context.
    enriched_chunk = {
        "chunk_id": "doc1_0",
        "text": "Context: Discusses animals.\n\n---\n\nOriginal: The cat sat on the mat.",
        "metadata": {
            "original_text": "The cat sat on the mat.",
            "contextual_summary": "Discusses animals.",
            "document_id": "doc1",
            "title": "Pet Stories",
        },
    }
    sample_embeddings = np.random.rand(1, 128).astype("float32")

    DB_PATH = "./rag_system/index_store/lancedb"
    db_manager = LanceDBManager(db_path=DB_PATH)
    vector_indexer = VectorIndexer(db_manager=db_manager)

    vector_indexer.index(
        table_name="enriched_text_embeddings",
        chunks=[enriched_chunk],
        embeddings=sample_embeddings,
    )

    try:
        tbl = db_manager.get_table("enriched_text_embeddings")
        df = tbl.limit(1).to_pandas()
        df["metadata"] = df["metadata"].apply(json.loads)
        print("\n--- Verification ---")
        print("Embedded Text:", df["text"].iloc[0])
        print("Original Text from Metadata:", df["metadata"].iloc[0]["original_text"])
    except Exception as e:
        print(f"Could not verify LanceDB table. Error: {e}")
