# Retrieval Pipeline

The active implementation is `rag_system/pipelines/retrieval_pipeline.py`, using `MultiVectorRetriever` in `rag_system/retrieval/retrievers.py`.

## Flow

1. The agent triages the query as direct answer, document RAG, or graph RAG when graph support is explicitly configured.
2. Document RAG embeds the query and searches every linked LanceDB table.
3. `search_type` selects one of:
   - `dense` / `vector`: vector similarity only.
   - `lexical` / `fts` / `bm25`: LanceDB full-text search only.
   - `hybrid`: both searches, fused with weighted reciprocal-rank fusion.
4. Candidates from all linked indexes are deduplicated and globally trimmed to `retrieval_k`.
5. Optional late-chunk tables use the `<main_table>_lc` naming convention. Neighbor expansion always reads from the table that produced the hit.
6. Optional ColBERT reranking, Provence sentence pruning, context expansion, synthesis, and verification run according to request/config flags.
7. Streaming callbacks expose analysis, retrieval, reranking, synthesis tokens, errors, and completion.

## Hybrid fusion

Hybrid search does not concatenate a fixed half of lexical results with a fixed half of vector results. Each leg retrieves candidates independently and weighted reciprocal-rank fusion combines their ranks:

```text
score = (1 - dense_weight) / (60 + lexical_rank)
      + dense_weight       / (60 + dense_rank)
```

`dense_weight` is clamped to 0–1 and defaults to 0.7.

## Multi-index and model compatibility

Sessions may link multiple indexes. The backend passes every corresponding `vector_table_name` to the RAG API. Because a query vector must match the stored vector dimensions, the backend rejects linking an index whose embedding model differs from the models already linked to that session.

## Cache isolation

Semantic cache entries are keyed by a namespace containing the session, selected tables, search type, and dense weight. A similar query in another session or against another index set cannot reuse a private answer.

## Routing

The overview router reads each index's generated JSONL summaries and inserts the actual summaries into its prompt. It no longer relies on hard-coded example document names. If no overview provides a confident route, document-related requests default toward RAG.

## Configuration

| Setting | Default | Meaning |
|---|---:|---|
| `retrieval_k` | 20 | Candidate count retained after cross-index merge |
| `search_type` | `hybrid` | Vector, lexical, or hybrid retrieval |
| `dense.weight` | 0.7 | Dense contribution to weighted RRF |
| `context_window_size` | 0 | Adjacent chunks included around a hit |
| `reranker.top_k` | 10 | Candidates retained by optional AI reranker |
| `latechunk.enabled` | build option | Query the corresponding late-chunk tables |

The active pipeline has no `DenseRetriever`, standalone `BM25Retriever`, `answer_stream()`, `BaseRetriever` registry, or `prompt_override` interface. Older references to those names described planned designs, not this implementation.
