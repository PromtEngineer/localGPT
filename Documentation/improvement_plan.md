# Remaining improvement plan

This file records limitations that still exist after the implementation/documentation alignment pass. It is a roadmap, not a list of current capabilities.

## High priority

1. Add a true model-backed end-to-end test fixture that builds a small index, restarts both services, queries it through the backend SSE endpoint, checks citations, and deletes it. Current automated tests cover contracts and core logic without downloading multi-gigabyte weights.
2. Replace the shared bearer token with real user identity and per-session/index authorization before any multi-user or public deployment.
3. Add structured schema migrations for SQLite and versioned migrations for LanceDB artifacts. Startup currently performs only backward-compatible schema/index cleanup.
4. Add upload content inspection: MIME sniffing, decompression/document complexity limits, malware scanning hooks, and OCR execution limits. Extension and byte limits alone do not make untrusted documents safe.
5. Introduce cancellation, job persistence, and resumability for long index builds. An interrupted process currently leaves the database build status recoverable only by retrying the rebuild.

## Architecture and scale

1. Replace process-global mutable agent/index configuration and the protective serialization locks with request-scoped immutable pipelines or a keyed worker pool.
2. Move indexing to a durable job queue with progress persistence rather than holding an HTTP request for the whole build.
3. Add an artifact transaction strategy so SQLite metadata, LanceDB tables, overviews, and original uploads commit or roll back as one logical build.
4. Add storage quotas, lifecycle/garbage collection, and reconciliation for orphaned uploads/vector tables.
5. Add observability: structured logs with request IDs, latency histograms by pipeline stage, model/cache metrics, and redaction rules.

## Retrieval quality

1. Calibrate weighted RRF and reranking on a maintained evaluation set rather than relying on universal defaults.
2. Add citation entailment checks that bind answer spans to exact source rows; retrieval references alone do not prove every claim is supported.
3. Add language-aware tokenization for lexical search and chunk sizing. Current overlap is a deterministic word-token window.
4. Evaluate semantic-cache false positives on a maintained query set; rebuilds already invalidate the process-wide query cache.
5. Complete graph extraction/retrieval or remove the dormant graph configuration surface.

## Product and code quality

1. Generate the existing TypeScript streaming/citation types and Python request validation from one versioned API schema so the two implementations cannot drift.
2. Replace ad-hoc `http.server` handlers with a framework that provides request schemas, multipart streaming, cancellation, OpenAPI generation, and middleware.
3. Add accessibility and browser-flow tests for index creation, linking, upload failures, streaming reconnection, and keyboard navigation.
4. Decide whether full page-image/VLM retrieval is a product requirement. If so, integrate it end to end with storage, retrieval fusion, citations, resource limits, UI, and tests; otherwise delete the experimental scaffold.
5. Pin or digest-lock Docker base images and add dependency/container vulnerability scanning in CI.
