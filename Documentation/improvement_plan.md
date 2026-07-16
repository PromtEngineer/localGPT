# Remaining improvement plan

This file contains known limits after the durable runtime implementation. Items already delivered—typed FastAPI/OpenAPI, durable runs and index jobs, replayable events, cancellation intent, artifacts, skills, tool policy, MCP/database/web/code tools, lightweight parsers, generated client contracts, structured logging, and a real model-backed workflow—are documented elsewhere and are not roadmap claims.

## Security and operations

1. Replace the optional shared bearer token with user identity, per-session/index authorization, and audit retention before any multi-user or public deployment.
2. Integrate a malware scanner at the upload/artifact boundary. Signature checks, archive limits, and parser isolation reduce risk but do not prove a file is safe.
3. Run the RAG worker itself in an isolated worker/container boundary for untrusted PDF/Office parsing and enforce CPU/memory/OCR time budgets externally.
4. Add formal versioned migrations for SQLite and LanceDB, plus backup/restore and disaster-recovery tests.
5. Pin container base images by digest and add dependency, secret, SBOM, and container scanning in CI.

## Durability and scale

1. Replace the in-process bounded executor with a multi-process durable queue when horizontal scaling is required. The database persists state/events/checkpoints, but queued tasks are not leased across replicas.
2. Add a cancellable internal indexing protocol. Current cancellation is immediate while queued and cooperative at agent/tool boundaries; a synchronous indexing/model HTTP call already in flight finishes before its result is discarded.
3. Add artifact/vector-table transactions, quotas, lifecycle policies, and orphan reconciliation across SQLite, blob storage, LanceDB, and overview manifests.
4. Move the internal RAG worker from its serialized legacy HTTP handler to request-scoped immutable pipelines or keyed worker processes.
5. Configure an OpenTelemetry exporter and production metric backend. The code currently emits redacted structured logs and optional spans but does not choose an operator-specific exporter.

## Retrieval quality

1. Grow `evals/` into representative multilingual and domain datasets and calibrate fusion, reranking, pruning, and cache thresholds against them.
2. Add claim-level citation entailment. Current citations identify retrieved source chunks and the evaluation harness checks answer/citation term recall, but retrieval alone does not prove every generated claim.
3. Add language-aware tokenization without forcing a transformer download for lightweight Ollama deployments.
4. Finish graph extraction/retrieval and multimodal page-image retrieval end to end, or remove those disabled configuration surfaces.

## Product

1. Surface durable run timelines, cancellation, retry, tool approvals, skills, artifacts, and connector status as first-class UI panels. The APIs and reconnecting client primitives exist; the current UI keeps its legacy chat/index flow.
2. Add Playwright accessibility and browser-flow coverage for upload rejection, index jobs, stream reconnection, approvals, and keyboard navigation.
