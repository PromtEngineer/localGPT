# LocalGPT backend

The browser-facing service is a typed FastAPI application on port 8000. It owns sessions, messages, index metadata, durable runs/events/checkpoints, artifacts, skills, tool policy, and OpenAPI. Retrieval and vector indexing are delegated to the internal RAG worker configured by `RAG_API_URL`.

```bash
pip install -r backend/requirements.txt
python -m backend.server
```

The default bind is `127.0.0.1:8000`; use `LOCALGPT_BACKEND_HOST` and `LOCALGPT_BACKEND_PORT`. Interactive API docs are at `/docs`. Regenerate the checked-in schema and client types with `npm run api:types`.

The agent runtime supports Ollama and OpenAI-compatible generation providers, explicit schema-validated tools, configured MCP/read-only database connectors, safe public-web tools, content-addressed artifacts, immutable skills, and optional Docker-isolated Python. Request permissions only narrow `LOCALGPT_AGENT_PERMISSIONS`; they cannot grant new server authority. Side-effecting tools require explicit approval.

Uploads are bounded, inspected for obvious signature/type mismatches and pathological Office archives, stored beneath `LOCALGPT_UPLOAD_DIR`, and recorded as artifacts with provenance. This preflight is not malware scanning; public deployments should add a scanner at the upload boundary.

See [the API reference](../Documentation/api_reference.md) and [architecture overview](../Documentation/architecture_overview.md).
