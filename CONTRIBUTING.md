# Contributing to LocalGPT

## Setup

Requirements are Python 3.10+, Node.js 18+, npm, Git, and Ollama for model-backed runs.

```bash
git clone https://github.com/YOUR_USERNAME/localGPT.git
cd localGPT
git remote add upstream https://github.com/PromtEngineer/localGPT.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm ci
cp .env.example .env
```

Start the development system with `python run_system.py --mode dev`, or run each module as shown in the README.

## Change workflow

1. Create a focused branch from the current default branch.
2. Add a failing regression test before a bug fix, or a failing behavioral test before a new capability.
3. Make the smallest coherent implementation and keep service boundaries intact: the backend owns chat persistence; the RAG API owns indexing/retrieval/model work; the browser calls the backend proxy.
4. Update documentation when a request field, default, security boundary, storage layout, or user-visible capability changes.
5. Run the verification suite before submitting.

```bash
python -m pytest -q
python -m compileall -q backend rag_system run_system.py localgpt_runtime.py
ruff check backend rag_system run_system.py localgpt_runtime.py tests
npm test
npm run typecheck
npm run lint
npm run build
docker compose config
docker compose -f docker-compose.local-ollama.yml config
```

Model-free tests should not require Ollama or download Hugging Face weights. Mark genuine model-backed integration tests separately and document their model/resource prerequisites.

## Code expectations

- Preserve existing user changes and avoid unrelated rewrites.
- Use type hints for new Python interfaces and concrete TypeScript types for API/event data.
- Validate untrusted input at the service boundary, not only in the UI.
- Keep index operations scoped to `LOCALGPT_UPLOAD_DIR` and session retrieval scoped to linked index tables.
- Never add credentials to `NEXT_PUBLIC_*`, source files, tests, or examples.
- Avoid enabling Hugging Face `trust_remote_code` by default.
- Do not describe experimental graph or page-image/VLM scaffolding as an implemented product feature.

## Pull request description

Include the problem, behavioral change, tests run, documentation updated, migration/compatibility impact, and any remaining risk. For retrieval changes, include evaluation evidence or a small reproducible corpus/query example rather than only a subjective output sample.
