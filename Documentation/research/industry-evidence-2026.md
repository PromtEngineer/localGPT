All four researchers returned. Here is the consolidated deliverable.

---

# STATE OF THE ART IN AGENTIC RETRIEVAL — RAW FINDINGS
**Compiled 2026-08-08. First-party engineering blogs, official docs, and named practitioners only.**

**Method note:** ~180 first-party pages were fetched directly across four parallel research tracks plus a primary Anthropic/OpenAI sweep. Every URL below was actually fetched unless explicitly marked `[snippet-only]` or listed under GAPS. Dates are as printed on the page. The session's WebSearch quota (200) was exhausted; late-stage sources were reached by direct URL. Several notable pages 403 to automated fetch (`openai.com/index/*`, `help.openai.com`, `perplexity.ai/hub/*`) — substitutes are noted.

---

## SECTION 1 — AGENTIC RAG vs CLASSIC PIPELINES: WHAT THE INDUSTRY CONVERGED ON

### 1.1 The one clean vendor decision table

**CLAIM:** LangChain ships an explicit three-way taxonomy with a tradeoff table. **2-Step RAG**: "the retrieval step is always executed before the generation step. This architecture is straightforward and predictable" — Control High / Flexibility Low / Latency Fast; use case "FAQs, documentation bots." **Agentic RAG**: "an agent… reasons step-by-step and decides when and how to retrieve information during the interaction" — Control Low / Flexibility High / Latency Variable; use case "Research assistants with multiple tools." **Hybrid RAG** in between. Explicit latency argument for the fixed pipeline: "Latency is generally more predictable in 2-Step RAG, as the maximum number of LLM calls is known and capped."
**SOURCE:** LangChain — Retrieval (OSS docs) — https://docs.langchain.com/oss/python/langchain/retrieval — undated (LangChain 1.x)
**STATUS:** GA docs
*This is the single clearest published "when is the fixed pipeline still right" answer from any vendor.*

**CLAIM:** LangChain's operational definition of agentic hinges on conditionality: "That choice is what makes the system agentic rather than a fixed retrieve-then-generate pipeline: retrieval runs only when the model requests it." Shipped patterns: query-or-respond gate, grade-documents-for-relevance, rewrite-question-and-re-retrieve (CRAG in LangGraph form).
**SOURCE:** LangChain — Build a custom RAG agent with LangGraph — https://docs.langchain.com/oss/python/langgraph/agentic-rag — undated
**STATUS:** OSS pattern / shipped

**CLAIM:** Microsoft made the choice a literal config enum. `TextSearchProvider.SearchTime` has exactly two values: `BeforeAIInvoke` (search runs prior to every model invocation = classic fixed pipeline, and it is the **default**) or on-demand via function calling (= agentic). Python bridges Semantic Kernel VectorStore collections into agent tools with `keyword_hybrid` and `semantic_hybrid`.
**SOURCE:** Microsoft — Agent Framework, "RAG" — https://learn.microsoft.com/en-us/agent-framework/agents/rag — ms.date 2025-11-11, updated 2026-07-10
**STATUS:** shipped (.NET/Python); Go "coming soon"

**CLAIM:** OpenAI publishes **no** "agentic RAG vs naive RAG" framing at all. Retrieval is only ever a tool: agents get context via `instructions`, input messages, function tools for "on-demand context — the LLM decides when it needs some data," and "Retrieval or web search" as "special tools" for "grounding the response in relevant contextual data."
**SOURCE:** OpenAI — Agents SDK, Context management — https://openai.github.io/openai-agents-python/context/ and Tools — https://openai.github.io/openai-agents-python/tools/ — undated
**STATUS:** GA — *notable negative finding: OpenAI never validates the fixed-pipeline form in docs.*

### 1.2 The framework vendors' own repositioning

**CLAIM:** LlamaIndex: "naive RAG by itself isn't sufficient to meet enterprise needs" — but names when it *is* sufficient: "search this handbook" and single-document summarization for human review. Adopt Agentic Document Workflows when workflows span multiple document types, enforce business rules, update systems of record, or run at high volume.
**SOURCE:** LlamaIndex / Jerry Liu — "Beyond Chatbots: Adopting Agentic Document Workflows for Enterprises" — https://www.llamaindex.ai/blog/beyond-chatbots-adopting-agentic-document-workflows-for-enterprises — 2025-04-23
**STATUS:** opinion-recommendation + product framing

**CLAIM:** ADW is positioned as beyond *both* IDP and RAG: "a step beyond both traditional Intelligent Document Processing (IDP) and RAG paradigms, which are focused on small, isolated steps." Four-stage typed pipeline: Parse → Retrieve → Reason → Act.
**SOURCE:** LlamaIndex — "Agentic Document Workflows: A Practical Guide" — https://www.llamaindex.ai/blog/introducing-agentic-document-workflows — 2025-01-09
**STATUS:** OSS pattern + product

**CLAIM — MAJOR REVERSAL:** LlamaIndex now says its own category is receding. Verbatim: "if you equip agents with good filesystem tools, they can do dynamic search over document collections that outperforms naive semantic search," and RAG frameworks — "the kind of thing LlamaIndex and LangChain (and some others) built — aren't as central as they used to be." Retained moat is parsing: frontier VLMs "struggle with the long tail of accuracy parsing information-rich pages, like line charts, extremely dense tables (~hundreds of rows/columns in a single page), and handwritten forms." LlamaParse: 50+ formats, 500M+ pages processed.
**SOURCE:** LlamaIndex — "LlamaIndex is more than a RAG Framework. It is Agentic Document Processing." — https://www.llamaindex.ai/blog/llamaindex-is-more-than-a-rag-framework — 2026-03-03
**STATUS:** vendor repositioning

**CLAIM:** Weaviate frames agentic retrieval as the default interaction layer for 2026 — Query Agent GA, Transformation/Personalization Agents in preview, agents as "primary data interaction tools," 2026 focus on "shared memory systems enabling agents that learn iteratively."
**SOURCE:** Weaviate — "Weaviate in 2025: Reliable Foundations for Agentic Systems" — https://weaviate.io/blog/weaviate-in-2025 — 2026-01-29
**STATUS:** GA + preview mix / partly aspirational

### 1.3 The infrastructure-side argument

**CLAIM (strongest vendor statement of the thesis):** Vespa's CEO argues the fixed vector-search-then-generate pipeline is a category error. "Agents, in contrast, are not so clueless. And they certainly aren't lazy!" Agents should "string together many of these queries to reach its goal," needing proximity-based lexical search, quality-weighted semantic search, temporal filtering and aggregation, and *a menu of rank profiles to choose from* — then run a sequenced fan-out: "First gaining an overview, then researching more specific topics, forming hypotheses, verifying important details in them." Three-stage maturity model: vector-only DB → hybrid + ML ranking → "search as code."
**SOURCE:** Vespa — Jon Bratseth — "Your agent wants to search like a 2010 quant" — https://blog.vespa.ai/your-agent-wants-to-search-like-a-2010-quant/ — 2026-07-07
**STATUS:** vendor position piece (no numbers, no cost analysis)

**CLAIM:** turbopuffer describes the workload shift concretely: "the LLM is very good at reasoning with the data. And so we're just the tool call," with agents issuing "an enormous amount of queries all at once" and "more concurrency than I've ever seen before." Cursor saw a 95% cost reduction post-migration; Notion does "a ridiculous amount of queries in every round trip."
**SOURCE:** turbopuffer — "Retrieval After RAG: Hybrid Search, Agents, and Database Design" (Latent Space) — https://turbopuffer.com/blog/podcast-latent-space — 2026-03-12
**STATUS:** shipped-in-production

**CLAIM:** Pinecone re-architected for agent workloads specifically, characterizing them as "Millions of namespaces," "fewer than 100k vectors" per namespace, and "sporadic and bursty query patterns." Adaptive LSM-tree indexing on write, blob-storage persistence with on-demand fetch/cache on read → "response times of approximately 10ms" for small-namespace linear scans; >100M vectors at ">1000 QPS."
**SOURCE:** Pinecone — "Optimizing Pinecone for agents (and more)" — https://www.pinecone.io/blog/optimizing-pinecone/ — 2025-03-17
**STATUS:** shipped-in-production

**CLAIM:** Databricks reports multi-step search agents beat single-step RAG with numbers: using their Knowledge Assistant as a tool inside a multi-step search agent beats RAG-as-a-tool by **over 30%** while *decreasing* time-to-completion by **8%**; "multi-step search agents are consistently more effective than single-step retrieval workflows." Their Instructed Retriever claims **~70%** over simplistic RAG, **~15%** over DIY reranking pipelines, and **35–50%** higher recall on StaRK-Instruct. Architectural argument: RAG "loses context after initial retrieval"; the fix is propagating "complete system specifications — from instructions to examples and index schema — through every stage of the search pipeline."
**SOURCE:** Databricks — "Instructed Retriever: Unlocking System-Level Reasoning in Search Agents" — https://www.databricks.com/blog/instructed-retriever-unlocking-system-level-reasoning-search-agents — 2026-01-06
**STATUS:** GA (shipped in Agent Bricks Knowledge Assistant)

**CLAIM:** Chroma's argument for why fixed pipelines fail specifically on multi-hop: single-pass retrieval "assumes that the information needed to answer a question can be retrieved in a single pass," whereas multi-hop needs "a chain of intermediate searches in which the output of one search informs the next."
**SOURCE:** Chroma — "Chroma Context-1: Training a Self-Editing Search Agent" — https://www.trychroma.com/research/context-1 — 2026-03-26
**STATUS:** research (Apache 2.0 weights, not a hosted product)

### 1.4 The counter-case: when agentic loops lose

**CLAIM:** Voyage published data that once first-stage retrieval is strong, putting an LLM in the loop as a ranker makes things *worse*: "Qwen 3 32B and Gemini 2.0 Flash actually degrade performance, with NDCG@10 dropping to 80.63% and 79.49%, respectively, from the baseline of 81.58%." LLM rerankers "cost 25-60x more than rerank-2.5" and are up to 48x slower.
**SOURCE:** Voyage AI — "The Case Against LLMs as Rerankers" — https://blog.voyageai.com/2025/10/22/the-case-against-llms-as-rerankers/ — 2025-10-22
**STATUS:** vendor research backing a GA product

**CLAIM:** Elastic ships Agent Builder GA but simultaneously documents that its own reranker preview "is cost prohibitive for high query rates and low query latency requirements," capping CPU reranking at top-30.
**SOURCE:** Elastic — Elastic Rerank docs — https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-rerank — undated
**STATUS:** preview

**CLAIM:** Exa productizes the frontier explicitly rather than picking a side: Fast (sub-500ms) for voice/coding agents, Auto, Deep ("agentic search with multiple sequential queries," "a few seconds of latency"), Deep Max (11–64s) for research. A deep-search MCP tool lets agents call the server **up to 10 times sequentially**.
**SOURCE:** Exa — "Introducing Exa 2.1" — https://exa.ai/blog/exa-api-2-1 — 2025-11-24; "The World's Fastest Search API" — https://exa.ai/blog/fastest-search-api — 2025-07-29
**STATUS:** GA

**CLAIM (Anthropic's own brake):** "Only increasing complexity when needed"; agentic systems "trade latency and cost for better task performance." Workflows give "predictability and consistency for well-defined tasks"; agents are for "when flexibility and model-driven decision-making are needed at scale." Bottom line: "Success in the LLM space isn't about building the most sophisticated system. It's about building the *right* system for your needs."
**SOURCE:** Anthropic — "Building Effective AI Agents" — https://www.anthropic.com/engineering/building-effective-agents — 2024-12-19
**STATUS:** foundational guidance, never retracted

---

## SECTION 2 — ANTHROPIC & OPENAI PUBLISHED GUIDANCE

### 2.1 Anthropic — Contextual Retrieval (the pre-agentic baseline, still live)

**CLAIM:** Contextual Retrieval prepends a 50–100 token LLM-generated situating blurb to each chunk before embedding *and* before BM25 indexing. Prompt used verbatim: "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else." Generated with Claude 3 Haiku.
**Numbers:** contextual embeddings alone → **35%** reduction in top-20 retrieval failure rate (5.7% → 3.7%); + contextual BM25 → **49%** (5.7% → 2.9%); + reranking → **67%** (5.7% → 1.9%). Pipeline shape: retrieve top-150 → rerank to top-20 → generate. Top-20 beat top-5 and top-10. One-time cost **$1.02 per million document tokens** with prompt caching.
**SOURCE:** Anthropic — "Introducing Contextual Retrieval" — https://www.anthropic.com/news/contextual-retrieval (also /engineering/contextual-retrieval) — 2024-09-19
**STATUS:** shipped technique / cookbook

**CLAIM:** Anthropic's own "don't build RAG" threshold: "If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt… with no need for RAG or similar methods," citing prompt caching as making this viable.
**SOURCE:** same as above — 2024-09-19
**STATUS:** shipped guidance

### 2.2 Anthropic — Context engineering (the pivot)

**CLAIM:** Definition: context engineering is "the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts." Prompt engineering is the discrete-task subset; context engineering is iterative, running every turn.
**CLAIM:** Context is "a finite resource with diminishing marginal returns." Mechanism given: transformers create "n² pairwise relationships for n tokens," and "models develop their attention patterns from training data distributions where shorter sequences are typically more common than longer ones." Outcome is a gradient, not a cliff — "reduced precision for information retrieval and long-range reasoning."
**CLAIM — the retrieval pivot:** a documented field shift from embedding-based pre-inference retrieval to **"just in time"** context. Verbatim: "Rather than pre-processing all relevant data up front, agents built with the 'just in time' approach maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools." Claude Code "uses targeted queries, store results, and leverage Bash commands like head and tail to analyze large volumes of data without ever loading the full data objects into context." This enables "progressive disclosure — … allows agents to incrementally discover relevant context through exploration."
**CLAIM — the hybrid hedge:** "the most effective agents might employ a hybrid strategy, retrieving some data up front for speed, and pursuing further autonomous exploration at its discretion," recommended for "contexts with less dynamic content, such as legal or finance work." And: "as model capabilities improve, agentic design will trend towards letting intelligent models act intelligently, with progressively less human curation." Bottom line: "do the simplest thing that works."
**CLAIM — sub-agent economics:** each subagent "explore[s] extensively, using tens of thousands of tokens or more, but return[s] only a condensed, distilled summary" — "typically 1,000-2,000 tokens."
**CLAIM — compaction craft:** "Start by maximizing recall to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve precision."
**CLAIM — overarching principle:** "find the smallest set of high-signal tokens that maximize the likelihood of your desired outcome."
**SOURCE:** Anthropic — "Effective context engineering for AI agents" — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents — 2025-09-29 (Rajasekaran, Dixon, Ryan, Hadfield)
**STATUS:** guidance backed by shipped features

**CLAIM (the nuanced code-search position, often mis-quoted):** "Semantic search is usually faster than agentic search, but less accurate, more difficult to maintain, and less transparent… we suggest starting with agentic search, and only adding semantic search if you need faster results."
**SOURCE:** same post — 2025-09-29

### 2.3 Anthropic — tool design for retrieval

**CLAIM:** Consolidate rather than proliferate: "Instead of implementing a `read_logs` tool, consider implementing a `search_logs` tool which only returns relevant log lines and some surrounding context." Tools "should take care to return only high signal information back to agents," with "pagination, range selection, filtering, and/or truncation with sensible default parameter values." Claude Code caps tool responses at **25,000 tokens** by default.
**CLAIM:** Natural-language identifiers beat UUIDs — "merely resolving arbitrary alphanumeric UUIDs to more semantically meaningful and interpretable language… significantly improves Claude's precision in retrieval tasks by reducing hallucinations."
**CLAIM:** `response_format` enum (concise vs detailed) — Slack example: concise used ~1/3 the tokens (72 vs 206).
**CLAIM:** Namespace by service and resource (`asana_search`, `asana_projects_search`). And: "If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better."
**SOURCE:** Anthropic — "Writing effective tools for AI agents—using AI agents" — https://www.anthropic.com/engineering/writing-tools-for-agents — 2025-09-11
**STATUS:** guidance

### 2.4 Anthropic — shipped retrieval-adjacent API surface (status matters here)

| Feature | Mechanism | Status (Aug 2026) | Key numbers |
|---|---|---|---|
| **Web search tool** | Server-side loop; "The API runs the searches… This process can repeat multiple times throughout a single request." `max_uses` hard cap. Citations always on. | **GA** | **$10 per 1,000 searches**. "Simple factual queries typically use 1–3 searches; comparative or multientity research can use 10 or more." `cited_text` (≤150 chars), `title`, `url` don't count toward tokens. |
| **Web search dynamic filtering** (`web_search_20260209`+) | Claude writes and runs code inside code execution that **filters results before they enter context**; `allowed_callers` defaults to `["code_execution_20260120"]` | **GA** | No extra charge for the code execution calls |
| **Web search response inclusion** (`web_search_20260318`+) | `response_inclusion: "excluded"` drops raw search blocks from the API response for agentic workflows | **GA** | Reduces output token cost |
| **Web fetch tool** | Claude "is not allowed to dynamically construct URLs" — can only fetch URLs already in conversation or from prior search results (`url_not_in_prior_context` error) | **GA** | Free beyond tokens. 10 kB page ≈ 2,500 tokens; 500 kB PDF ≈ 125,000 tokens |
| **Search result content blocks** | `{"type":"search_result", source, title, content[], citations:{enabled}}` — from tool calls or top-level user content; gives custom RAG the same citation quality as web search | **GA, no beta header**, all active models except Haiku 3 | — |
| **Citations** | Sentence-chunked for PDFs/plain text; custom content blocks used as-is. Returns `char_location` / `page_location` / `content_block_location` | **GA**, all active models | `cited_text` **does not count toward output tokens**, and not toward input tokens when passed back. Anthropic: "the citations feature is significantly more likely to cite the most relevant quotes from documents than purely prompt-based approaches" |
| **Memory tool** (`memory_20250818`) | Client-side file CRUD under `/memories`: view, create, str_replace, insert, delete, rename. Framed as "just-in-time context retrieval." | **GA, no beta header**, all Claude 4+ | API auto-injects: "ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE… ASSUME INTERRUPTION" |
| **Context editing** (`clear_tool_uses_20250919`, `clear_thinking_20251015`) | Server-side clearing of oldest tool results; `trigger` default 100k input tokens, `keep` default 3 tool uses, `clear_at_least`, `exclude_tools`, `clear_tool_inputs` | **Beta** (`context-management-2025-06-27`) | Launch numbers: **39%** improvement on internal agentic search eval (memory+editing), **29%** editing alone, **84%** token reduction in a 100-turn web search eval |
| **Compaction** (`compact_20260112`) | Server-side whole-conversation summarization; default trigger **150,000** input tokens, min 50,000; API drops all blocks before the compaction block | **Beta** (`compact-2026-01-12`) | Anthropic calls it "the recommended strategy for managing context in long-running conversations and agentic workflows" |
| **Tool Search Tool** (`defer_loading: true`) | Retrieval applied to tool *definitions* | **Beta** (`advanced-tool-use-2025-11-20`) | ~77K → ~8.7K tokens, "**85% reduction**… preserving 95% of context window." MCP eval accuracy: Opus 4 **49% → 74%**; Opus 4.5 **79.5% → 88.1%** |
| **Programmatic Tool Calling** | Claude writes Python that calls tools; intermediate results never enter context | **Beta** | **43,588 → 27,297 tokens (37%)** on complex research; knowledge retrieval **25.6% → 28.5%**; GIA **46.5% → 51.2%** |
| **Tool Use Examples** (`input_examples`) | Concrete usage patterns beyond JSON Schema | **Beta** | accuracy **72% → 90%** on complex parameter handling |
| **Agent Skills** | Three-level progressive disclosure: L1 name+description in system prompt (~100 tokens/skill), L2 SKILL.md body on trigger (<5k tokens), L3 bundled files at zero token cost until read | **GA** (API beta header `skills-2025-10-02`) | "the amount of context that can be bundled into a skill is effectively unbounded" |

**SOURCES:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool · /web-fetch-tool · /memory-tool · https://platform.claude.com/docs/en/build-with-claude/search-results · /citations · /context-editing · /compaction · https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview · https://www.anthropic.com/engineering/advanced-tool-use (2025-11-24) · https://claude.com/blog/context-management (2025-09-29) · https://claude.com/blog/skills (2025-10-16) — all accessed 2026-08-08

### 2.5 Anthropic — MCP and the tool-token tax

**CLAIM:** "58 tools consuming approximately 55K tokens before the conversation even starts"; agents with thousands of tools must process "hundreds of thousands of tokens before reading a request." Second problem: intermediate results pass through context twice — a two-hour Google Drive transcript into Salesforce "potentially adding 50,000 tokens."
**CLAIM — the fix:** present MCP servers as code APIs on a filesystem (`./servers/google-drive/getDocument.ts`); "models are great at navigating filesystems. Presenting tools as code on a filesystem allows models to read tool definitions on-demand, rather than reading them all up-front." Result: "**150,000 tokens to 2,000 tokens — a time and cost saving of 98.7%**." Filtering a 10,000-row spreadsheet "to five rows instead of 10,000" in the execution environment. `search_tools` supports detail levels ("name only, name and description, or the full definition with schemas").
**SOURCE:** Anthropic — "Code execution with MCP: building more efficient AI agents" — https://www.anthropic.com/engineering/code-execution-with-mcp — 2025-11-04 (Adam Jones, Conor Kelly)
**STATUS:** OSS pattern / recommendation

**PRIORITY FLAG:** Cloudflare published the same thesis **39 days earlier** with a different rationale — a *training-data* argument: LLMs have seen enormous amounts of real TypeScript but comparatively little synthetic tool-call syntax. "If you present an LLM with too many tools, or overly complex tools, it may struggle to choose the right one"; "the output of each tool call must feed into the LLM's neural network, just to be copied over to the inputs of the next call, wasting time, energy, and tokens." Cloudflare does not cite Anthropic or prior work.
**SOURCE:** Cloudflare — "Code Mode: the better way to use MCP" — https://blog.cloudflare.com/code-mode/ — 2025-09-26 (Kenton Varda, Sunil Pai)
**STATUS:** shipped (Workers/Agents SDK)

**CLAIM:** MCP tool search is now **default-on** in Claude Code: "MCP tool definitions are deferred by default and loaded on demand via tool search, so only tool names consume context until Claude uses a specific tool."
**SOURCE:** Anthropic — "How Claude Code works" — https://code.claude.com/docs/en/how-claude-code-works
**STATUS:** shipped default

**CLAIM (quiet qualification of MCP, from MCP's own author):** "CLI tools are the most context-efficient way to interact with external services" — recommends installing `gh` rather than going through the GitHub API.
**SOURCE:** Anthropic — Claude Code best practices — https://code.claude.com/docs/en/best-practices
**STATUS:** shipped guidance

**CLAIM (MCP spec):** MCP models retrieval two ways — **resources** (application-driven, URI-addressed, with `resources/list`, `resources/read`, URI templates, optional `subscribe`/`listChanged`, and annotations carrying `audience`/`priority`/`lastModified`) and **tools** (model-driven). The spec mandates neither for search; in practice the ecosystem converged on tools. Notably, **OpenAI's deep research API imposes the tool contract**: remote MCP servers must implement "A `search` tool that takes a query and returns search results. A `fetch` tool that takes an id from the search results and returns the corresponding document."
**SOURCES:** MCP spec — https://modelcontextprotocol.io/specification/2025-06-18/server/resources — rev 2025-06-18 · OpenAI — https://developers.openai.com/api/docs/guides/deep-research
**STATUS:** standard / GA

**CLAIM:** Anthropic's Managed Agents virtualizes sessions (append-only event logs), harnesses (orchestration loops), and sandboxes. Retrieval note: all tools use a standardized `execute(name, input) → string` interface covering custom tools, MCP servers, and Anthropic's own. Context is retrievable by positional slice via `getEvents()`; "context can be an object in a REPL that the LLM programmatically accesses by writing code to filter or slice it." OAuth tokens live in a vault outside the sandbox behind an MCP proxy so "the harness is never made aware of any credentials."
**SOURCE:** Anthropic — "Scaling Managed Agents: Decoupling the brain from the hands" — https://www.anthropic.com/engineering/managed-agents — 2026-04-08
**STATUS:** hosted service, available

### 2.6 OpenAI — retrieval guidance and hosted retrieval defaults

**CLAIM — `file_search` / vector store defaults (the most-copied numbers in the industry):** `max_chunk_size_tokens` = **800**, `chunk_overlap_tokens` = **400** (the "auto" strategy uses exactly these). Configurable range 100–4,096; overlap must not exceed half the chunk size. Embedding model referenced: `text-embedding-3-small`.
**CLAIM:** Shipped query-understanding and ranking knobs: `rewrite_query=true` (rewritten form returned in `search_query`), `ranking_options` with `ranker` (`auto` / `default-2024-08-21`) and `score_threshold` (0.0–1.0), hybrid tuning via `embedding_weight` and `text_weight`, attribute filtering (max 16 keys, 256 chars, `eq/ne/gt/gte/lt/lte/in/nin` + `and`/`or`). Standalone search endpoint returns 10 by default, max 50. Limits: 512 MB and 5M tokens per file. Storage: first 1 GB free, then **$0.10/GB/day**.
**SOURCE:** OpenAI — Retrieval guide — https://developers.openai.com/api/docs/guides/retrieval — accessed 2026-08-08; File search tool — https://developers.openai.com/api/docs/guides/tools-file-search; Vector stores API ref — https://developers.openai.com/api/docs/api-reference/vector-stores/create
**STATUS:** GA
*Note: the docs advertise "semantic and keyword search" but publish no detail on the hybrid mechanics or reranker internals.*

**CLAIM — `web_search` as an orchestration spectrum:** three explicitly named tiers — "non-reasoning web search" (quick lookup), "agentic search with reasoning models," and "deep research… using hundreds of sources." `search_context_size` low/medium/high tunes how much result content enters context but "does not set an exact token count or guarantee a specific number of sources." Web search context is capped at **128k even when the model's context window is larger**. Up to **100** `allowed_domains` or `blocked_domains`. A `sources` field returns every URL consulted — "typically exceeds the number of inline citations." Inline citations "must be made clearly visible and clickable."
**SOURCE:** OpenAI — Web search tool guide — https://developers.openai.com/api/docs/guides/tools-web-search — accessed 2026-08-08
**STATUS:** GA

**CLAIM — GPT-5-era prompting guidance on bounding agentic search.** OpenAI ships prompt patterns for calibrating "agentic eagerness." To reduce search: lower `reasoning_effort` ("reduces exploration depth but improves efficiency and latency"); set early-stop criteria ("You can name exact content to change. Top hits converge (~70%) on one area/path."); impose a fixed tool budget ("Usually, this means an absolute maximum of 2 tool calls. If you think you need more time to investigate, update the user."); and give an escape hatch ("even if it might not be fully correct"). The `<context_gathering>` block: "Start broad, then fan out to focused subqueries. In parallel, launch varied queries; read top hits per query… Avoid over searching for context." / "Batch search → minimal plan → complete task. Search again only if validation fails or new unknowns appear. **Prefer acting over more searching.**"
**SOURCE:** OpenAI Cookbook — GPT-5 prompting guide — https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide — no date printed on page
**STATUS:** GA guidance
*This is the closest thing OpenAI has to Anthropic's "start wide then narrow" — and it points the opposite direction: bias toward less search.*

**CLAIM:** OpenAI Agents SDK compaction is a session wrapper that **does not summarize**: `OpenAIResponsesCompactionSession` "clears and rewrites history rather than summarizing it." Other levers are `SessionSettings(limit=N)`, `session_input_callback`, `pop_item()`. Session backends: SQLite, AsyncSQLite, OpenAIConversations, Redis, SQLAlchemy, MongoDB, Dapr, AdvancedSQLite, EncryptedSession.
**SOURCE:** OpenAI — Agents SDK Sessions — https://openai.github.io/openai-agents-python/sessions/
**STATUS:** shipped — *the weakest compression story of the four majors: no summarization node, no tool-result pruning primitive.*

**CLAIM — ChatGPT company knowledge:** "powered by a version of GPT-5 that's trained to look across multiple sources to give more comprehensive and accurate answers." Streams intermediate looking-at steps; returns the specific snippets used with citations; respects existing company permissions. Apps must expose **File Search plus `search` and `fetch` actions** to be eligible. Connectors were renamed "apps" on 2025-12-17. Connectors are "access connectors, which fetch content when a user asks a question, built using MCP."
**SOURCE:** OpenAI Help Center — "Company knowledge in ChatGPT (Business, Enterprise, and Edu)" — https://help.openai.com/en/articles/12628342-company-knowledge-in-chatgpt-business-enterprise-and-edu — `[snippet-only; 403 to fetch]`
**STATUS:** shipped-in-production

---

## SECTION 3 — WHAT PRODUCTION SYSTEMS USE AT EACH STAGE

### 3.1 Query understanding / decomposition / routing

**CLAIM:** Weaviate Query Agent went **GA 2025-09-17** after ~6 months of preview. Shipped feature list *is* a query-understanding pipeline as a service: query planning and cross-collection routing, decomposition into concurrent searches, dynamic schema-valid filter construction, query expansion, reranking/aggregation, answer citation. Two modes: Ask (with generation) and Search (retrieval-only).
**SOURCE:** Weaviate — "Accelerating Data Workflows with Query Agent, now GA" — https://weaviate.io/blog/query-agent-generally-available — 2025-09-17
**STATUS:** GA

**CLAIM:** Weaviate published head-to-head numbers for agentic Search Mode vs plain hybrid (Arctic 2.0 dense + BM25 via RRF): overall **+17% Success@1, +11% Recall@5**. Natural Questions Success@1 0.43 → 0.52, Recall@5 0.70 → 0.81; EnronQA Success@1 0.56 → 0.74; **BRIGHT Biology Success@1 0.13 → 0.44**, Recall@5 0.11 → 0.35. Credited to "query expansion, query decomposition, schema introspection, and reranking." **No latency reported.**
**SOURCE:** Weaviate — "Search Mode Benchmarking" — https://weaviate.io/blog/search-mode-benchmarking — 2025-09-23
**STATUS:** GA benchmark

**CLAIM:** Elastic Agent Builder reached **GA** (Cloud Serverless and 9.3) with query planning as a first-class capability: "generate[s] optimized hybrid, semantic, and structured queries" from natural language, automatic index selection, MCP exposure. Pitch is consolidation — removes the need for "separate data stores, vector databases, RAG pipelines, search layers, query translators, and tool orchestrators."
**SOURCE:** Elastic — "Agent Builder now GA: Ship context-driven agents in minutes" — https://www.elastic.co/search-labs/blog/agent-builder-elastic-ga — 2026-01-22
**STATUS:** GA

**CLAIM:** Azure AI Search agentic retrieval is a four-stage pipeline: (1) app calls a knowledge base with query + conversation history; (2) **query planning** — an LLM decomposes into focused subqueries; (3) **query execution** — "All subqueries run simultaneously," each keyword/vector/hybrid, each **semantically reranked (L2) independently**, references retained for citation; (4) synthesis into a unified response plus an execution activity log. Reasoning effort is an explicit knob: `minimal` **skips query planning entirely**; `low` (default) and `medium` invoke the planner.
**SOURCE:** Microsoft — "Agentic retrieval in Azure AI Search" — https://learn.microsoft.com/en-us/azure/search/search-agentic-retrieval-concept — ms.date 2026-06-02, updated 2026-07-02
**STATUS:** **GA in the 2026-04-01 REST API**; Azure portal and Foundry portal remain **preview-only**

**CLAIM — Microsoft's published cost model for agentic retrieval:** worked example assumes **3 subqueries per plan**, **50 chunks reranked per subquery**, 500 tokens/chunk, 2,000 input tokens of chat history, 350-token output plan. For 2,000 retrievals: $3.30 Azure AI Search reranking + $1.02 Azure OpenAI planning = **$4.32**. Billing shifts from per-query (classic) to **per-token** (agentic). Stated plainly: "Agentic retrieval adds latency compared to a single-query pipeline." Cost guidance: "Lower the reasoning effort… Reduce the number of knowledge sources (indexes); consolidating content can lower **fan-out**."
**SOURCE:** same as above
**STATUS:** GA

**CLAIM:** Elastic separately published an agentic-search-plus-autotuning reference architecture: an LLM agent fills search templates (V1–V4) from natural language; an XGBoost Learn-to-Rank model with **48 features** (property attributes + engagement signals) reranks, retrained from logged interactions.
**SOURCE:** Elastic — "Agentic search: autotuning relevance in Elasticsearch" — https://www.elastic.co/search-labs/blog/agentic-search-relevance-autotuning-elasticsearch — 2025-11-19
**STATUS:** reference architecture, not a SKU

**CLAIM:** Jason Liu argues query understanding should surface *facets and metadata*, not just top-k chunks — "Agent Peripheral Vision: Providing agents with structured metadata about the broader information space beyond just the top-k results." Client numbers cited (unaudited): 90% reduction in clarification questions, 75% reduction in expert escalations, 4x improvement in resolution times.
**SOURCE:** Jason Liu — https://jxnl.co/writing/2025/08/27/facets-context-engineering/ — 2025-08-27
**STATUS:** practitioner field report

**NEGATIVE FINDING:** No first-party vendor page was found shipping **HyDE** as a named GA feature. Vendors ship query expansion, decomposition, rewriting, and planning — not HyDE by name. Flagged as under-searched rather than proven absent.

### 3.2 Hybrid retrieval — and the RRF-vs-weighted split

**CLAIM:** Across every embedding model Vespa tested, hybrid beat semantic-only: "Every single model scored higher with hybrid retrieval than semantic-only. On average, the best hybrid method beats semantic-only by **3-5 percentage points**." Storage for 100M × 768-dim: FP32 307 GB / FP16 154 GB / INT8 77 GB / binary 9.6 GB (32x). "Vespa can do ~1 billion hamming distance calculations per second, roughly 7x more than prenormalized angular distance." INT8 on CPU "2.7-3.4x faster while keeping 94-98% of the quality"; INT8 on GPU is *4-5x slower* than FP32.
**SOURCE:** Vespa — "Embedding Tradeoffs, Quantified" — https://blog.vespa.ai/embedding-tradeoffs-quantified/ — 2026-01-14
**STATUS:** benchmark on GA features

**CLAIM — vendors disagree on fusion.** Weaviate moved its default **off RRF**: `relativeScoreFusion` (min-max normalize, then weighted sum) is default from v1.24; `rankedFusion` (pure RRF) was default in ≤v1.23. Docs: "it retains more information from the original searches than `rankedFusion`, which only retains the rankings." Server default `alpha = 0.75` (vector-leaning).
**SOURCE:** Weaviate — Hybrid search concepts — https://docs.weaviate.io/weaviate/concepts/search/hybrid-search; "Hybrid Search Explained" — https://weaviate.io/blog/hybrid-search-explained — 2025-01-27
**STATUS:** GA

**CLAIM (opposite):** Qdrant calls RRF "the de facto standard in the field" and documents only RRF.
**SOURCE:** Qdrant — "Hybrid Search with Qdrant's Query API" — https://qdrant.tech/articles/hybrid-search/ — 2024-07-25
**STATUS:** GA

**CLAIM (hedge):** MongoDB GA'd both — `$rankFusion` (RRF over ranks) on 8.0+, `$scoreFusion` (normalized weighted average) on 8.2+. Customer quote: "This has improved the context retrieval accuracy for our Eddy AI chatbot by 30%" (Kovai.co).
**SOURCE:** MongoDB — https://www.mongodb.com/company/blog/product-release-announcements/boost-search-relevance-mongodb-atlas-native-hybrid-search — 2025-06-25, updated 2026-06-30
**STATUS:** GA

**CLAIM (third position — skip fusion, rerank instead):** Pinecone's cascading retrieval yields "up to 48% better performance — and 24% better, on average" over dense alone, and "**8% better than score fusion**" on BEIR. Their learned sparse model `pinecone-sparse-english-v0` gives "Up to 44% (average 23%) better NDCG@10" on TREC DL and "up to 24% (8% on average)" on BEIR vs BM25.
**SOURCE:** Pinecone — "Introducing cascading retrieval" — https://www.pinecone.io/blog/cascading-retrieval/ — 2024-12-02
**STATUS:** GA

**CLAIM:** Pinecone positions SPLADE as not production-ready — word-piece SPLADE embeddings are "still new and highly experimental." Sparse storage is "1000x smaller (and cheaper)" than dense at 100M scale.
**SOURCE:** Pinecone — "Don't be dense: Launching sparse indexes in Pinecone" — https://www.pinecone.io/learn/sparse-retrieval/ — 2025-03-05
**STATUS:** preview at time of writing

**CLAIM:** Elastic's default semantic path is **sparse-neural, not dense**: `semantic_text` with no inference endpoint specified uses ELSER by default. Hybrid retrievers support "linear/generic rescoring alongside Reciprocal Rank Fusion (RRF)." BBQ GA'd with "up to 5x faster queries and 3.9x higher throughput" vs OpenSearch FAISS.
**SOURCE:** Elastic — "What's new in Elastic 9.0 / 8.18" — https://www.elastic.co/blog/whats-new-elastic-search-9-0-0 — 2025-04-14
**STATUS:** GA

**CLAIM — BM25 is being re-invested in *because of* agents:** turbopuffer's FTS v2 claims "up to 20x better full-text search performance" via a 10x smaller on-disk index plus MAXSCORE dynamic pruning. On ~5M Wikipedia docs, k=100: "lord of the rings" 75ms → 6ms; multi-term 174ms → 20ms. Rationale verbatim: "full-text search is equally important for recall and performance in agent-initiated queries," since agents write longer queries than humans.
**SOURCE:** turbopuffer — "FTS v2: up to 20x faster full-text search" — https://turbopuffer.com/blog/fts-v2 — 2026-02-03
**STATUS:** shipped-in-production

**CLAIM — a reference agent-memory retrieval stack with real numbers:** RRF over BM25 + Jina v5 dense (over-fetching 80 candidates per leg, `rank_constant=30`), then a Jina v2 cross-encoder on the merged pool; a single unified `recall_memory` tool spanning three memory indices; pre-recall on verbatim user messages to bypass LLM paraphrasing; `refresh=True` on episodic writes. Over 168 questions: **R@10 = 0.89** (0.85–0.893 across four runs), R@5 0.75; semantic facts R@10 ≈ 0.81, episodic 0.98, procedural 1.0; **zero cross-tenant leaks** under Elasticsearch DLS per-user API keys.
**SOURCE:** Elastic Search Labs — "Agent memory on Elasticsearch: hybrid retrieval and DLS" — https://www.elastic.co/search-labs/blog/agent-memory-elasticsearch — 2026-06-16
**STATUS:** reference implementation with documented benchmark

### 3.3 Late interaction / ColBERT / ColPali / MUVERA — production status is vendor-dependent

| Vendor | Position | Status | Numbers |
|---|---|---|---|
| **Vespa** | Verbatim FAQ: "Is Long-ColBERT in Vespa ready for production? **Yes.**" `tensor<int8>(context{}, token{}, v[16])`, 16 bytes/token vector | shipped-in-production | Reranks top-10 "below 50ms," significant nDCG@10 gain over BM25 |
| **Vespa (ColPali)** | 1 PDF page = 1,030 vectors × 128 dims. Binary quantization to 16-dim int8 → **32x storage reduction**, hamming MaxSim "~3.5x faster than float dot product," "~200M 128-bit hamming distances per second per CPU core" | shipped | DocVQA nDCG@5: float-float 52.4, binary-binary 49.5, **binary + float rerank 51.6** |
| **Weaviate** | MUVERA **GA in 1.31+**. Formula `repetitions * 2^ksim * dprojections` (defaults 4/16/10). Multi-vectors support PQ/BQ/RQ/SQ | **GA** | Memory −~70%; 12GB → <1GB; import 20+ min → 3–6 min (110k objects). **Candid recall cliff: needs ef>512 for 80%+, ef 2048 for >90%**, "decreasing the query throughput" |
| **Qdrant** | MUVERA is **in FastEmbed 0.7.2+, not in the engine**. Late interaction recommended **only as a reranker, never first-stage** (store multivectors with HNSW `m=0`) | client library only | Full multivector 1.27s → MUVERA-only 0.15s (~8x) → MUVERA+rerank 0.18s. NDCG@10 0.347 → 0.343; MUVERA alone recovers only ~70% of quality |
| **Elastic** | "ColPali and ColBERT now supported with MaxSim" | GA per release blog | — |
| **Pinecone** | **No native multi-vector support.** Own research stores ConstBERT embeddings "as metadata." Verdict: "end-to-end ColBERT is notably slow, even when using the official PLAID engine" (hundreds of ms); as a reranker "MaxSim computation for hundreds of documents takes only a few milliseconds." Cascade: ~1000 → 100 (multi-vector) → 10 (cross-encoder) | research-only | MSMARCO nDCG@10: ColBERT e2e 74.6, ConstBERT e2e 73.1, ConstBERT-rerank 74.4; BEIR avg ColBERT 48.8 vs ConstBERT-rerank 50.2 |
| **Vectara** | Deliberately rejecting it — Boomerang successor is "a single-vector architecture intended to work with conventional vector-search infrastructure without the vector-volume increase associated with patch-level multi-vector retrieval," naming ColPali/VLM2Vec as what it avoids | aspirational / in-development | — |

**SOURCES:** Vespa — https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/ (2024-03-01, Bergum) · https://blog.vespa.ai/scaling-colpali-to-billions/ (2024-09-20, Bergum) · Weaviate — https://weaviate.io/blog/muvera (2025-06-05), https://weaviate.io/blog/weaviate-1-31-release (2025-06-03), https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors · Qdrant — https://qdrant.tech/articles/muvera-embeddings/ (2025-09-05), https://qdrant.tech/articles/hybrid-search/ (2024-07-25) · Elastic — https://www.elastic.co/blog/whats-new-elastic-search-9-0-0 (2025-04-14) · Pinecone — https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/ (2025-05-28) · Vectara — https://www.vectara.com/blog/moving-beyond-text-conversion-the-future-of-enterprise-search (2026-08-05)

**CLAIM:** Ben Clavié / answer.ai published the token-pooling result that made ColBERT storage tractable ("considerable memory & disk footprint reduction"), plus `answerai-colbert-small` and the `rerankers` library unifying cross-encoder / ColBERT / LLM ranking under one API.
**SOURCES:** https://www.answer.ai/posts/colbert-pooling.html (2024-06-27) · https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html (2024-08-13) · https://www.answer.ai/posts/2024-09-16-rerankers.html (2024-09-16)
**STATUS:** open-source research/tooling

### 3.4 Reranking

**CLAIM:** Elastic Rerank is a **184M-param DeBERTa-v3 cross-encoder** (86M backbone + 98M embedding layer), distilled from a bi-encoder+cross-encoder ensemble over ~3M queries incl. ~180k synthetic pairs. BEIR nDCG@10: BM25 0.426 → **0.565**, vs Cohere v3 0.529 and bge-reranker-v2-gemma (2B) 0.568 — "an average improvement of 39% across the full suite." Per-dataset: NQ 90%, MS MARCO 85%, Climate-FEVER 80%, FiQA-2018 76%.
**CRITICAL STATUS CAVEAT:** still **technical preview** as of current docs; needs a 4GB ML node standalone / "at minimum an 8GB ML node" with ELSER; Elastic's own warning: "the preview version is cost prohibitive for high query rates and low query latency requirements"; "We would recommend shallow reranking for CPU inference: **no more than top-30 results**."
**SOURCES:** Elastic — https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-2 (2024-11-25) · https://www.elastic.co/search-labs/blog/elastic-rerank-model-introduction (2024-12-10) · docs https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-rerank
**STATUS:** preview

**CLAIM:** Voyage rerank-2.5 / rerank-2.5-lite are **instruction-following** with 32K context ("8x that of Cohere Rerank v3.5"), improving retrieval accuracy "by 7.94% and 7.16% over Cohere Rerank v3.5" across 93 datasets, "12.70% and 10.36%" on MAIR.
**SOURCE:** Voyage AI — https://blog.voyageai.com/2025/08/11/rerank-2-5/ — 2025-08-11
**STATUS:** GA

**CLAIM (the sharpest anti-LLM-reranker data):** NDCG@10 — rerank-2.5 **84.32%**, rerank-2.5-lite 83.12%, GPT-5 ~71.71%, Gemini 2.5 Pro ~70.89%, Qwen3-32B ~69.54%. With a strong first stage, LLM rerankers *degrade*: baseline 81.58% → 80.63% (Qwen3-32B) / 79.49% (Gemini 2.0 Flash). Cost: "LLMs cost 25-60x more than rerank-2.5" ($1.25–$3 vs $0.05 per 1M tokens); rerank-2.5 is "9x, 36x, and 48x faster than Claude Sonnet 4.5, GPT-5, and Gemini 2.5 Pro." Also: **listwise sliding-window beats single-pass long-context reranking by 26.6%, 25.27%, and 22.2%**.
**SOURCE:** Voyage AI — https://blog.voyageai.com/2025/10/22/the-case-against-llms-as-rerankers/ — 2025-10-22
**STATUS:** vendor research

**CLAIM:** Jina reranker v3 is **listwise, not pointwise and not late-interaction** — a 0.6B Qwen3-based model doing "causal attention between the query and *all* candidate documents within a single context window," branded "last but not late" in explicit contrast to ColBERT. 131K context, up to 64 docs/pass. BEIR nDCG@10 **61.94**, "outperforming Qwen3-Reranker-4B while being 6× smaller."
**CLAIM:** v3.5 (0.6B) uses hybrid attention ("three sliding-window layers followed by two global layers") + self-distillation: BEIR **63.20** vs Qwen3-Reranker-4B 62.28; "+9.6 nDCG@10 over v3" on Struct-IR; long-context latency 16.1s → **10.3s**; prefill throughput 11.9k → 18.6k tokens/s (A100, FA-2, top-100 listwise, batch 1).
**SOURCES:** Jina AI — https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval/ (2025-10-03) · https://jina.ai/news/jina-reranker-v3-5-faster-listwise-reranking-hybrid-attention-self-distillation (2026-08-03)
**STATUS:** GA

**CLAIM:** Cohere split reranking into quality/latency SKUs: `rerank-v4.0-pro` for "state-of-the-art quality and complex use-cases" and `rerank-v4.0-fast` for "low latency and high throughput use-cases," 32k context, semi-structured (JSON) document support.
**SOURCES:** Cohere — https://docs.cohere.com/changelog/rerank-v4.0 (undated) · https://cohere.com/blog/rerank-4 (2025-12-11, body would not render) · https://docs.cohere.com/docs/rerank
**STATUS:** GA — **date conflict unresolved:** the Rerank 3.5 blog page reported 2024-12-02 on fetch while a search snippet claimed 2025-07-10. No v4.0 benchmark numbers were verifiable.

**CLAIM:** Contextual AI ships an instruction-following reranker family at 1B/2B/6B (+ NVFP4 quantized), claiming "~35% increase in recency-awareness with our quantized 2B reranker compared to the second-best reranker" and beating all comparators on TREC 2025 Product Search "at a superior throughput, latency, and cost." Open weights on HuggingFace. Platform benchmark: reranker **61.2 on BEIR**, "outperforming the next best solution (Voyage-v2 at 58.3) by 2.9%"; full RAG agent "71.2% performance, a 5.4% improvement over the strongest baseline" (Cohere retrieval + Claude-3.5-Sonnet at 66.8%).
**SOURCES:** https://contextual.ai/blog/rerank-v2 (2025-08-27) · https://contextual.ai/blog/introducing-instruction-following-reranker (2025-03-11) · https://contextual.ai/blog/platform-benchmarks-2025 (2025-01-15)
**STATUS:** GA + open weights (vendor self-eval)

**CLAIM:** MongoDB shipped a **native reranker inside the database** — Voyage AI rerank-2.5, 32K context, "improves retrieval accuracy by up to 30%," public preview on MongoDB 8.3. Rationale is explicitly agentic: "In an agentic workflow, retrieval is iterative… A weak result does not just hurt one response; it can send the next step off course and drive up cost."
**SOURCE:** MongoDB — https://www.mongodb.com/company/blog/product-release-announcements/improving-agent-retrieval-native-reranking-hybrid-search — 2026-07-02
**STATUS:** preview

**CLAIM:** Elastic's `text_similarity_reranker` retriever is the shipped mechanism (nested first-stage retriever + inference endpoint + `rank_window_size` + `min_score`). Elastic's stated value case is *calibrated scores enabling cutoffs*, not just ordering.
**SOURCE:** Elastic — https://www.elastic.co/search-labs/blog/semantic-reranking-with-retrievers — 2024-05-28
**STATUS:** GA

**CLAIM (cost warning):** Qdrant declines to endorse a default: "reranking can be slow. Processing millions of documents can take hours, which is why rerankers focus on refining results, not searching through the entire document collection." Names cross-encoder, ColBERT-multivector, and LLM rerankers as three valid families.
**SOURCE:** Qdrant — https://qdrant.tech/documentation/search-precision/reranking-semantic-search/ — undated
**STATUS:** documentation guidance

### 3.5 Chunking, context compression, pruning

**CLAIM:** Chroma's chunking eval found chunker choice moves recall by **up to 9%**, and smaller chunks win. ClusterSemanticChunker @200 tokens → recall 87.3%, precision 8.0%, IoU 8.0%; LLMChunker (GPT-4o) → recall 91.9% but precision/IoU 3.9%; RecursiveCharacterTextSplitter @200 no-overlap → recall 88.1%. Explicit callout: **OpenAI's documented default (800 tokens / 400 overlap) produced "slightly below-average recall and the lowest scores across all other metrics."**
**SOURCE:** Chroma — "Evaluating Chunking Strategies for Retrieval" — https://www.trychroma.com/research/evaluating-chunking — 2024-07-03
**STATUS:** vendor research report
*Direct contradiction: Elastic's `semantic_text` default is "250 words (approximately 400 tokens)"; OpenAI's is 800/400.*

**CLAIM:** Jina's **late chunking** = encode the whole document with a long-context encoder first, then pool per chunk; no LLM in the loop. BeIR nDCG gains over naive chunking: NFCorpus 23.46 → 29.98, SciFact 64.20 → 66.10, TRECCOVID 63.36 → 64.70, FiQA2018 33.25 → 33.84. "The longer the document, the more effective the late chunking strategy becomes."
**SOURCE:** Jina AI — https://jina.ai/news/late-chunking-in-long-context-embedding-models/ — 2024-08-22
**STATUS:** GA (in the jina-embeddings API)

**CLAIM — direct vendor-vs-vendor attack on Anthropic:** Jina explicitly frames Anthropic's contextual retrieval as inferior engineering — "a brute-force approach" where "each chunk is sent to the LLM along with the full document" — and claims late chunking has "No additional storage since the embedding size remains the same," is "Significantly faster than using an LLM to generate enrichment," and is "highly resilient to boundary cues" whereas Anthropic's method "relies on accurate and readable chunks."
**SOURCE:** Jina AI — Han Xiao — "What Late Chunking Really Is & What It's Not: Part II" — https://jina.ai/news/what-late-chunking-really-is-and-what-its-not-part-ii/ — 2024-10-03
**STATUS:** vendor position piece
*No vendor published a head-to-head of late chunking vs contextual retrieval under matched conditions. No first-party vendor page was found shipping Anthropic-style contextual chunk augmentation as a GA feature.*

**CLAIM — the empirical basis everyone cites for pruning:** Chroma's context rot study across 18 models (Anthropic Opus 4 / Sonnet 4 / 3.7 / 3.5 / Haiku 3.5; OpenAI o3, GPT-4.1 family, GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo; Google Gemini 2.5 Pro/Flash, 2.0 Flash; Alibaba Qwen3 235B-A22B / 32B / 8B) concludes models "do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows." Key retrieval findings: lower needle-question semantic similarity accelerates degradation; "**Even a single distractor reduces performance relative to the baseline**, and adding four distractors compounds this degradation further"; counterintuitively "models perform worse when the haystack preserves a logical flow of ideas"; on LongMemEval, "significantly higher performance on focused prompts compared to full prompts" (~300 tokens vs ~113k). Methodological attack: "long context evaluations for these models often demonstrate consistent performance across input lengths. However, these evaluations are narrow in scope and not representative of how long context is used in practice."
**SOURCE:** Chroma — "Context Rot: How Increasing Input Tokens Impacts LLM Performance" — https://www.trychroma.com/research/context-rot — 2025-07-14 (Kelly Hong, Anton Troynikov, Jeff Huber)
**STATUS:** technical report — *this is the single most-cited empirical justification for reranking, pruning, and top-k discipline in the 2025–26 literature.*

**CLAIM:** Chroma built context pruning *into the retrieval model itself*: **Context-1** is a 20B model (from gpt-oss-20B, SFT + RL) acting as a retrieval subagent that "can selectively discard tangential information" from its own context mid-search. Reported: web 0.88 final-answer-found, legal 0.89, email 0.92, finance 0.64 F1; "up to 10x faster" than frontier models; "400-500 tok/s end to end" on vLLM. Apache 2.0 weights.
**SOURCE:** Chroma — https://www.trychroma.com/research/context-1 — 2026-03-26
**STATUS:** research-only, open weights

**CLAIM:** Weaviate's position: pruning is mandatory — "The worst memory system is the one that faithfully stores everything," requiring "periodic pruning, merging duplicates, deleting outdated facts."
**SOURCE:** Weaviate — "Context Engineering — LLM Memory and Retrieval for AI Agents" — https://weaviate.io/blog/context-engineering — 2025-12-09
**STATUS:** vendor guidance

**CLAIM:** LangChain reimplemented Anthropic's context-editing strategy as portable middleware. `SummarizationMiddleware` (`model`, `trigger` e.g. `("tokens", 4000)`, `keep` e.g. `("messages", 20)`) "persistently updates state by permanently replacing old messages with a summary." `ContextEditingMiddleware` with `ClearToolUsesEdit` "automatically prunes tool results… when the total input token count exceeds configured thresholds." Documented hazard: "When deleting messages, make sure that the resulting message history is valid" — tool results must follow tool calls. *History: at LangChain 1.0 alpha (2025-09-08) middleware shipped with only HITL, summarization, and Anthropic prompt caching — context editing was not in the initial set.*
**SOURCES:** https://docs.langchain.com/oss/python/langchain/short-term-memory · https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware · https://reference.langchain.com/python/langchain/agents/middleware/context_editing/ContextEditingMiddleware · https://www.langchain.com/blog/agent-middleware (2025-09-08)
**STATUS:** stable in LangChain 1.x

**CLAIM:** Claude Code compaction is **tiered**: "It clears older tool outputs first, then summarizes the conversation if needed. Your requests and key code snippets are preserved; detailed instructions from early in the conversation may be lost." Anti-thrash guard: "If a single file or tool output is so large that context refills immediately after each summary, Claude Code stops auto-compacting after a few attempts and shows an error instead of looping." Steerable via `/compact <focus>` and a "Compact Instructions" section in CLAUDE.md.
**SOURCE:** Anthropic — https://code.claude.com/docs/en/how-claude-code-works
**STATUS:** shipped-in-production

**CLAIM:** Anthropic names context as the governing constraint of the whole product: "Most best practices are based on one constraint: Claude's context window fills up fast, and performance degrades as it fills." Named failure mode: "**The infinite exploration.** You ask Claude to 'investigate' something without scoping it. Claude reads hundreds of files, filling the context." Fix: scope narrowly or use subagents — "Since context is your fundamental constraint, subagents are one of the most powerful tools available."
**SOURCE:** Anthropic — https://code.claude.com/docs/en/best-practices
**STATUS:** shipped-in-production

**CLAIM:** Anthropic says compaction alone is **insufficient** for long-horizon work — the model can exhaust context mid-implementation and leave undocumented half-finished features. Recommended harness: initializer agent creating `init.sh`, `claude-progress.txt`, and an initial git commit; then coding agents that read those at session start and update them before ending; a structured JSON feature list with **"over 200 features"** marked pass/fail; "work on only one feature at a time" and self-verify end-to-end (they used Puppeteer MCP) before marking complete. Rule stated: "It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality."
**SOURCE:** Anthropic — "Effective harnesses for long-running agents" — https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents — 2025-11-26 (Justin Young et al.)
**STATUS:** guidance from production

### 3.6 Grounding and citation verification

**CLAIM:** Google's **check-grounding API** is a standalone, shipped verification step: returns "an overall support score of 0 to 1" approximating the fraction of claims grounded in supplied facts, plus claim-level citations mapped by byte position, optional per-claim scores (`enableClaimLevelScore`), and a boolean per claim for whether grounding is even required. Limits: answer ≤4096 tokens, up to **200 facts** at ≤10k chars each; `citation_threshold` defaults to **0.6**. Latency target **<500ms** so it can run inline. Grounding is strict — a mostly-correct claim with one wrong date gets no citation. (Page notes Vertex AI Search is being renamed **Agent Search**.)
**SOURCE:** Google Cloud — "Check grounding with RAG" — https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding — accessed 2026-08-08
**STATUS:** GA

**CLAIM:** "Grounding with Google Search" supports **dynamic retrieval**: Gemini predicts whether a search would help, emitting a 0–1 prediction score; `dynamicRetrievalConfig` threshold defaults to **0.7**, below which no search fires. The search-vs-no-search decision is a *scored gate*, not an agent judgment.
**SOURCE:** Google Cloud — https://cloud.google.com/blog/products/ai-machine-learning/how-vertex-ai-grounding-helps-build-more-reliable-models — `[snippet-only]`
**STATUS:** GA

**CLAIM:** Anthropic runs citation verification as a **separate subagent** — a `CitationAgent` processes documents and the drafted report post-research to identify specific locations for citations, "ensuring all claims are properly attributed to their sources."
**SOURCE:** Anthropic — https://www.anthropic.com/engineering/multi-agent-research-system — 2025-06-13
**STATUS:** shipped-in-production

**CLAIM:** Contextual AI ships **span-level** groundedness scoring, GA: "scores are reported for individual text spans allowing for precise detection of unsupported claims," returned in the API so developers can hide ungrounded claims or add caveats. Separately their Grounded Language Model (GLM) prioritizes retrieved knowledge over parametric knowledge via a `/generate` API.
**SOURCE:** Contextual AI — https://contextual.ai/new/groundedness-scoring-of-model-responses-now-generally-available — `[snippet-only; fetched page rendered nav only]`
**STATUS:** shipped API

**CLAIM:** Azure AI Content Safety groundedness detection ships non-reasoning mode (fast binary for online use), reasoning mode (returns a `reasoning` field explaining ungrounded segments), and a correction feature returning `corrected Text` realigned to sources. Domains MEDICAL / GENERIC; tasks QnA / Summarization.
**SOURCE:** Microsoft — https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness — `[snippet-only]`
**STATUS:** shipped API (preview per quickstart)

**CLAIM:** Vectara's HHEM is the longest-running public hallucination leaderboard: commercial **HHEM-2.3**, open **HHEM-2.1-Open**. Method: 7,700+ curated articles across news/tech/science/medicine/legal/sports/business/education, summarize using only document facts, temperature 0, refusals filtered. As of **2026-05-11**: Finix S1 32B 1.8%, GPT-5.4-nano 3.1%, Gemini 2.5 Flash Lite 3.3%. Measures **factual consistency**, explicitly not summary quality.
**SOURCE:** Vectara — https://github.com/vectara/hallucination-leaderboard — snapshot 2026-05-11
**STATUS:** shipped API + open model + benchmark

**CLAIM:** OpenAI enforces grounding at the **display layer** instead of the model layer: inline citations "must be made clearly visible and clickable," with a separate `sources` list of every URL consulted that "typically exceeds the number of inline citations." OpenAI ships **no faithfulness or groundedness grader** (see §5).
**SOURCE:** OpenAI — https://developers.openai.com/api/docs/guides/tools-web-search
**STATUS:** GA

### 3.7 Memory across turns

| System | Model | Status | Notable |
|---|---|---|---|
| **Anthropic memory tool** | Client-side file CRUD under `/memories`; "just-in-time context retrieval" | **GA**, no beta header | Auto-injected system protocol includes "ASSUME INTERRUPTION." Pairs with compaction: "compaction keeps the active context small…, memory preserves the information that must survive summarization." Path-traversal protection is the integrator's burden |
| **Claude Code auto memory** | "The first 200 lines or 25KB of MEMORY.md, whichever comes first, load at the start of each session" | shipped | Separate from CLAUDE.md; explicit warning: "Bloated CLAUDE.md files cause Claude to ignore your actual instructions!" |
| **LangMem / LangGraph Store** | semantic / episodic / procedural (procedural = evolved prompt instructions) | GA at release (2025-02-18) | Candid gap: "LangMem currently lacks opinionated utilities for this [episodic] type" |
| **Google ADK MemoryService** | Three backends: `InMemoryMemoryService` (keyword), `VertexAiMemoryBankService` (LLM extraction + consolidation), `VertexAiRagMemoryService` (vector similarity) | shipped | **The agentic-vs-fixed split appears again at the memory layer**: `load_memory` tool (agent-initiated) vs `preload_memory` tool (automatic at conversation start) |
| **Vertex Agent Engine Memory Bank** | "uses Generative AI models to generate memories" | **status not stated on the overview page — do not assert GA** | ADK is the documented first-class path |
| **Mem0** | short-term (conversation history, working memory, attention context) + long-term (factual, episodic, semantic). "The search pipeline pulls from all layers, ranking user memories first, then session notes, then raw history" | shipped | LoCoMo **92.5** at mean **6,956 tokens** per retrieval vs "25,000+" full-context; Single Hop 94.6 / Multi-Hop 95.4 / Open-domain 82.3 / Temporal 92.5; "median latency stays flat at +1ms." Warning: "Avoid storing secrets or unredacted PII… Mem0 is retrievable by design" |
| **Zep / Graphiti** | Bi-temporal knowledge graph; "Temporal edge invalidation instead of LLM summarization" for contradictions; vector + full-text + graph traversal "without requiring LLM-based reranking" | Graphiti OSS; Zep commercial (SOC 2, HIPAA, BYOC) | "Sub-200ms retrieval latency versus seconds to tens of seconds." LongMemEval: up to **18.5%** accuracy gain over full-context, "90% faster," "less than 2% of baseline tokens." DMR: Zep 94.8% vs MemGPT 93.4% — with the honest caveat that GPT-4o full-context hit 98.2%, "suggesting DMR benchmark limitations" |
| **Letta** | V1 memory-blocks SDK **deprecating** in favor of Agent SDK with **MemFS** (git-tracked memory, "agent dreaming") | V1 deprecating | MemFS detail page 404'd — specifics unverified |
| **CrewAI** | **Collapsed the taxonomy**: "CrewAI replaced separate memory types with a unified `Memory` class." LLM infers scope/categories/importance on save; retrieval ranks by semantic similarity + recency + importance | shipped | Default LanceDB at `./.crewai/memory`, OpenAI `text-embedding-3-large` (3072-d). Non-blocking saves, auto dedup, "deep recall" multi-step LLM analysis for complex queries; simple queries skip the LLM. *A direct counter-move to the semantic/episodic/procedural consensus.* |
| **Mastra** | Message History, Working Memory (injected as system message), Semantic Recall, and **Observational Memory** — "background agents that compress old messages into dense observations" | shipped | Sub-agent delegation gets a fresh `threadId` and deterministic `resourceId` `{parentResourceId}-{agentName}` |
| **Pydantic AI** | **No memory abstraction, deliberately.** Just `message_history` between runs and typed `deps` via `RunContext` | shipped | "An agent run might represent an entire conversation… However, a conversation might also be composed of multiple runs" |

**SOURCES:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool · https://code.claude.com/docs/en/how-claude-code-works · https://www.langchain.com/blog/langmem-sdk-launch (2025-02-18) · https://adk.dev/sessions/memory/ · https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview · https://docs.mem0.ai/core-concepts/memory-types and https://mem0.ai/research (2026-08-07) · https://blog.getzep.com/state-of-the-art-agent-memory/ (2025-01-22) and https://help.getzep.com/graphiti/getting-started/overview · https://docs.letta.com/concepts/letta · https://docs.crewai.com/en/concepts/memory · https://mastra.ai/docs/memory/overview · https://pydantic.dev/docs/ai/core-concepts/agent/

---

## SECTION 4 — DEEP-RESEARCH / MULTI-HOP PRODUCTS

### 4.1 Perplexity

**CLAIM:** Own crawler and index over **200 billion unique URLs**, on "tens of thousands of CPUs and hundreds of terabytes of RAM," tens of thousands of indexing operations per second, ~200M daily queries.
**CLAIM — pipeline shape:** explicitly multi-stage and progressive — hybrid lexical+semantic retrieval → heuristic prefiltering for staleness → embedding-based scorers → **cross-encoder rerankers on the narrowed candidate set**. Scoring happens at both document and **sub-document span** level so agents get atomic units rather than whole pages.
**CLAIM — latency:** median **358ms** / p95 **763ms**, vs competitors at 513–1,375ms median and 808–2,188ms p95.
**CLAIM — quality (own open-sourced `search_evals`):** SimpleQA **.930**, FRAMES **.453** (single-step); BrowseComp **.371**, HLE **.288** (deep research mode).
**SOURCE:** Perplexity — "Architecting and Evaluating an AI-First Search API" — https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api — 2026-07-29
**STATUS:** shipped-in-production

**CLAIM:** Sonar Deep Research is "exhaustive searches across hundreds of sources." The docs' own sample run shows **21 search queries** and **193,947 reasoning tokens** for a single report, total **$0.816**. Perplexity prices the search *loop* separately from tokens: input $2/1M, output $8/1M, **citation tokens $2/1M**, **reasoning tokens $3/1M**, **search queries $5/1K**. In their sample, reasoning ($0.582) + searches ($0.105) dominated output ($0.091) by ~6x.
**SOURCE:** Perplexity — Sonar Deep Research docs — https://docs.perplexity.ai/docs/sonar/models/sonar-deep-research — undated (accessed 2026-08-08)
**STATUS:** GA

**CLAIM:** Perplexity's retrieval backend is **Vespa**. Vespa's CEO describes the priorities as completeness/freshness/speed, chunks as first-class retrieval units alongside documents, and "multiple stages of progressively advanced ranking" including cross-encoder rerankers.
**SOURCE:** Vespa — Jon Bratseth — "How Perplexity beat Google on AI Search with Vespa.ai" — https://blog.vespa.ai/perplexity-show-what-great-rag-takes/ — 2025-10-06
**STATUS:** shipped-in-production

**GAP FLAG:** Perplexity has published **no Comet orchestration architecture post**. Comet ships as two tiers (Assistant reads page context; Agent takes actions, with per-action permission gating), but no fan-out counts or loop description exist first-party. `perplexity.ai/hub/blog/*` returns 403 across the board.

### 4.2 OpenAI Deep Research

**CLAIM:** Deep research is "an early version of OpenAI o3 optimized for web browsing," trained with **end-to-end RL on browsing tasks** — searching, clicking, scrolling, file interpretation, and sandboxed Python are *learned behaviors*, not a hand-written loop. Training mixed auto-gradable ground-truth tasks with open-ended rubric-graded tasks, scored by a chain-of-thought grader model. The model "pivots as needed in reaction to information it encounters" — mid-trajectory replanning is trained. OpenAI explicitly warns citations may contain errors and that prompt injection encountered during browsing can alter model behavior.
**SOURCE:** OpenAI — Deep Research System Card (Deployment Safety Hub) — https://deploymentsafety.openai.com/deep-research — 2025-02-25
**STATUS:** shipped-in-production
*Note: `openai.com/index/introducing-deep-research/` returns 403; the commonly-cited HLE 26.6% / GAIA figures were NOT verified here and are not asserted.*

**CLAIM:** The API exposes `o3-deep-research` and `o4-mini-deep-research`. At least one data source is mandatory: `web_search_preview`, `file_search` (**max 2 vector stores**), or a remote MCP server implementing `search`+`fetch`; `code_interpreter` optional. Output stream includes `web_search_call` actions (search, open_page, find_in_page), `code_interpreter_call`, `file_search_call`, `mcp_tool_call`, and a final `message` with inline citations (`annotations` carrying url, title, start_index, end_index).
**CLAIM — termination is developer-controlled:** `max_tool_calls` explicitly "to constrain costs and latency." Background mode strongly recommended with webhooks (incompatible with ZDR).
**CLAIM — no built-in clarification step.** OpenAI's own guidance is to preprocess the prompt with a cheaper model (docs reference `gpt-5.6`) before handing it to the deep research model.
**SOURCE:** OpenAI — Deep research API guide — https://developers.openai.com/api/docs/guides/deep-research — accessed 2026-08-08
**STATUS:** GA

### 4.3 Anthropic Research

**CLAIM:** Orchestrator-worker. Lead agent analyzes the query, develops strategy, spawns subagents exploring different aspects **in parallel** — "3-5 subagents in parallel rather than serially," subagents themselves using "3+ tools in parallel." Parallelization "cut research time by up to 90% for complex queries."
**CLAIM — hard-coded effort scaling in the orchestrator prompt:** simple fact-finding = **1 agent, 3-10 tool calls**; direct comparison = **2-4 subagents, 10-15 calls each**; complex research = **10+ subagents** with divided responsibilities.
**CLAIM — search prompt principles:** breadth-first then narrow — "start with short, broad queries, evaluate what's available, then progressively narrow focus." Extended thinking to plan; interleaved thinking after tool results to evaluate quality and identify gaps. Source-quality heuristics favor "specialized tools over generic ones" and primary sources over "SEO-optimized content farms" — added after early agents "consistently chose" lower-quality sources.
**CLAIM — numbers:** Opus 4 lead + Sonnet 4 subagents "outperformed single-agent Claude Opus 4 by **90.2%** on our internal research eval." Agents use "about 4× more tokens than chat interactions"; multi-agent "about **15×** more tokens than chats." "**Token usage by itself explains 80% of the variance**" in BrowseComp-style eval performance. Upgrading to Sonnet 4 gave larger gains than doubling the token budget on Sonnet 3.7.
**CLAIM — stated limits:** not appropriate for tasks requiring all agents to share identical context, many agent-to-agent dependencies, or real-time coordination — and "most coding tasks involve fewer truly parallelizable tasks than research." Known bottleneck: "Current lead-agent synchronous execution of subagents creates information-flow delays and prevents mid-research steering." Economics: "multi-agent systems require tasks where the value… is high enough to pay for the increased performance."
**SOURCE:** Anthropic — "How we built our multi-agent research system" — https://www.anthropic.com/engineering/multi-agent-research-system — 2025-06-13 (Hadfield, Zhang, Lien, Scholz, Fox, Ford)
**STATUS:** shipped-in-production

**CLAIM (launch framing):** "Claude operates agentively, conducting multiple searches that build on each other while determining exactly what to investigate next," integrated with Google Workspace + web search, answers "in minutes," inline citations.
**SOURCE:** https://claude.com/blog/research — 2025-04-15
**STATUS:** beta at publication (Max/Team/Enterprise; US, Japan, Brazil); now shipped

### 4.4 Microsoft

**CLAIM:** M365 Copilot **Researcher** runs an explicit iterative loop — Reasoning (pick next subtask + identify missing detail) → Retrieval (documents, emails, chats, calendar, transcripts, web) → Review (score relevance, write findings to a **scratch pad**) — terminating on **diminishing returns**: it stops at iteration *m* when marginal insight ΔI_m < ε. Powered by OpenAI's deep research model combined with Copilot orchestration and "deep search."
**SOURCE:** Microsoft — "Researcher agent in Microsoft 365 Copilot" — https://techcommunity.microsoft.com/blog/microsoft365copilotblog/researcher-agent-in-microsoft-365-copilot/4397186 — `[snippet-only; page renders title only. Treat the ΔI<ε formalism as attributed but not directly verified]`
**STATUS:** GA (Researcher and Analyst GA'd 2025-06-02)

**CLAIM:** Microsoft frames Researcher's cost as intentional: "Researcher agent deliberately spends more time retrieving and analyzing," respects existing M365 permissions/policies, and may ask clarifying questions before researching.
**SOURCE:** Microsoft Learn — https://learn.microsoft.com/en-us/microsoft-365/copilot/researcher-agent — ms.date 2026-02-19, updated 2026-05-06
**STATUS:** GA

**CLAIM — GraphRAG:** LLM-generated entity/relationship knowledge graph, bottom-up community detection with pre-generated community summaries, answering global questions ("top 5 themes") vector similarity cannot. Claimed to "consistently outperform baseline RAG" on comprehensiveness, source grounding, viewpoint diversity. **The launch post discusses no cost numbers at all.**
**SOURCE:** Microsoft Research — https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/ — 2024-02-13
**STATUS:** research prototype → open source

**CLAIM — LazyGraphRAG (the walk-back):** defers LLM use — NLP noun-phrase extraction instead of LLM entity extraction at index time, sentence-level relevance filtering before LLM processing, and "best-first and breadth-first search dynamics in an iterative deepening manner" at query time. Indexing cost "identical to vector RAG and **0.1% of the costs of full GraphRAG**"; "**more than 700 times lower query cost**" than GraphRAG Global Search at comparable quality; at **4%** of GraphRAG global-search spend it beats competing methods on both local and global queries. Eval: 5,590 AP news articles, 100 synthetic queries (50 local / 50 global).
**SOURCE:** Microsoft Research — https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/ — 2024-11-25
**STATUS:** research prototype (open sourced)

### 4.5 Google

**CLAIM:** Gemini Deep Research "creates a multi-step research plan for you to either revise or approve" (**human-in-the-loop planning**), then "continuously refines its analysis, browsing the web the way you do: searching, finding interesting pieces of information and then starting a new search based on what it's learned. It repeats this process multiple times." Uses the 1M-token context window; exports to Google Docs with source links. **Google does not publish a number of sites browsed.**
**SOURCE:** Google — https://blog.google/products/gemini/google-gemini-deep-research/ — 2024-12-11
**STATUS:** shipped-in-production

**CLAIM:** Google published a human-preference number rather than an orchestration number: raters preferred Gemini Deep Research reports over competitors "by more than a 2-to-1 margin."
**SOURCE:** Google — https://blog.google/products/gemini/deep-research-gemini-2-5-pro-experimental/ — 2025-04-08
**STATUS:** shipped

**CLAIM:** Built on Gemini 2.5 Pro, "optimized to perform task prioritization" and "able to identify when it reaches a dead-end when browsing" — an explicit stopping/backtracking heuristic. HLE **7.95% (Dec 2024) → 26.9%, and 32.4% with higher compute (June 2025)**.
**SOURCE:** Google DeepMind / I/O 2025 — https://blog.google/innovation-and-ai/models-and-research/google-deepmind/google-gemini-updates-io-2025/ — 2025-05 — `[snippet-only]`
**STATUS:** shipped

**CLAIM:** Vertex AI RAG Engine is a six-stage managed pipeline (ingestion → transformation/chunking → embedding → indexing → retrieval → generation) with pluggable backends (RagManagedDb, Vector Search 2.0, Feature Store, Weaviate, Pinecone, Agent Platform Search) and a "Reranking for RAG" stage. **GA only in `europe-west3`/`europe-west4`; allowlist-GA in `us-central1`/`us-east1`/`us-east4`; preview in 14 other regions.**
**SOURCE:** Google Cloud — https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview — accessed 2026-08-08
**STATUS:** mixed GA/preview by region

### 4.6 Glean and Exa

**CLAIM:** Glean's Agentic Engine 2 has three published mechanisms: **adaptive planning** (intent interpretation → plan proposal → context grounding in the Enterprise Graph, with continuous re-planning), **tool orchestration** following an explicit "**explore → narrow → retrieve**" flow with sub-agent delegation, and **dual-layer memory** (session-level + persistent Personal/Enterprise Graph).
**Numbers:** **94% completeness** on end-to-end tasks; **19.4%** enumeration-quality gain over Engine 1; WritingBench **75.7% → 82.5%**; **20%** improvement on previously-downvoted queries; **21%** increase in Assistant query usage during testing; Enterprise Graph ingests "**3× more signals**." *Baseline is their own Engine 1.*
**SOURCE:** Glean — https://www.glean.com/blog/live-fall-25-agentic-engine2-performance — 2025-09-25
**STATUS:** shipped-in-production

**CLAIM:** Glean's earlier post describes a **Reflect** stage — the agent self-assesses confidence in initial search results and decides whether additional tools are warranted *before* committing to the expensive agentic path, i.e. a cheap/expensive routing gate. Claimed "**24%** increase in relevance." Retrieval is hybrid (semantic + lexical + knowledge graph over 100+ connectors), role-based-permission scoped so unauthorized content never reaches the LLM.
**SOURCE:** Glean — https://www.glean.com/blog/agentic-reasoning-future-ai — 2024-11-19
**STATUS:** shipped-in-production

**CLAIM:** Exa Agent "divides the task into many subtasks and assigns subagents to research various domains at once," and uses "a fusion of frontier and cost-effective models to find the most cost-effective methodology" — **model-tier routing inside the research loop**. Claims "up to **94%** reductions in token usage." WideSearch: ~50% Row-F1 at ~$0.50/query.
**CLAIM — research budget as a first-class product dial:** fixed-price effort tiers `minimal` $0.012 / `low` $0.025 / `medium` $0.10 (default) / `high` $0.50 / `xhigh` $1.00 per request, plus metered `auto` (default cap $5) and `max` beta (cap $20). Billed as Agent Compute Units (1 ACU = $0.10) + $0.005 per search call. Async run model with polling/SSE/batch, replayable events, continuation from a completed run; outputs carry **field-level grounding** citations against a JSON `outputSchema`.
**SOURCES:** Exa — https://exa.ai/blog/exa-agent (2026-06-16) · Agent API guide https://exa.ai/docs/reference/agent-api-guide (accessed 2026-08-08)
**STATUS:** GA (`max` effort in beta)

**CLAIM:** Exa Deep Max published accuracy/latency frontier: Deep Search QA **90% at 64s** (vs You Frontier 84%/5908s, Parallel Ultra 8x 82%/1703s); FRAMES **94% in 11s** (vs Parallel Ultra 88%/1457s); HLE-Search **80% at 25s**, "matching GPT 5.4 on quality but at half the latency"; "up to 20x faster than the closest competitor." Architecture: "dozens of parallel calls to Exa Search," each "target[ing] a different angle," over an in-house index returning "results in under a second." Index described as "many petabytes," "semantic+lexical databases from scratch."
**SOURCE:** Exa — https://exa.ai/blog/deep-max — 2026-04-20
**STATUS:** GA (pricing on request)

### 4.7 Cross-cutting: orchestration shapes and stopping rules

**The published designs split cleanly:**

| Org | Shape | Stopping rule |
|---|---|---|
| **Anthropic Research** | Orchestrator + 3-5 (up to 10+) parallel subagents, each with its own context window, returning distilled 1-2k-token summaries | Model judgment steered by prompt heuristics; `max_uses` as a hard backstop |
| **OpenAI Deep Research** | One RL-trained agent, sequential trajectory | Developer-set hard cap (`max_tool_calls`) |
| **Azure AI Search agentic retrieval** | **No agent in the retrieval layer** — one LLM planning pass, then stateless parallel subquery fan-out with per-subquery L2 reranking and a merge | **No stopping rule — it is a single fixed fan-out round, not a loop** |
| **Microsoft Researcher** | Sequential reason/retrieve/review cycles with a scratch pad | Marginal-information threshold ΔI < ε |
| **Google Deep Research** | Human-approved plan, then iterative browse-and-refine | Trained dead-end detection + task prioritization |
| **Glean** | Adaptive planner + confidence-gated sub-agent delegation, explore → narrow → retrieve | Confidence gate before committing to the expensive path |
| **Exa Agent** | Subagents per domain with model-tier routing | Monetary cap (effort tier / ACU budget) |
| **Perplexity Sonar DR** | Query fan-out over own index, multi-stage rerank | Undisclosed; sample run = 21 queries |

**Permissions as an architectural constraint (four orgs, independently):** Glean scopes retrieval to role-based ACLs so unauthorized content never reaches the model; OpenAI company knowledge "respects your existing company permissions"; Anthropic's argument for indexless code search includes avoiding permission-synchronization problems inherent to a centralized index; Elastic implements it as native document-level security in the query itself (zero cross-tenant leaks measured).

**Citation verification, three different shapes:** Anthropic = a dedicated CitationAgent pass. Google = a standalone check-grounding API with a 0–1 support score and per-claim "grounding required" flags at <500ms. OpenAI = display-layer enforcement plus a `sources` superset. Anthropic's web search makes citations non-optional and token-free; web fetch makes them opt-in and off by default.

---

## SECTION 5 — EVALS FOR AGENTIC RETRIEVAL

### 5.1 The structural news: OpenAI is exiting the eval-product space

**CLAIM:** Verbatim timeline: "June 3, 2026 | Deprecation announced for the Evals platform." / "Oct 31, 2026 | Existing evals become read-only." / "Nov 30, 2026 | The Evals dashboard and API are scheduled to shut down." Graders are also deprecated "as part of the evals and fine-tuning workflows they support." Recommended migration: OpenAI **Datasets** and — remarkably — third-party **Promptfoo**.
**SOURCES:** OpenAI — Deprecations — https://developers.openai.com/api/docs/deprecations (announced 2026-06-03) · Working with evals — https://developers.openai.com/api/docs/guides/evals · Graders — https://developers.openai.com/api/docs/guides/graders
**STATUS:** shipped feature being retired

**CLAIM:** OpenAI's grader set was always thin: **String Check** (0/1), **Text Similarity** (BLEU/METEOR/ROUGE/fuzzy/cosine), **Score Model** (LLM returns a numeric score), **Python** (arbitrary code returning a float). **No faithfulness, groundedness, or citation grader ever existed.** There is no "label model" grader despite common belief.
**SOURCE:** same
**STATUS:** deprecating

**CLAIM:** OpenAI's post-Evals agent story is **trace grading, not metrics**: "The fastest way to identify workflow-level issues" is trace grading; "Graders let you score those traces with structured criteria." No named agent metrics ship — you write the grader. Stated dimensions (methodology only): tool selection, data/argument precision, agent handoff accuracy. Named anti-patterns: perplexity/BLEU, "vibe-based evals," deferring evals to production.
**SOURCES:** https://developers.openai.com/api/docs/guides/agent-evals · https://developers.openai.com/api/docs/guides/evaluation-best-practices
**STATUS:** shipped feature + methodology

### 5.2 Is RAGAS still the default?

**CLAIM:** Ragas repositioned away from "a bag of reference-free RAG metrics" toward an experiments-first loop — the docs home headlines moving "from 'vibe checks' to systematic evaluation loops," foregrounding `experiments`, custom metrics via decorators, and dataset/result tracking. RAG is now **one of seven metric families** (RAG; NVIDIA metrics; Agents/Tool Use; Natural Language Comparison; SQL; General Purpose incl. Aspect Critic and Rubrics; Summarization).
**SOURCES:** https://docs.ragas.io/en/stable/ (2025-12-09) · https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ (2025-12-09) · https://docs.ragas.io/en/stable/concepts/experimentation/ (2025-12-09)
**STATUS:** open-source library
*Notably, the experiments docs give **no** guidance on when prefab metrics stop being sufficient — the crux of the disagreement below.*

**CLAIM:** Ragas' agentic metrics: Topic Adherence (precision/recall/F1 vs `reference_topics`), Tool Call Accuracy (0–1; strict-order default or flexible), Tool Call F1 (order-independent), Agent Goal Accuracy (**binary 0/1**, with or without reference).
**SOURCE:** https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/ — 2025-12-09
**STATUS:** open-source

**CLAIM — the flat "No."** Asked "Should I use 'ready-to-use' evaluation metrics?" Hamel Husain and Shreya Shankar answer **"No."** Supporting: "Generic evaluations waste time and create false confidence"; "These metrics measure abstract qualities that may not matter for your use case"; "All you get from using these prefab evals is you don't know what they actually do."
**IMPORTANT ACCURACY NOTE:** RAGAS is **not named** in the FAQ. The critique is categorical ("prefab evals"), not vendor-specific. The only vendors named are observability platforms: "Vendors I encounter the most organically in my work are: Langsmith, Arize and Braintrust." Do not attribute a named anti-Ragas quote to Hamel.
**SOURCE:** Hamel Husain & Shreya Shankar — AI Evals FAQ — https://hamel.dev/blog/posts/evals-faq/ — published 2025-05-28, last modified **2026-07-18**
**STATUS:** methodology recommendation

**CLAIM — the clearest replacement paradigm:** Contextual AI's **LMUnit** = natural-language unit tests. A unit test is "A specific, clear, testable statement or question in natural language about a desirable quality of an LLM's response." Continuous 1–5 score, `POST https://api.contextual.ai/v1/lmunit`, inputs `query`/`response`/`unit_test`, 7,000-token cap. Claims: SOTA on FLASK and BigGenBench; top-5 on RewardBench at 93.5%; beat GPT-4o and Claude 3.5 Sonnet "by over 9%" on in-house finance/engineering data.
**SOURCES:** https://contextual.ai/lmunit/ (2024-12-18; open-sourced 2025-07-22) · https://docs.contextual.ai/api-reference/lmunit/lmunit
**STATUS:** shipped API + open model

### 5.3 The practitioner canon (Hamel, Jason Liu, Eugene Yan)

**CLAIM (Hamel, 2024):** "Don't rely on generic evaluation frameworks to measure the quality of your AI. Instead, create an evaluation system specific to your problem." Three levels: unit tests/assertions → human & model eval → A/B testing. And: "Many vendors want to sell you tools that claim to eliminate the need for a human to look at the data."
**SOURCE:** https://hamel.dev/blog/posts/evals/ — 2024-03-29

**CLAIM (Hamel, 2024 — Critique Shadowing):** "If your evaluations consist of a bunch of metrics that LLMs score on a 1-5 scale (or any other scale), you're doing it wrong." Method: a single **principal domain expert** issues binary pass/fail + written critique; iterate the judge to alignment (Honeycomb case: ">90% agreement between the LLM and Phillip" in three iterations). On generic judges: "Nothing is strictly wrong with them. It's just that many people are led astray by them." Track precision and recall separately, not raw agreement.
**SOURCE:** https://hamel.dev/blog/posts/llm-judge/ — 2024-10-29

**CLAIM (Hamel, 2025):** Error analysis is "the single most valuable activity in AI development and consistently the highest-ROI activity." Generic metrics "create a false sense of measurement and progress" and "fragment your attention." Custom annotation tooling: "every domain has unique needs that off-the-shelf tools rarely address"; teams with good data viewers "iterate 10x faster."
**SOURCE:** https://hamel.dev/blog/posts/field-guide/ — 2025-03-24

**CLAIM (Hamel & Shreya — the most citable operational numbers in the field):** Error-analysis pipeline: representative dataset → open coding with domain experts → axial coding into a failure taxonomy → iterate to theoretical saturation, "**~100 traces minimum**"; "if ~20 traces don't turn up a new category, you can stop"; weekly "review 10-20 traces… focusing on outliers"; "at least 100+ fresh traces each review cycle." Binary over Likert because Likert lets people "hide uncertainty in middle values" and needs larger samples. Judge validation: "Focus on achieving high True Positive Rate (TPR) and True Negative Rate (TNR) with your judge on a held out labeled test set," then "correct its estimates to determine the actual failure rate." Judge model: "Using the same model is usually fine because the judge is doing a different task than your main LLM pipeline." Custom annotation UI is "the single most impactful investment you can make."
**CLAIM (retrieval-specific, and they DO prescribe classical IR metrics):** Recall@k, Precision@k, MRR; build eval sets by "reverse-generating queries from documents." Synthetic query generation must use **structured dimensions** — define variation categories → hand-write ~20 tuples → two-step generation tuple→NL query — explicitly *not* unstructured "give me test queries." Caveats: synthetic data fails for domain-specific content, low-resource languages, high-stakes domains.
**CLAIM (agent eval):** two-phase — end-to-end task success as a black box, then step-level diagnostics using a **transition failure matrix** mapping last successful state vs first failure location to find workflow hotspots.
**SOURCE:** https://hamel.dev/blog/posts/evals-faq/ — pub 2025-05-28, mod 2026-07-18

**CLAIM (Jason Liu, 2024 — the RAG flywheel):** 9 stages: Initial Implementation → Synthetic Data Generation → Fast Evaluations → Real-World Data Collection → Classification/Analysis → System Improvements → Production Monitoring → User Feedback → Iteration. "Generate synthetic questions for each chunk of text in your database. Use these to test your retrieval system and calculate precision and recall scores." Retrieval evals are "lightning-fast (milliseconds vs. seconds per question)." Leading indicators: retrieval experiments run per week, precision/recall improvement on synthetic data, time to run the eval suite. Segmentation: unsupervised clustering of real questions into topics, few-shot topic classifiers, plus an **"Other" bucket monitored over time** to detect drift in user needs.
**SOURCE:** https://jxnl.co/writing/2024/08/19/rag-flywheel/ — 2024-08-19

**CLAIM (Jason Liu, 2025):** "If your retrieval is wrong—pulling the wrong chunk entirely—no model version will fix that." Distinguishes **inventory gaps** (data missing) from **capability gaps** (data present, not surfaced). Feedback UX: thumbs up/down plus "Highlight which snippet is wrong or missing," feeding embedding/reranker training.
**SOURCE:** https://jxnl.co/writing/2025/01/24/systematically-improving-rag-applications/ — 2025-01-24

**CLAIM (Jason Liu, 2025 — "There Are Only 6 RAG Evals"):** Tier 2: Context Relevance (C|Q, retrieval), Faithfulness/Groundedness (A|C, generation), Answer Relevance (A|Q, end-to-end). Tier 3: Context Support Coverage (C|A), Question Answerability (Q|C), Self-Containment (Q|A). Beneath these: "Retrieval Precision & Recall… don't require LLMs, and provide quick feedback for retriever tuning" — MAP@K and MRR@K. Critique: "Don't waste time on complexity theater"; "teams obsess over generation quality while neglecting to ensure retrieval (C|Q) is even working correctly."
**SOURCE:** https://jxnl.co/writing/2025/05/19/there-are-only-6-rag-evals/ — 2025-05-19

**CLAIM (Eugene Yan):** "I tend to be skeptical of correlation metrics… where possible, I have my evaluators return binary outputs." Objective tasks (factuality, toxicity) → direct scoring; subjective tasks (tone, persuasiveness) → pairwise. Documented judge biases: position bias (gpt-3.5 ~50%, claude-v1 ~70%), verbosity bias (both preferred longer responses ">90% of the time"), self-enhancement (gpt-4 +10% own-output win rate; claude-v1 +25%). Faithfulness correlation for gpt-4 only ρ≈0.55; **HaluEval best model 58.5% accuracy**.
**SOURCE:** https://eugeneyan.com/writing/llm-evaluators/ — 2024-08

**CLAIM (Eugene Yan — AlignEval):** "Align AI to human. Calibrate human to AI. Repeat." Label ≥20 samples pass/fail *before* writing criteria — "working backward from the data" so criteria reflect real, frequent defects. Reports sample size, recall, precision, F1, Cohen's κ, TP/FP/TN/FN.
**SOURCE:** https://eugeneyan.com/writing/aligneval/ — 2024-10

**CLAIM (Jo Kristian Bergum — the practical middle path):** calibrate an LLM judge against a small human-labeled set, then scale it into a Cranfield-style test collection. 90 human-labeled query-passage pairs over 26 queries → calibrate GPT-4o → then label **10,372 unique query-passage pairs from 386 real queries**. "There are few cases where the LLM disagrees by more than one level. In only 1 case does it assign irrelevant for something that the human assigned as highly relevant."
**SOURCE:** Vespa — https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/ — 2024-07-03

### 5.4 Anthropic's published eval position (the most conservative shipped stance)

**CLAIM:** An eval is "a test for an AI system: give an AI an input, then apply grading logic to its output to measure success." Three grader classes with explicit tradeoffs — code (fast/cheap/objective/reproducible but "brittle to valid variations"), model-based (flexible/scalable, "non-deterministic," more expensive), human ("gold standard quality" but expensive/slow). Why agents are harder: "Agents use tools across many turns, modifying state in the environment and adapting as they go—which means mistakes can propagate and compound."
**CLAIM — sample size:** "**20-50 simple tasks drawn from real failures is a great start.**"
**CLAIM — judge calibration:** "Model-based graders should be closely calibrated with human experts to gain confidence that there is little divergence"; "give the LLM a way out, like providing an instruction to return 'Unknown'."
**CLAIM — the killer line:** "**As a rule, we do not take eval scores at face value until someone digs into the details of the eval and reads some transcripts.**"
**CLAIM — search-specific:** they built "evals covering both directions: queries where the model should search… and queries where it should answer from existing knowledge."
**CLAIM — lifecycle:** automated evals pre-launch/CI-CD → production monitoring for distribution drift → A/B testing once traffic suffices → continuous user-feedback triage + transcript review.
**SOURCE:** Anthropic — "Demystifying evals for AI agents" — https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents — 2026-01-09
**STATUS:** methodology recommendation (first-party)

**CLAIM (the research-agent rubric, verbatim criteria):** "factual accuracy (do claims match sources?), citation accuracy (do the cited sources match the claims?), completeness (are all requested aspects covered?), source quality (did it use primary sources over lower-quality secondary sources?), and tool efficiency (did it use the right tools a reasonable number of times?)." Single LLM call, 0.0–1.0 plus pass/fail. Started with **~20 queries**. Prioritized "**end-state evaluation rather than turn-by-turn analysis**" because "agents could follow different valid paths." Human testing caught what automation missed — notably an **SEO-optimized-content bias over authoritative academic sources**, a directly agentic-retrieval failure mode.
**SOURCE:** https://www.anthropic.com/engineering/multi-agent-research-system — 2025-06-13

### 5.5 Agent-specific eval as a shipped product

**CLAIM — Microsoft Foundry ships the most granular suite, split System vs Process, and it is deliberately BINARY.** Verbatim: agent evaluators "function like unit tests for agentic systems—they take agent messages as input and output binary Pass/Fail scores (or scaled scores converted to binary scores based on thresholds)."

| Evaluator | Class | Output |
|---|---|---|
| Task Completion (preview) | System | Binary |
| Customer Satisfaction (preview) | System | **1–5 Likert** (the lone exception) |
| Task Adherence (preview) | System | Binary |
| Task Navigation Efficiency | System | Binary + precision/recall/F1 (`exact_match` / `in_order_match` / `any_order_match`) |
| Intent Resolution (preview) | System | Binary via threshold on 1–5 |
| Tool Call Accuracy | Process | Binary via threshold on 1–5 |
| Tool Selection | Process | Binary |
| Tool Input Accuracy | Process | Binary (6 strict criteria) |
| Tool Output Utilization | Process | Binary |
| Tool Call Success | Process | Binary |
| Quality Grader (preview) | Quality | Binary |

Recommended judge model: `gpt-5-mini`.
**DOCUMENTED BLIND SPOT:** Foundry's own docs say to **avoid** `tool_call_accuracy`, `tool_input_accuracy`, `tool_output_utilization`, `tool_call_success`, and `groundedness` when the agent calls Azure AI Search, Bing Grounding, Bing Custom Search, SharePoint Grounding, Code Interpreter, Fabric Data Agent, or Web Search — i.e. **exactly the agentic-retrieval tools are the least-supported case.**
**SOURCE:** Microsoft — https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/agent-evaluators — ms.date 2026-06-02
**STATUS:** shipped (several in public preview)

**CLAIM — LangSmith/AgentEvals ships four deterministic trajectory-match modes plus an LLM-judge trajectory evaluator.** `create_trajectory_match_evaluator` modes: **strict** (exact messages+tool calls, same order), **unordered** (same tool calls, any order), **subset** (agent calls only tools from reference, no extras), **superset** (reference tools plus extras allowed). LLM-judge variant "can assess nuanced aspects like efficiency and appropriateness," reference trajectory optional.
**SOURCE:** https://docs.langchain.com/langsmith/trajectory-evals
**STATUS:** OSS + shipped

**CLAIM:** LangSmith shipped **Insights Agent** (clusters production traces to surface failure modes — "Group by poor interactions: Cluster based on how your agent is messing up") and **Multi-turn Evals** scoring whole conversations on "Semantic intent… Semantic outcomes… Agent trajectory," firing online when a conversation concludes. *This is vendor productization of Hamel's error analysis.*
**SOURCE:** https://www.langchain.com/blog/insights-agent-multiturn-evals-langsmith — 2025-10-23
**STATUS:** shipped

**CLAIM — Databricks MLflow 3 ships single-turn AND multi-turn judges.** Single-turn: relevance, retrieval quality, safety, groundedness, correctness, `ToolCallCorrectness`, `ToolCallEfficiency` (trace-based). Retrieval judges are separate first-class scorers: `RetrievalRelevance`, `RetrievalGroundedness`, `RetrievalSufficiency` (requires ground truth). **Multi-turn: conversation completeness, user frustration, knowledge retention across interactions, role adherence, safety across the conversation.**
**SOURCES:** https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/ (2026-06-23) · https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/ · https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/ (2026-07-28, recommends migrating to MLflow 3)
**STATUS:** shipped

**CLAIM — the strongest vendor-side concession that generic judges are inadequate:** MLflow ships judge→human **alignment** as a product with three optimizers — **MemAlign** (default, "dual-memory system for fast and cheap few-shot alignment"), **SIMBA** (DSPy), **GEPA** (DSPy reflection). Requires "minimum 10" human-labeled traces, recommends ≥30% each positive and negative. Claim: "**Aligned judges show 30-50% reduction in false positives/negatives compared to generic evaluation prompts.**"
**SOURCE:** https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment/
**STATUS:** shipped

**CLAIM:** LlamaIndex ships retrieval eval **separately** from response eval — retrieval: MRR and hit-rate with synthetic question-context pair generation from unstructured text; response: Faithfulness, Answer/Context Relevancy, Correctness, Semantic Similarity, Guideline adherence. Explicitly notes "many of these current evaluation modules do *not* require ground-truth labels."
**SOURCE:** https://developers.llamaindex.ai/python/framework/module_guides/evaluating/
**STATUS:** OSS

### 5.6 Retrieval-stage metrics and the label problem

**CLAIM:** Consensus metric set is Precision@K, Recall@K, MRR@K, MAP@K, NDCG@K, with rank-awareness as the selection criterion. "NDCG@K is highly popular for evaluating retrieval systems" and is the default for the MTEB Retrieval category.
**SOURCE:** Weaviate — https://weaviate.io/blog/retrieval-evaluation-metrics — 2024-05-28

**CLAIM:** "It begins… with offline metrics to predict the system's performance before deployment." Cites Spotify using Recall@K during training then MRR@K on larger eval sets.
**SOURCE:** Pinecone — https://www.pinecone.io/learn/offline-evaluation/ — 2023-06-30

**CLAIM — the label-coverage problem, quantified:** "A single summary statistic misses many nuances to evaluating search." Elastic computes a **judge rate** — "the average percentage of the top-10 documents that have a score in the qrels file." Using Phi-3-mini-4k to fill unjudged documents they found **57.6% of tested unlabeled docs were actually relevant**, with LLM–human agreement "close to 80%." BEIR is described as the community's nDCG@10 "holy grail."
**SOURCE:** Elastic Search Labs — https://www.elastic.co/search-labs/blog/evaluating-search-relevance-part-1 — 2024-07-16

**CLAIM — public benchmarks don't predict production:** MTEB/BEIR are "generic, which fail to capture the domain-specificity of real-world retrieval applications," "overly clean," and contaminated — "Models have already seen most of these benchmarks in their training data." 11.91% of Wikipedia English query pairs exceeded 0.9 cosine similarity to LLM reproductions. **Ranking inversion demonstrated:** jina-embeddings-v3 beats text-embedding-3-large "across all MTEB English tasks," but on the WandBot production corpus Recall@10 was voyage-3-large 0.679 > text-embedding-3-large 0.602 > jina-embeddings-v3 0.532. LLM-judge/human alignment reached 75.2% after iteration.
**SOURCE:** Chroma — "Generative Benchmarking" — https://www.trychroma.com/research/generative-benchmarking — 2025-04-07 (Hong, Troynikov, Huber, with Morgan McGuire of W&B)
**STATUS:** technical report — *this quietly invalidates the BEIR/nDCG@10 leaderboard arms race that Cohere, Voyage, Jina, Elastic, and Contextual AI all use as their primary marketing claim.*

### 5.7 Benchmarks vendors actually cite in 2025–26

| Benchmark | What it measures | Detail | Maintainer |
|---|---|---|---|
| **BRIGHT** | Reasoning-intensive retrieval — the benchmark that broke the "embeddings are solved" narrative | 1,385 queries from StackExchange, LeetCode, math competitions, requiring "in-depth reasoning to identify relevant documents," vs prior benchmarks where "keyword or semantic-based retrieval is usually sufficient." Metric nDCG@10. Leaderboard: Mira-Reasoning-Retrieval 66.9 (2026-04-22), INF-X-Retriever 63.4 (2025-12-20), RakanEmbed4B 52.4 (2026-03-20) | HKU, Princeton, UW, **Google Cloud AI Research** — https://brightbenchmark.github.io/ |
| **BEIR** | Zero-shot IR baseline; nDCG@10 lingua franca | 18 public datasets; NDCG/MAP/Recall/Precision at k ∈ {1,3,5,10,100,1000} | Thakur, Reimers, Gurevych, Lin, Rücklé — https://github.com/beir-cellar/beir |
| **BrowseComp** | Browsing-agent persistence and creativity | **1,266 questions**, short verifiable answers. Wei, Sun, Papay, McKinney, Han, Fulford, Chung, Passos, Fedus, Glaese | OpenAI — arXiv:2504.12516 (2025-04-16); openai.com/index/browsecomp/ 403s |
| **FRAMES** | Unified factuality + retrieval + reasoning | 824 multi-hop questions, 2–15 Wikipedia articles each. **Baselines: Gemini-Pro-1.5 40.8% naive, 47.4% BM25@4, 66% multi-step retrieval+reasoning, 72.9% oracle retrieval** — a ~26-point retrieval gap, the strongest single argument for evaluating retrieval separately | Google — arXiv:2409.12941 (2024-09) `[snippet-only]` |
| **TREC RAG** | Academic gold standard | 2026 tasks: Retrieval (R) and RAG. **Corpus changed for 2026 to ClimbMix-400b (NVIDIA), replacing MS MARCO v2.1**, via Pyserini REST. "RAG25 nuggets" as dev data; **ResearchRubrics** for system testing; **RAGDoll** as the automated end-to-end eval framework. 2026: topics July 6, submissions Aug 8, conference Nov | https://trec-rag.github.io/ |
| **DeepResearch Bench** | Deep-research agents | 100 expert-crafted tasks (50 EN / 50 ZH), 22 fields, derived from 96,000+ user queries. **RACE** (comprehensiveness, insight/depth, instruction adherence, clarity) + **FACT** (extracts factual claims, verifies cited sources, computes citation accuracy). **As of May 2026 migrated to GPT-5.5 as primary evaluator, replacing Gemini-2.5-Pro** | USTC — https://github.com/Ayanami0730/deep_research_bench |
| **MMTEB / MTEB v2** | 500+ tasks, 250+ languages | Adds instruction following, long-document retrieval, code retrieval. Headline: smaller multilingual models often outperform large LLMs | ICLR 2025, arXiv:2502.13595 `[snippet-only; leaderboard Space returned loading shell]` |
| **Vectara HHEM** | Factual consistency (not summary quality) | See §3.6 | https://github.com/vectara/hallucination-leaderboard |

### 5.8 Online / production eval

**CLAIM:** Microsoft Foundry ships four mechanisms: "Continuous evaluation: Quality and safety evaluation of production traffic **at a sampled rate**"; "Scheduled evaluation… using test datasets **to detect system drift**"; "Scheduled red teaming"; "Azure Monitor alerts." Plus **cluster analysis** to group evaluation failures. OpenTelemetry tracing supporting LangChain, LangGraph, OpenAI Agents SDK, Microsoft Agent Framework. Playground evals are **on by default and consumption-billed**.
**SOURCE:** https://learn.microsoft.com/en-us/azure/foundry/concepts/observability — ms.date 2026-07-31, updated 2026-08-01

**CLAIM:** LangSmith online evals: configurable sampling (0.1 = 10% of filtered runs); weekly LLM spend limits per project/dataset that auto-pause evaluators; filters to trigger **only on runs with negative user feedback**, specific tool invocations, or metadata (e.g. customer tier); backfill rules onto historical runs.
**SOURCE:** https://docs.langchain.com/langsmith/online-evaluations

**CLAIM:** Braintrust — "Online scoring evaluates production traces automatically as they're logged, running asynchronously with no impact on latency." Model = Data + Task + Scorers/Classifiers, where "scorers measure quality with numeric scores, while classifiers apply categorical labels."
**SOURCE:** https://www.braintrust.dev/docs/guides/evals

**CLAIM:** Arize Phoenix uses "function calling (tool use) to extract structured judgments" so the judge emits a **categorical label** which Phoenix "maps… to its numeric score" — label-first, score-derived, matching the Hamel/Yan binary preference. Native OpenTelemetry, "up to 20x performance gains via concurrency and batching," explanations on all LLM evals.
**SOURCE:** https://arize.com/docs/phoenix/evaluation/llm-evals

**CLAIM:** Galileo's alignment mechanisms are **CLHF** ("Use CLHF to continuously improve the metrics") and **Autotune** ("continuously provide feedback in natural language that automatically improves the metrics"), with **Luna-2** small models as the metric-computation engine. RAG metrics: chunk relevance, context relevance, context precision, Precision@K, chunk attribution, chunk utilization, context adherence, completeness.
**SOURCE:** https://docs.galileo.ai/concepts/metrics/overview

---

## SECTION 6 — WHERE VENDORS DISAGREE (consolidated)

**D1 — Code retrieval: embeddings vs grep vs deterministic index. The sharpest live fight.**

| Party | Position | Evidence | Date |
|---|---|---|---|
| **Cursor** | Embeddings are *necessary* at scale; use both. "semantic search is currently necessary to achieve the best results, especially in large codebases" / "the combination of these two leads to the best outcomes" | Offline +12.5% avg QA accuracy (range 6.5%–23.5%); online A/B +0.3% code retention overall, **+2.6% on codebases with 1,000+ files**, −2.2% dissatisfied follow-ups. Trained a custom embedding model on agent session traces | https://cursor.com/blog/semsearch — 2025-11-06 |
| **Cognition** | *Both* plain embeddings and plain agentic search are wrong. Embedding search gives "inaccurate results for complex queries" and "context pollution"; agentic search needs "dozens of sequential roundtrips." Fix: SWE-grep, an RL-trained retrieval model doing up to 8 parallel tool calls over max 4 turns | SWE-grep-mini 2,800+ tok/s (20x Haiku 4.5's 140); SWE-grep 650+ tok/s. "An order of magnitude faster" while "matching or outperforming" frontier models. 5-second "flow window" design target | https://cognition.com/blog/swe-grep — 2025-10-16; https://cognition.com/blog/swe-1-5 — 2025-10-29 |
| **Sourcegraph** | **Two positions 7 months apart.** Oct 2025: "Semantic search complements keyword search. It doesn't replace it… the best results come from using both." May 2026: embedding retrieval "returns plausible-looking results that miss cross-cutting impact"; on large codebases "agents ship plausible-looking code with latent bugs." Argues for "exact symbol definitions, callsites, and implementers" | Argument, not benchmark. Ships MCP Server, Deep Search, Code Search ("Deterministic, exact, exhaustive engine") | https://sourcegraph.com/blog/semantic-code-search-what-it-is-and-how-it-works — 2025-10-06; https://sourcegraph.com/blog/agentic-coding — 2026-05-21 |
| **Anthropic** | Product stance: "RAG-powered AI coding tools… embed the entire codebase… At large scale, those systems can fail because embedding pipelines can't keep up with active engineering teams." Claude Code traverses the filesystem, greps, follows references — "no embedding pipeline or centralized index to maintain." But the nuanced version concedes: "Semantic search is usually faster than agentic search, but less accurate… start with agentic search, and only add semantic search if you need faster results" | No published retrieval benchmark | https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start — 2026-05-14; context engineering post 2025-09-29 |
| **LlamaIndex** | "agents with good filesystem tools… outperform naive semantic search" | Assertion | 2026-03-03 |

**Direct contradiction:** Cursor says semantic search is *necessary* specifically on large codebases; Sourcegraph says approximate retrieval fails specifically on large codebases. Both ship products; both have commercial interest (Cursor trained its own embedding model, Sourcegraph sells a deterministic index).
**ACCURACY NOTE:** No first-party Anthropic page says "we removed vector search from Claude Code" or "embeddings don't work for code." That claim circulates in secondary sources. Anthropic's architecture reveals the position; the prose is more hedged. And Anthropic's own pro-embeddings artifact (Contextual Retrieval) has never been retracted — the honest reconciliation is **by domain**: embeddings for document corpora, agentic search for code.

**D2 — Multi-agent vs single agent. Published one day apart.**
- **Cognition (2025-06-12):** "running multiple agents in collaboration only results in fragile systems." Two principles: "Share context, and share full agent traces, not just individual messages" and "Actions carry implicit decisions, and conflicting decisions carry bad results." Names OpenAI Swarm and Microsoft AutoGen as libraries that "actively push concepts which I believe to be the wrong way." Calls context engineering "the #1 job of engineers building AI agents." — https://cognition.com/blog/dont-build-multi-agents
- **Anthropic (2025-06-13):** multi-agent beat single-agent by **90.2%** on their internal research eval; "Multi-agent systems excel at valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools." — https://www.anthropic.com/engineering/multi-agent-research-system
- **The reconciliation both sides state:** Anthropic concedes "most coding tasks involve fewer truly parallelizable tasks than research." Cognition's product is a *coding* agent. LangChain independently landed on the same boundary in their own reference implementation: "**We restrict multi-agent to research, and write the report in one-shot**" — https://www.langchain.com/blog/open-deep-research — 2025-07-16. Anthropic's Nov 2025 long-running-agents post is agnostic and admits uncertainty about whether specialized sub-agents help.

**D3 — "More compute = better research" vs cost discipline.**
Anthropic asserts token usage explains **80%** of eval variance and accepts **15×** chat token cost. Microsoft's Azure docs push the opposite: lower reasoning effort, consolidate indexes to reduce fan-out, reorganize content "so the most relevant information can be found with fewer sources." OpenAI's own GPT-5 prompting guidance says "**Prefer acting over more searching**" and shows a 2-tool-call budget example. Exa monetizes the middle by exposing compute as a per-request price tier. Perplexity's cost breakdown shows reasoning tokens outweighing output tokens ~6×.

**D4 — Where the stopping rule lives.** OpenAI = developer hard cap. Anthropic = model judgment + prompt heuristics + `max_uses` backstop. Microsoft Researcher = marginal-information threshold. Google = trained dead-end detection. Azure agentic retrieval = **no stopping rule at all** (single fixed fan-out). Exa = a monetary cap.

**D5 — RRF vs weighted score fusion.** Weaviate moved its default *off* RRF; Qdrant calls RRF "the de facto standard"; MongoDB GA'd both and declines to steer; Pinecone claims reranking beats both by 8% on BEIR. **Nobody has published a head-to-head with numbers.**

**D6 — Late interaction: production or not.** Vespa "Yes" explicitly; Weaviate GA with a candid recall cliff; Elastic GA per release blog; Qdrant reranker-only and client-library-only; Pinecone no native support, calls end-to-end ColBERT "notably slow"; Vectara actively rejecting patch-level multi-vector.

**D7 — LLM rerankers.** Voyage publishes that they are 25–60x more expensive, up to 48x slower, and *degrade* NDCG@10 when the first stage is good. Meanwhile Chroma trains a 20B LLM to *be* the retriever, SID/turbopuffer trains an LLM search agent claiming 0.77 vs 0.45 recall over classical rerank pipelines (5.5s vs 131s, $0.62 vs $240 per 1k questions), and Exa sells frontier-LLM agentic search at 90% Deep Search QA. **The two camps are not measuring the same thing** — Voyage measures *reranking a fixed candidate list*; the others measure *multi-hop search where the agent issues new queries*. That distinction is doing all the work and no vendor states it plainly.

**D8 — Chunk size.** Chroma's own eval says ~200 tokens dominates and calls 800-token defaults worst-in-class. OpenAI's default is 800/400. Elastic's `semantic_text` default is "250 words (approximately 400 tokens)." Jina argues the question is wrong-framed and you should late-chunk a long-context encoder instead.

**D9 — Contextual retrieval, publicly disparaged.** Jina calls Anthropic's method "a brute-force approach" and claims late chunking is cheaper, faster, and more robust to bad boundaries. No matched head-to-head exists. No vendor ships Anthropic-style contextual chunk augmentation as a GA feature.

**D10 — GraphRAG vs vector-only vs no-index.** Microsoft Research argues graph structure is required for global/aggregative questions, then walks back the cost 1000x with LazyGraphRAG. Anthropic argues the opposite for code: no index at all. Glean and Perplexity both land on hybrid. Cohere frames it as "From GraphRAG to agentic search" (https://cohere.com/blog/ai-retrieval-graphrag-and-agentic-search — 2025-04-28; body would not render).

**D11 — Prefab eval metrics: a flat "No" vs an entire industry shipping them.** Hamel & Shreya: "No." / "All you get from using these prefab evals is you don't know what they actually do." Against: Ragas, Azure Foundry (11 agent evaluators), MLflow (14+ predefined judges), Galileo, Phoenix, Braintrust autoevals. **Partial reconciliation:** MLflow concedes the point quantitatively — aligned judges cut FP/FN "30-50%" vs generic prompts. Anthropic splits the difference: "You don't need to invent an evaluation from scratch… Use these methods as a foundation, then extend them to your domain."

**D12 — Binary vs Likert.** Hamel: 1–5 scales mean "you're doing it wrong." Eugene Yan: "where possible, I have my evaluators return binary outputs." Databricks: yes/no. Azure Foundry: binary output — **but internally 1–5 with a threshold** for several evaluators, and Customer Satisfaction is straight Likert. Contextual AI's LMUnit is explicitly continuous 1–5. Net: the field converged on **binary at the decision boundary** while several vendors keep a graded score underneath.

**D13 — Reference-free metrics.** Ragas and LlamaIndex both advertise that many modules "do *not* require ground-truth labels." Hamel, Yan, Bergum, and MLflow all argue a judge is meaningless until validated against human labels (TPR/TNR, Cohen's κ, ≥10–20 labeled examples). Bergum's calibrate-small-then-scale method is the practical middle path.

**D14 — Trajectory vs end-state agent eval.** Azure Foundry endorses process evaluation (5 tool-level evaluators). Anthropic explicitly rejected turn-by-turn for their research agent because "agents could follow different valid paths." LangSmith's `subset`/`superset`/`unordered` modes are essentially a compromise for exactly this.

**D15 — LLM-judge trust levels.** Anthropic is the most conservative shipped position ("we do not take eval scores at face value until someone… reads some transcripts"). Eugene Yan's survey supplies the floor: HaluEval best model 58.5% accuracy; 30–60% recall on inconsistency detection despite >95% specificity. Elastic reports ~80% LLM–human agreement on relevance; Bergum reports near-total agreement within one level. **Net: judges are reliable enough for relevance labeling, unreliable for hallucination detection — and vendors ship groundedness judges anyway.**

**D16 — Where the LLM belongs in relevance tuning (a vendor publishing evidence against its own pitch).** Vespa's autoresearch experiment: a free-form LLM agent got the biggest in-domain lift (+8.9%) but retained only **21%** of it out-of-domain; the same agent constrained to Vespa's rank-feature library retained **99%**. Manual tuning retained 80%.

**D17 — Priority on "code execution instead of tool calls."** Cloudflare published 2025-09-26; Anthropic published 2025-11-04 without citing it. Different rationales: Cloudflare = models write TypeScript better than they emit tool-call syntax (a training-data argument); Anthropic = the filesystem enables on-demand tool-definition loading and keeps intermediate data out of context.

---

## SECTION 7 — SOURCES REJECTED FOR QUALITY

- `connorshorten300.medium.com/muvera-with-rajesh-jayaram` — Medium; fails the bar even though the author is Weaviate staff.
- Medium posts by Jagadeesan Ganesh, Nikhil Mogre, Abdullah Grewal/buzzgrewal, HARSHA J S, DhanushKumar/Stackademic, Towards AI — anonymous/individual Medium, explicitly out of scope.
- `towardsdatascience.com` — "How Cursor Actually Indexes Your Codebase" (third-party reverse-engineering); Vespa ODQA repost (aggregator, 2020-era).
- `letsdatascience.com/blog/vector-databases-compared-...` — SEO listicle, unknown author.
- `cipherprojects.com/.../weaviate-vs-qdrant-...` — vendor-adjacent marketing roundup, unknown author.
- `chat-deep.ai/docs/deepseek-vector-database-guide/` — content-farm aggregator.
- `zilliz.com/comparison/weaviate-vs-qdrant` — first-party domain but competitive-marketing page, not engineering content.
- `interestingengineering.substack.com/p/from-bm25-to-agentic-rag` — third-party newsletter.
- KDnuggets "How to Implement Agentic RAG Using LangChain" Parts 1–2 — third-party tutorial site.
- `marktechpost.com`, `pureai.com`, `hyper.ai` (FRAMES writeups) — SEO news aggregators / mirrors.
- `llm-stats.com/benchmarks/frames` — third-party leaderboard scraper.
- `siftq.com/blog/using-frames-benchmark-...` — vendor SEO blog, not on the list.
- `leeroopedia.com` (Ragas agent eval) — auto-generated wiki, unattributed.
- `aws.amazon.com/blogs/machine-learning` (Bedrock + Ragas) — AWS not on the allowed-org list; third-party framing of Ragas.
- `elastic.co` "Agentic RAG with LangChain & Elasticsearch" — not first-party LangChain (used for the LangChain claim it was sought for).
- NVIDIA "Traditional RAG vs. Agentic RAG" — vendor, but not on the allowed list for that claim.
- `forum.weaviate.io` threads, `news.ycombinator.com/item?id=44387617`, `community.databricks.com/t5/technical-blog/...` — community/forum content, not official.
- `github.com/mlbrnm/contextualretrieval`, `github.com/autollama/autollama` — community implementations, not vendor-published.
- `github.com/explodinggradients/ragas` issue #2122 — a user bug report, not documentation.
- `podcasts.chainofthought.xyz/.../jo-kristian-bergum` — third-party podcast summary, not Bergum's own writing.
- Latent Space "Normsky architecture" podcast — third-party podcast (the turbopuffer Latent Space post *was* used because it is hosted on turbopuffer.com as first-party).
- `wowelec.wordpress.com`, `robertheubanks.substack.com` — personal blogs, not named practitioners on the list.
- `newsletter.weaviate.io/p/muvera-...` — newsletter digest; superseded by the primary blog post.
- `news.microsoft.com/source/features/ai/6-surprising-ways-...` — consumer PR feature, no orchestration detail.
- `comet-framer-prod.perplexity.ai` — staging host, not a citable published page.
- `ZenML` LLMOps Database entry on Cursor — secondhand summary; used the Cursor primary instead.
- Educative.io, DataCamp, IBM Think, perfectiongeeks.com, langchain-opentutorial.gitbook.io, buildmvpfast.com, digitalapplied.com, mindstudio.ai, meta-intelligence.tech, codemyspec.com, developersdigest.tech, matthewkruczek.ai, usewire.io, mcp.directory, theunwindai.com — third-party course/SEO/affiliate content.
- X/Twitter and LinkedIn posts (Jerry Liu, Cursor, Cheng Lou, jobergum) — surfaced in search, not fetchable as stable dated content; treated as unverified.
- ~30 arXiv preprints on agentic RAG surveys, benchmarks, and reference architectures (2501.09136, 2504.13587, 2507.09477, 2604.16394, etc.) — academic work outside the stated vendor/practitioner bar. Exceptions made only where the paper *is* the org's first-party publication (OpenAI BrowseComp, Google FRAMES, MMTEB).
- `ir.nist.gov/trec-covid` PDFs, `github.com/castorini/anserini` — IR toolkit/run files, not vendor architecture publications.

---

## SECTION 8 — GAPS AND UNVERIFIED ITEMS (read before publishing anything from this)

**403 / unfetchable first-party pages (findings rest on snippets or substitutes):**
- `openai.com/index/*` (introducing-deep-research, browsecomp, new-tools-for-building-agents, introducing-company-knowledge, memory-and-new-controls-for-chatgpt) — all 403. **The widely-cited Deep Research HLE 26.6% / GAIA figures are NOT verified here and are not asserted.** ChatGPT consumer memory is entirely unverified.
- `help.openai.com/*` — 403. The company-knowledge finding is snippet-based.
- `perplexity.ai/hub/blog/*` — 403 across the board. All Perplexity findings come from `research.perplexity.ai` and `docs.perplexity.ai`, which do render.
- `cdn.openai.com/.../a-practical-guide-to-building-agents.pdf` — returned unparseable binary (7MB). **OpenAI's single-vs-multi-agent guidance is unverified.**
- `techcommunity.microsoft.com/.../researcher-agent-...` — renders title only. The ΔI<ε stopping rule is snippet-sourced.
- `contextual.ai/new/groundedness-scoring-...`, `learn.microsoft.com/.../groundedness`, Google `dynamicRetrievalConfig` blog, Google FRAMES, MMTEB paper — snippet-only.
- `cohere.com/blog/ai-retrieval-graphrag-and-agentic-search` and `cohere.com/blog/rerank-4` — nav/header only. **Cohere North orchestration and Rerank v4.0 benchmarks are genuine gaps.** Cohere Rerank 3.5's date is contradictory (page says 2024-12-02, snippet said 2025-07-10).
- 404s (URL guesses, do not cite): `developers.openai.com/blog/agentkit/`, `docs.cloud.google.com/gemini-enterprise/docs/overview`, `redis.io/blog/`, `docs.letta.com/memory/memfs`, `devin.ai/blog/why-we-built-riptide`, `devin.ai/blog/codemaps`, several `learn.microsoft.com/agent-framework/.../memory` paths, `docs.cohere.com/docs/rerank-best-practices`, W&B Weave scorers (403), Arize Phoenix pre-tested-evals tables (404).

**Uncovered orgs:** Meta AI (no first-party agentic-retrieval product documentation found). Cohere North. W&B Weave. Patronus AI. Redis. Zilliz/Milvus (page rendered title-only; RRF/WeightedRanker support is snippet-level only). Databricks long-context-RAG posts exist but were not fetched — **do not cite Databricks long-context numbers from this report** (the Instructed Retriever numbers in §1.4 *were* fetched and are solid).

**Status not stated on source page:** Vertex AI Agent Engine Memory Bank GA-vs-preview. Google check-grounding GA-vs-preview (inferred GA from context).

**Under-searched, not proven absent:** HyDE as a shipped vendor feature. Provence-style context pruning as a vendor feature (only Chroma Context-1 and Weaviate guidance verified). Windsurf's original retrieval posts are gone post-acquisition (404), so their position is only reachable through Cognition's SWE-grep/SWE-1.5 posts.

**Budget note:** the session's 200 WebSearch calls were exhausted mid-research. Late-stage sources were reached by direct URL fetch. Any finding marked `[snippet-only]` should be re-verified before publication.