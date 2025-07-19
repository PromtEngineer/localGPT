from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str, chat_history: List[Dict[str, Any]] | None = None) -> List[str]:
        """Decompose *query* into standalone sub-queries.

        Parameters
        ----------
        query : str
            The latest user message.
        chat_history : list[dict] | None
            Recent conversation turns (each item should contain at least the original
            user query under the key ``"query"``). Only the **last 5** turns are
            included to keep the prompt short.
        """

        # ---- Limit history to last 5 user turns and extract the queries ----
        history_snippets: List[str] = []
        if chat_history:
            # Keep only the last 5 turns
            recent_turns = chat_history[-5:]
            # Extract user queries (fallback: full dict as string if key missing)
            for turn in recent_turns:
                history_snippets.append(str(turn.get("query", turn)))

        # Serialize chat_history for the prompt (single string)
        chat_history_text = " | ".join(history_snippets)

        # ---- Build the new SYSTEM prompt with added legacy examples ----
        system_prompt = """
You are an expert at query decomposition for a Retrieval-Augmented Generation (RAG) system.

Return one RFC-8259-compliant JSON object and nothing else.
Schema:
{
“requires_decomposition”: <bool>,
“reasoning”:              <string>,  // ≤ 50 words
“resolved_query”:         <string>,  // query after context resolution
“sub_queries”:            <string[]> // 1–10 standalone items
}

Think step-by-step internally, but reveal only the concise reasoning.

⸻

Context Resolution  (perform FIRST)

You will receive:
	•	query – the current user message
	•	chat_history – the most recent user turns (may be empty)

If query contains pronouns, ellipsis, or shorthand that can be unambiguously linked to something in chat_history, rewrite it to a fully self-contained question and place the result in resolved_query.
Otherwise, copy query into resolved_query unchanged.

⸻

When is decomposition REQUIRED?
	•	MULTI-PART questions joined by “and”, “or”, “also”, list commas, etc.
	•	COMPARATIVE / SUPERLATIVE questions (two or more entities, e.g. “bigger, better, fastest”).
	•	TEMPORAL / SEQUENTIAL questions (changes over time, event timelines).
	•	ENUMERATIONS (pros, cons, impacts).
	•	ENTITY-SET COMPARISONS (A, B, C revenue…).

When is decomposition NOT REQUIRED?
	•	A single, factual information need.
	•	Ambiguous queries needing clarification rather than splitting.

⸻

Output rules
	1.	Use resolved_query—not the raw query—to decide on decomposition.
	2.	If requires_decomposition is false, sub_queries must contain exactly resolved_query.
	3.	Otherwise, produce 2–10 self-contained questions; avoid pronouns and shared context.

⸻
"""

        # ---- Append NEW examples provided by the user ----
        new_examples = """

Normalise pronouns and references: turn “this paper” into the explicit title if it can be inferred, otherwise leave as-is.
chat_history: “What is the email address of the computer vision consultants?”
query: “What is their revenue?”

{
  "requires_decomposition": false,
  "reasoning": "Pronoun resolved; single information need.",
  "resolved_query": "What is the revenue of the computer vision consultants?",
  "sub_queries": [
    "What is the revenue of the computer vision consultants?"
  ]
}

Context resolution (single info need)
chat_history: “What is the email address of the computer vision consultants?”
query: “What is the address?”

{
  "requires_decomposition": false,
  "reasoning": "Pronoun resolved; single information need.",
  "resolved_query": "What is the physical address of the computer vision consultants?",
  "sub_queries": [
    "What is the physical address of the computer vision consultants?"
  ]
}

Context resolution (single info need)
chat_history: “ComputeX has a revenue of 100M?”
query: “Who is the CEO?”

{
  "requires_decomposition": false,
  "reasoning": "entities normalization.",
  "resolved_query": "who is the CEO of ComputeX",
  "sub_queries": [
    "who is the CEO of ComputeX"
  ]
}

No unique antecedent → leave unresolved
chat_history: “Tell me about the paper.”
query: “What is the address?”

{
  "requires_decomposition": false,
  "reasoning": "Ambiguous reference; cannot resolve safely.",
  "resolved_query": "What is the address?",
  "sub_queries": ["What is the address?"]
}

Temporal + Comparative
chat_history: ""
query: “How did Nvidia’s 2024 revenue compare with 2023?”

{
  "requires_decomposition": true,
  "reasoning": "Needs revenue for two separate years before comparison.",
  "resolved_query": "How did Nvidia’s 2024 revenue compare with 2023?",
  "sub_queries": [
    "What was Nvidia’s revenue in 2024?",
    "What was Nvidia’s revenue in 2023?"
  ]
}

Enumeration (pros / cons / cost)
chat_history: ""
query: “List the pros, cons, and estimated implementation cost of adopting a vector database.”

{
  "requires_decomposition": true,
  "reasoning": "Three distinct information needs: pros, cons, cost.",
  "resolved_query": "List the pros, cons, and estimated implementation cost of adopting a vector database.",
  "sub_queries": [
    "What are the pros of adopting a vector database?",
    "What are the cons of adopting a vector database?",
    "What is the estimated implementation cost of adopting a vector database?"
  ]
}

Entity-set comparison (multiple companies)
chat_history: ""
query: “How did Nvidia, AMD, and Intel perform in Q2 2025 in terms of revenue?”

{
  "requires_decomposition": true,
  "reasoning": "Need revenue for each of three entities before comparison.",
  "resolved_query": "How did Nvidia, AMD, and Intel perform in Q2 2025 in terms of revenue?",
  "sub_queries": [
    "What was Nvidia's revenue in Q2 2025?",
    "What was AMD's revenue in Q2 2025?",
    "What was Intel's revenue in Q2 2025?"
  ]
}

Multi-part question (limitations + mitigations)
chat_history: ""
query: “What are the limitations of GPT-4o and what are the recommended mitigations?”

{
  "requires_decomposition": true,
  "reasoning": "Two distinct pieces of information: limitations and mitigations.",
  "resolved_query": "What are the limitations of GPT-4o and what are the recommended mitigations?",
  "sub_queries": [
    "What are the known limitations of GPT-4o?",
    "What are the recommended mitigations for the limitations of GPT-4o?"
  ]
}
"""

        # ---- Append legacy examples that already existed in the old prompt ----
        legacy_examples_header = """
⸻

Additional legacy examples
"""

        legacy_examples_body = """
**Example 1: Multi-Part Query**
Query: "What were the main findings of the aiconfig report and how do they compare to the results from the RAG paper?"
JSON Output:
{
  "reasoning": "The query asks for two distinct pieces of information: the findings from one report and a comparison to another. This requires two separate retrieval steps.",
  "sub_queries": [
    "What were the main findings of the aiconfig report?",
    "How do the findings of the aiconfig report compare to the results from the RAG paper?"
  ]
}

**Example 2: Simple Query**
Query: "Summarize the contributions of the DeepSeek-V3 paper."
JSON Output:
{
  "reasoning": "This is a direct request for a summary of a single document and does not contain multiple parts.",
  "sub_queries": [
    "Summarize the contributions of the DeepSeek-V3 paper."
  ]
}

**Example 3: Comparative Query**
Query: "Did Microsoft or Google make more money last year?"
JSON Output:
{
  "reasoning": "This is a comparative query that requires fetching the profit for each company before a comparison can be made.",
  "sub_queries": [
    "How much profit did Microsoft make last year?",
    "How much profit did Google make last year?"
  ]
}

**Example 4: Comparative Query with different phrasing**
Query: "Who has more siblings, Jamie or Sansa?"
JSON Output:
{
  "reasoning": "This comparative query needs the sibling count for both individuals to be answered.",
  "sub_queries": [
    "How many siblings does Jamie have?",
    "How many siblings does Sansa have?"
  ]
}
"""

        full_prompt = (
            system_prompt
            + new_examples
            # + legacy_examples_header
            # + legacy_examples_body
            + """

⸻

Now process

Input payload:

""" + json.dumps({"query": query, "chat_history": chat_history_text}, indent=2) + """
"""
        )

        # ---- Call the LLM ----
        response = self.llm_client.generate_completion(self.llm_model, full_prompt, format="json")

        response_text = response.get('response', '{}')
        try:
            # Handle potential markdown code blocks in the response
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()

            data = json.loads(response_text)

            sub_queries = data.get('sub_queries') or [query]
            reasoning = data.get('reasoning', 'No reasoning provided.')

            print(f"Query Decomposition Reasoning: {reasoning}")

            # Fallback: ensure at least the resolved_query if sub_queries empty
            if not sub_queries:
                sub_queries = [data.get('resolved_query', query)]

            # Deduplicate while preserving order
            sub_queries = list(dict.fromkeys(sub_queries))

            # Enforce 10 sub-query limit per new requirements
            return sub_queries[:10]
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from query decomposer: {response_text}")
            return [query]

class HyDEGenerator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def generate(self, query: str) -> str:
        prompt = f"Generate a short, hypothetical document that answers the following question. The document should be dense with keywords and concepts related to the query.\n\nQuery: {query}\n\nHypothetical Document:"
        response = self.llm_client.generate_completion(self.llm_model, prompt)
        return response.get('response', '')

class GraphQueryTranslator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def _generate_translation_prompt(self, query: str) -> str:
        return f"""
You are an expert query planner. Convert the user's question into a structured JSON query for a knowledge graph.
The JSON should contain a 'start_node' (the known entity in the query) and an 'edge_label' (the relationship being asked about).
The graph has nodes (entities) and directed edges (relationships). For example, (Tim Cook) -[IS_CEO_OF]-> (Apple).
Return ONLY the JSON object.

User Question: "{query}"

JSON Output:
"""

    def translate(self, query: str) -> Dict[str, Any]:
        prompt = self._generate_translation_prompt(query)
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            return json.loads(response.get('response', '{}'))
        except json.JSONDecodeError:
            return {}