from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str, chat_history: List[Dict[str, Any]] | None = None, max_sub_queries: int = 10) -> List[str]:
        """Decompose *query* into standalone sub-queries.

        Two prompt variants, selected on whether conversation history exists:

        - `_decompose_single_turn` — the FROZEN prompt every single-turn bench
          number was measured against (arm C..K). It must stay byte-identical:
          even cosmetic edits (smart quotes, added examples) were measured to
          shift temp-0 decompositions on 5-25/120 gold queries and cost 2
          Sonnet-confirmed rfc rows (arm L, 2026-08-16). Any change here
          re-triggers the full 5-bench gate.
        - `_decompose_multi_turn` — adds last_assistant_answer (the decomposer
          cannot otherwise resolve pronouns whose antecedent the ASSISTANT
          introduced; measured wrong-entity resolution on multiturn.jsonl
          mt_07) plus the ellipsis rule 1b (mt_09). History is user-queries
          only: interleaving answers made the 4b model substitute a previous
          turn's question (m1 arm, 11/12).
        """
        if chat_history:
            return self._decompose_multi_turn(query, chat_history, max_sub_queries)
        return self._decompose_single_turn(query, max_sub_queries=max_sub_queries)

    def _decompose_multi_turn(self, query: str, chat_history: List[Dict[str, Any]], max_sub_queries: int = 10) -> List[str]:
        """Decompose *query* into standalone sub-queries.

        Parameters
        ----------
        query : str
            The latest user message.
        chat_history : list[dict] | None
            Recent conversation turns (each item should contain at least the original
            user query under the key ``"query"``). Only the **last 5** turns are
            included to keep the prompt short.
        max_sub_queries : int
            Hard cap on the returned sub-queries (``query_decomposition.max_sub_queries``).
        """

        # ---- History for context resolution: the last 5 user queries, plus
        # ONLY the last assistant answer as a separate field. Without any
        # answer the decomposer cannot resolve pronouns whose antecedent was
        # introduced by the assistant (user asks "who is the largest
        # supplier?" -> answer names Acme -> "their lead time?" has no
        # antecedent in user turns alone; measured resolving to the wrong
        # entity on eval/goldset/multiturn.jsonl mt_07). Interleaving every
        # answer into chat_history was measured worse: the 4b decomposer
        # anchored on a previous turn and substituted its query (mt_09).
        history_snippets: List[str] = []
        last_assistant_answer = ""
        if chat_history:
            recent_turns = chat_history[-5:]
            for turn in recent_turns:
                history_snippets.append(
                    str(turn.get("query", turn)) if isinstance(turn, dict) else str(turn))
            last_assistant_answer = " ".join(
                str(recent_turns[-1].get("answer", "") or "").split()) if isinstance(recent_turns[-1], dict) else ""
            if len(last_assistant_answer) > 300:
                last_assistant_answer = last_assistant_answer[:300] + "..."

        # Serialize chat_history for the prompt (single string)
        chat_history_text = " | ".join(history_snippets)

        # ---- Build the SYSTEM prompt ----
        system_prompt = """
You are an expert at query decomposition for a Retrieval-Augmented Generation (RAG) system.

Return one RFC-8259-compliant JSON object and nothing else.
Schema:
{
"requires_decomposition": <bool>,
"reasoning":              <string>,  // ≤ 50 words
"resolved_query":         <string>,  // query after context resolution
"sub_queries":            <string[]> // 1–10 standalone items
}

Think step-by-step internally, but reveal only the concise reasoning.

⸻

Context Resolution  (perform FIRST)

You will receive:
	•	query – the current user message
	•	chat_history – the most recent user turns (may be empty)
	•	last_assistant_answer – what the assistant replied to the most recent turn (may be empty); use it only to resolve references that chat_history alone cannot

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
	1b.	resolved_query must ask for the SAME fact as query. Never substitute an earlier question from chat_history: a follow-up like "And when was it approved?" asks a NEW fact even when it continues the previous topic.
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

Context resolution (antecedent introduced by the assistant's answer)
chat_history: "Which company is the largest supplier?"
last_assistant_answer: "Acme Industrial is the largest supplier, providing 40 percent of volume."
query: "What is their lead time?"

{
  "requires_decomposition": false,
  "reasoning": "Pronoun resolved via the assistant's answer; single information need.",
  "resolved_query": "What is the lead time of Acme Industrial?",
  "sub_queries": [
    "What is the lead time of Acme Industrial?"
  ]
}

Ellipsis continuation asks a NEW fact (do not repeat the previous question)
chat_history: "What does a building permit cost? | When was the permit application submitted?"
last_assistant_answer: "The permit application was submitted on 3 March 2024."
query: "And when was it approved?"

{
  "requires_decomposition": false,
  "reasoning": "Follow-up introduces a new fact (approval date); ellipsis resolved without repeating the earlier question.",
  "resolved_query": "When was the building permit application approved?",
  "sub_queries": [
    "When was the building permit application approved?"
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

        full_prompt = (
            system_prompt
            + new_examples
            + """

⸻

Now process

Input payload:

""" + json.dumps({"query": query, "chat_history": chat_history_text,
                  "last_assistant_answer": last_assistant_answer}, indent=2) + """
"""
        )

        # ---- Call the LLM ----
        # Greedy decode: sampled decomposition made the SAME query split
        # differently run-to-run, which both destabilizes answers and made
        # every synthesis A/B compare partially-different row sets.
        response = self.llm_client.generate_completion(
            self.llm_model, full_prompt, format="json",
            options={"temperature": 0})

        response_text = response.get('response', '{}')
        try:
            # Handle potential markdown code blocks in the response
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()

            data = json.loads(response_text)

            sub_queries = data.get('sub_queries') or [query]
            reasoning = data.get('reasoning', 'No reasoning provided.')

            print(f"Query Decomposition Reasoning: {reasoning}")

            # Deduplicate while preserving order
            sub_queries = list(dict.fromkeys(sub_queries))

            return sub_queries[:max(1, int(max_sub_queries))]
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from query decomposer: {response_text}")
            return [query]

# GraphQueryTranslator was removed on 2026-08-09 (roadmap item 2.5) together with
# GraphExtractor and GraphRetriever. Evidence: Documentation/research/
# academic-evidence-2026.md §6 — GraphRAG loses on single-hop, its multi-hop
# gains are contested, and it costs 41–57x at indexing and up to ~377x in query
# tokens. Nothing in this repo ever armed it.
    def _decompose_single_turn(self, query: str, max_sub_queries: int = 10) -> List[str]:
        """Decompose *query* into standalone sub-queries.

        Parameters
        ----------
        query : str
            The latest user message.
        max_sub_queries : int
            Hard cap on the returned sub-queries (``query_decomposition.max_sub_queries``).
        """

        # ---- Build the SYSTEM prompt ----
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

        full_prompt = (
            system_prompt
            + new_examples
            + """

⸻

Now process

Input payload:

""" + json.dumps({"query": query, "chat_history": ""}, indent=2) + """
"""
        )

        # ---- Call the LLM ----
        # Greedy decode: sampled decomposition made the SAME query split
        # differently run-to-run, which both destabilizes answers and made
        # every synthesis A/B compare partially-different row sets.
        response = self.llm_client.generate_completion(
            self.llm_model, full_prompt, format="json",
            options={"temperature": 0})

        response_text = response.get('response', '{}')
        try:
            # Handle potential markdown code blocks in the response
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()

            data = json.loads(response_text)

            sub_queries = data.get('sub_queries') or [query]
            reasoning = data.get('reasoning', 'No reasoning provided.')

            print(f"Query Decomposition Reasoning: {reasoning}")

            # Deduplicate while preserving order
            sub_queries = list(dict.fromkeys(sub_queries))

            return sub_queries[:max(1, int(max_sub_queries))]
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from query decomposer: {response_text}")
            return [query]

# GraphQueryTranslator was removed on 2026-08-09 (roadmap item 2.5) together with
# GraphExtractor and GraphRetriever. Evidence: Documentation/research/
# academic-evidence-2026.md §6 — GraphRAG loses on single-hop, its multi-hop
# gains are contested, and it costs 41–57x at indexing and up to ~377x in query
# tokens. Nothing in this repo ever armed it.