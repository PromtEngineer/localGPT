from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional, Tuple

class CrossEncoderReranker:
    """
    A cross-encoder reranker backed by a local Hugging Face sequence-classification model.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Initializing cross-encoder reranker with model '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else None,
        ).to(self.device).eval()
        
        print("Cross-encoder reranker loaded successfully.")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5, *, early_exit: bool = True, margin: float = 0.4, min_scored: int = 8, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.

        If *early_exit* is True the cross-encoder scores documents in mini-batches and
        stops once the best-so-far score beats the worst-so-far by *margin* after at
        least *min_scored* docs have been processed.  This accelerates "easy" queries
        where strong positives dominate.
        """
        if not documents:
            return []

        # Sort by the upstream (hybrid) score so that the strongest candidates are evaluated first.
        docs_sorted = sorted(documents, key=lambda d: d.get('score', 0.0), reverse=True)

        scored_pairs: List[tuple[float, Dict[str, Any]]] = []

        with torch.no_grad():
            for start in range(0, len(docs_sorted), batch_size):
                batch_docs = docs_sorted[start : start + batch_size]
                batch_pairs = [[query, d['text']] for d in batch_docs]

                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)

                logits = self.model(**inputs).logits.view(-1)
                batch_scores = logits.float().cpu().tolist()

                scored_pairs.extend(zip(batch_scores, batch_docs))

                # --- Early-exit check ---
                if early_exit and len(scored_pairs) >= min_scored:
                    # Current best and worst among *already* scored docs
                    best_score = max(scored_pairs, key=lambda x: x[0])[0]
                    worst_score = min(scored_pairs, key=lambda x: x[0])[0]
                    if best_score - worst_score >= margin:
                        break

        # Sort final set and attach scores
        sorted_by_score = sorted(scored_pairs, key=lambda x: x[0], reverse=True)
        reranked_docs: List[Dict[str, Any]] = []
        for score, doc in sorted_by_score[:top_k]:
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = score
            reranked_docs.append(doc_with_score)

        return reranked_docs

def is_qwen3_reranker(model_name: str) -> bool:
    """True for the Qwen3-Reranker family (causal-LM yes/no scorers)."""
    return "qwen3-reranker" in (model_name or "").lower()


class QwenRerankerScorer:
    """
    Reranker for the ``Qwen/Qwen3-Reranker-*`` family.

    These are **causal LMs**, not ``AutoModelForSequenceClassification`` models.
    Loading them through the `rerankers` library's cross-encoder path builds a
    ``Qwen3ForSequenceClassification`` with a **randomly initialised** ``score``
    head, which produces meaningless (untrained) scores.  This class implements
    the scoring scheme published on the model card instead: the query/document
    pair is wrapped in the model's chat template, the model is asked whether the
    document satisfies the query, and the score is the probability of the "yes"
    token against the "no" token at the final position.

    Interface mirrors what ``RetrievalPipeline`` and ``eval/run_eval.py`` expect
    from the `rerankers` lib branch: ``rank(query=..., docs=[...])`` returns a
    list of ``(score, original_index)`` tuples sorted by score, descending.
    """

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        instruction: Optional[str] = None,
        max_length: int = 2048,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_name = model_name
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Initializing Qwen3 reranker '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        print("Qwen3 reranker loaded successfully.")

    # -- internals ---------------------------------------------------------

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _score_batch(self, pairs: List[str]) -> List[float]:
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        enc = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max(budget, 16),
        )
        enc["input_ids"] = [
            self.prefix_tokens + ids + self.suffix_tokens for ids in enc["input_ids"]
        ]
        enc = self.tokenizer.pad(enc, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        stacked = torch.stack(
            [logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1
        )
        probs = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return probs.cpu().tolist()

    # -- public API --------------------------------------------------------

    def score(self, query: str, docs: List[str]) -> List[float]:
        """Relevance probability in [0, 1], one per document, in input order."""
        scores: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            scores.extend(self._score_batch([self._format_pair(query, d) for d in batch]))
        return scores

    def rank(self, query: str, docs: List[str], top_k: Optional[int] = None
             ) -> List[Tuple[float, int]]:
        """``[(score, original_index), …]`` sorted by score, descending."""
        if not docs:
            return []
        scores = self.score(query, docs)
        pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        out = [(score, idx) for idx, score in pairs]
        return out[:top_k] if top_k else out

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5,
               **_ignored) -> List[Dict[str, Any]]:
        """Dict-in/dict-out form, matching ``CrossEncoderReranker.rerank``."""
        if not documents:
            return []
        pairs = self.rank(query, [d["text"] for d in documents], top_k=top_k)
        return [documents[idx] | {"rerank_score": score} for score, idx in pairs]


if __name__ == '__main__':
    # This test requires an internet connection to download the models.
    try:
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
        
        query = "What is the capital of France?"
        documents = [
            {'text': "Paris is the capital of France.", 'metadata': {'doc_id': 'a'}},
            {'text': "The Eiffel Tower is in Paris.", 'metadata': {'doc_id': 'b'}},
            {'text': "France is a country in Europe.", 'metadata': {'doc_id': 'c'}},
        ]
        
        reranked_documents = reranker.rerank(query, documents)
        
        print("\n--- Verification ---")
        print(f"Query: {query}")
        print("Reranked documents:")
        for doc in reranked_documents:
            print(f"  - Score: {doc['rerank_score']:.4f}, Text: {doc['text']}")

    except Exception as e:
        print(f"\nAn error occurred during the CrossEncoderReranker test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
