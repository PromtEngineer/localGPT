import logging
from typing import List, Dict
from textwrap import shorten

logger = logging.getLogger("rag-system")

# Global log format – only set if user has not configured logging
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def log_query(query: str, sub_queries: List[str] | None = None) -> None:
    """Emit a nicely-formatted block describing the incoming query and any
    decomposition."""
    border = "=" * 60
    logger.info("\n%s\nUSER QUERY: %s", border, query)
    if sub_queries:
        for i, q in enumerate(sub_queries, 1):
            logger.info("  sub-%d → %s", i, q)
    logger.info("%s", border)


def log_retrieval_results(results: List[Dict], k: int) -> None:
    """Show chunk_id, truncated text and score for the first *k* rows."""
    if not results:
        logger.info("Retrieval returned 0 documents.")
        return
    logger.info("Top %d results:", min(k, len(results)))
    header = f"{'chunk_id':<14} {'score':<7} preview"
    logger.info(header)
    logger.info("-" * len(header))
    for row in results[:k]:
        preview = shorten(row.get("text", ""), width=60, placeholder="…")
        logger.info("%s %-7.3f %s", str(row.get("chunk_id"))[:12], row.get("score", 0.0), preview) 