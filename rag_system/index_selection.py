"""Single source of truth for which linked index a session chats against.

When a session has multiple indexes linked, the backend resolves the vector
TABLE from this choice while the RAG server resolves the EMBEDDING MODEL and
fusion config from it. The two must agree: picking different indexes embeds
queries with the wrong model for the table being searched, which silently
returns garbage. They diverged once before (backend used the last-linked
index, the RAG server the first) — both now call this helper, and a
regression test trips if either reintroduces its own pick.

This module must stay dependency-free so anything can import it cheaply.
"""
from typing import Optional, Sequence


def select_active_index_id(idx_ids: Sequence[str]) -> Optional[str]:
    """Return the id of the index whose table/model/config a chat should use.

    Policy: the most recently linked index wins. Returns None when the
    session has no linked indexes.
    """
    if not idx_ids:
        return None
    return idx_ids[-1]
