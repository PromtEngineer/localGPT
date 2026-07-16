from __future__ import annotations

import asyncio

from backend.agent_runtime.builtin_tools import RagRetrievalClient


class _Database:
    def get_indexes_for_session(self, session_id):
        assert session_id == "session-1"
        return ["index-a", "index-b"]

    def get_index(self, index_id):
        return {
            "index-a": {"vector_table_name": "table_a"},
            "index-b": {"vector_table_name": "table_b"},
        }[index_id]


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"answer": "grounded", "source_documents": []}


def test_retrieval_client_sends_backend_owned_table_scope(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("backend.agent_runtime.builtin_tools.requests.post", fake_post)
    client = RagRetrievalClient("http://rag.test", database=_Database())

    result = asyncio.run(
        client.search(
            "question",
            session_id="session-1",
            retrieval_k=5,
            options={"table_names": ["untrusted_override"]},
        )
    )

    assert result["answer"] == "grounded"
    assert captured["url"] == "http://rag.test/chat"
    assert captured["json"]["table_names"] == ["table_a", "table_b"]
