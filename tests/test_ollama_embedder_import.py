def test_ollama_embedder_does_not_require_transformers(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("transformers") or name == "torch":
            raise AssertionError(f"eager heavyweight import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    from rag_system.indexing.representations import OllamaEmbedder

    assert OllamaEmbedder("nomic-embed-text").model_name == "nomic-embed-text"
