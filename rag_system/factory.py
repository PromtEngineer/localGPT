import copy

from dotenv import load_dotenv


def _build_llm_client():
    """Create the LLM client and its config for the active LLM_BACKEND.

    Uses local imports to prevent circular dependencies with rag_system.main.
    """
    from rag_system.main import LLM_BACKEND, OLLAMA_CONFIG, WATSONX_CONFIG

    if LLM_BACKEND.lower() == "watsonx":
        from rag_system.utils.watsonx_client import WatsonXClient

        if not WATSONX_CONFIG["api_key"] or not WATSONX_CONFIG["project_id"]:
            raise ValueError(
                "Watson X configuration incomplete. Please set WATSONX_API_KEY and WATSONX_PROJECT_ID "
                "environment variables."
            )

        client = WatsonXClient(
            api_key=WATSONX_CONFIG["api_key"],
            project_id=WATSONX_CONFIG["project_id"],
            url=WATSONX_CONFIG["url"],
        )
        return client, WATSONX_CONFIG

    from rag_system.utils.ollama_client import OllamaClient

    return OllamaClient(host=OLLAMA_CONFIG["host"]), OLLAMA_CONFIG


def get_pipeline_config(mode: str = "default") -> dict:
    """Return a deep copy of a pipeline profile so callers cannot mutate the master config."""
    from rag_system.main import PIPELINE_CONFIGS

    return copy.deepcopy(PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS["default"]))


def get_agent(mode: str = "default"):
    """Factory function to get an instance of the RAG agent for the specified mode."""
    from rag_system.agent.loop import Agent

    load_dotenv()

    llm_client, llm_config = _build_llm_client()
    config = get_pipeline_config(mode)

    return Agent(
        pipeline_configs=config,
        llm_client=llm_client,
        ollama_config=llm_config,
    )


def get_indexing_pipeline(mode: str = "default"):
    """Factory function to get an instance of the Indexing Pipeline for the specified mode."""
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline

    load_dotenv()

    llm_client, llm_config = _build_llm_client()
    config = get_pipeline_config(mode)

    return IndexingPipeline(config, llm_client, llm_config)
