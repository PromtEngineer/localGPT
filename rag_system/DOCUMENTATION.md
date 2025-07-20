# RAG System Documentation

This document provides a detailed overview of the RAG (Retrieval-Augmented Generation) system, its architecture, and how to use it.

## System Overview

This RAG system is a sophisticated, multimodal question-answering system designed to work with a variety of documents. It can understand and process both the text and the visual layout of documents, and it uses a knowledge graph to understand the relationships between the entities in the documents.

The system is built around an agentic workflow that allows it to:

*   **Decompose complex questions** into smaller, more manageable sub-questions.
*   **Triage queries** to determine if they can be answered directly or if they require retrieval from the knowledge base.
*   **Verify answers** against the retrieved context to ensure they are accurate and supported by the documents.

## Architecture

The system is composed of two main pipelines: an indexing pipeline and a retrieval pipeline.

### Indexing Pipeline

The indexing pipeline is responsible for processing the documents and building the knowledge base. It performs the following steps:

1.  **Text Extraction**: The pipeline uses `PyMuPDF` to extract the text from each page of the PDF documents, preserving the original layout.
2.  **Text Embedding**: The extracted text is then passed to a text embedding model (`Qwen/Qwen3-Embedding-0.6B`) to create numerical vector representations of the text.
3.  **Knowledge Graph Creation**: The text is also passed to a `GraphExtractor` that uses a large language model (`qwen2.5vl:7b`) to extract entities and their relationships. This information is then used to build a knowledge graph, which is stored as a `.gml` file.
4.  **Indexing**: The text embeddings and the knowledge graph are then stored in a LanceDB database.

### Retrieval Pipeline

The retrieval pipeline is responsible for answering user queries. It uses an agentic workflow that includes the following steps:

1.  **Triage**: The agent first triages the user's query to determine if it can be answered directly or if it requires retrieval from the knowledge base.
2.  **Query Decomposition**: If the query is complex, the agent uses a `QueryDecomposer` to break it down into smaller, more manageable sub-questions.
3.  **Retrieval**: The agent then uses a `MultiVectorRetriever` and a `GraphRetriever` to retrieve relevant information from the knowledge base.
4.  **Verification**: The retrieved context is then passed to a `Verifier` that uses an LLM to check if the context is sufficient to answer the query.
5.  **Synthesis**: Finally, the agent uses an LLM to synthesize a final answer from the verified context.

## API Endpoints

The system provides the following command-line endpoints:

*   `index`: This endpoint runs the indexing pipeline to process the documents and build the knowledge base.
*   `chat`: This endpoint runs the retrieval pipeline to answer a user's query.
*   `show_graph`: This endpoint displays the knowledge graph in a human-readable format and also provides a visual representation of the graph.

### Usage

To run the system, use the following commands:

```bash
# Activate the virtual environment
source rag_system/rag_venv/bin/activate

# Index the documents
python rag_system/main.py index

# Ask a question
python rag_system/main.py chat "Your question here"

# Show the knowledge graph
python rag_system/main.py show_graph
```
