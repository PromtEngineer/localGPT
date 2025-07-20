# Multimodal RAG System

This document provides a detailed overview of the multimodal Retrieval-Augmented Generation (RAG) system implemented in this directory. The system is designed to process and understand information from PDF documents, combining both textual and visual data to answer complex queries.

## 1. Overview

This RAG system is a sophisticated pipeline that leverages state-of-the-art open-source models to provide accurate, context-aware answers from a document corpus. Unlike traditional RAG systems that only process text, this implementation is fully multimodal. It extracts and indexes both text and images from PDFs, allowing a Vision Language Model (VLM) to reason over both modalities when generating a final answer.

The core capabilities include:
-   **Multimodal Indexing**: Extracts text and images from PDFs and creates separate vector embeddings for each.
-   **Hybrid Retrieval**: Combines dense vector search (for semantic similarity) with traditional keyword-based search (BM25) for robust retrieval.
-   **Advanced Reranking**: Utilizes a powerful reranker model to improve the relevance of retrieved documents before they are passed to the generator.
-   **VLM-Powered Synthesis**: Employs a Vision Language Model to synthesize the final answer, allowing it to analyze both the text and the images from the retrieved document chunks.

## 2. Architecture

The system is composed of several key Python modules that work together to form the RAG pipeline.

### Key Modules:

-   `main.py`: The main entry point for the application. It contains the configuration for all models and pipelines and orchestrates the indexing and retrieval processes.
-   `rag_system/pipelines/`: Contains the high-level orchestration for indexing and retrieval.
    -   `indexing_pipeline.py`: Manages the process of converting raw PDFs into indexed, searchable data.
    -   `retrieval_pipeline.py`: Handles the end-to-end process of taking a user query, retrieving relevant information, and generating a final answer.
-   `rag_system/indexing/`: Contains all modules related to data processing and indexing.
    -   `multimodal.py`: Responsible for extracting text and images from PDFs and generating embeddings using the configured vision model (`colqwen2-v1.0`).
    -   `representations.py`: Defines the text embedding model (`Qwen2-7B-instruct`) and other data representation generators.
    -   `embedders.py`: Manages the connection to the **LanceDB** vector database and handles the indexing of vector embeddings.
-   `rag_system/retrieval/`: Contains modules for retrieving and ranking documents.
    -   `retrievers.py`: Implements the logic for searching the vector database to find relevant text and image chunks.
    -   `reranker.py`: Contains the `QwenReranker` class, which re-ranks the retrieved documents for improved relevance.
-   `rag_system/agent/`: Contains the `Agent` loop that interacts with the user and the RAG pipelines.
-   `rag_system/utils/`: Contains utility clients, such as the `OllamaClient` for interacting with the Ollama server.

### Data Flow:

1.  **Indexing**:
    -   The `MultimodalProcessor` reads a PDF and splits it into pages.
    -   For each page, it extracts the raw text and a full-page image.
    -   The `QwenEmbedder` generates a vector embedding for the text.
    -   The `LocalVisionModel` (using `colqwen2-v1.0`) generates a vector embedding for the image.
    -   The `VectorIndexer` stores these embeddings in separate tables within a **LanceDB** database.
2.  **Retrieval**:
    -   A user submits a query to the `Agent`.
    -   The `RetrievalPipeline`'s `MultiVectorRetriever` searches both the text and image tables in LanceDB for relevant chunks.
    -   The retrieved documents are passed to the `QwenReranker`, which re-orders them based on relevance to the query.
    -   The top-ranked documents (containing both text and image references) are passed to the Vision Language Model (`qwen-vl`).
    -   The VLM analyzes the text and images to extract key facts.
    -   A final text generation model (`llama3`) synthesizes these facts into a coherent, human-readable answer.

## 3. Models

This system relies on a suite of powerful, open-source models.

| Component             | Model                               | Framework      | Purpose                                     |
| --------------------- | ----------------------------------- | -------------- | ------------------------------------------- |
| **Image Embedding**   | `vidore/colqwen2-v1.0`              | `colpali`      | Generates vector embeddings from images.    |
| **Text Embedding**    | `Qwen/Qwen2-7B-instruct`            | `transformers` | Generates vector embeddings from text.      |
| **Reranker**          | `Qwen/Qwen-reranker`                | `transformers` | Re-ranks retrieved documents for relevance. |
| **Vision Language Model** | `qwen2.5vl:7b`                      | `Ollama`       | Extracts facts from text and images.        |
| **Text Generation**   | `llama3`                            | `Ollama`       | Synthesizes the final answer.               |

## 4. Configuration

All system configurations are centralized in `main.py`.

-   **`OLLAMA_CONFIG`**: Defines the models that will be run via the Ollama server. This includes the final text generation model and the Vision Language Model.
-   **`PIPELINE_CONFIGS`**: Contains the configurations for both the `indexing` and `retrieval` pipelines. Here you can specify:
    -   The paths for the LanceDB database and source documents.
    -   The names of the tables to be used for text and image embeddings.
    -   The Hugging Face model names for the text embedder, vision model, and reranker.
    -   Parameters for the reranker and retrieval process (e.g., `top_k`, `retrieval_k`).

To change a model, simply update the corresponding model name in this configuration file.

## 5. Usage

To run the system, you first need to ensure the required models are available.

### Prerequisites:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Ollama Models**:
    ```bash
    ollama pull llama3
    ollama pull qwen2.5vl:7b
    ```
3.  **Hugging Face Models**: The `transformers` and `colpali` libraries will automatically download the required models the first time they are used. Ensure you have a stable internet connection.

### Running the System:

1.  **Execute the Main Script**:
    ```bash
    python rag_system/main.py
    ```
2.  **Indexing**: The script will first run the indexing pipeline, processing any documents in the `rag_system/documents` directory and storing their embeddings in LanceDB.
3.  **Querying**: Once indexing is complete, the RAG agent will be ready. You can ask questions about the documents you have indexed.
    ```
    > What was the revenue growth in Q3?
    ```
4.  **Exit**: To stop the agent, type `quit`.
