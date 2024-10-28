import logging
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from habana_frameworks.torch.utils.library_loader import load_habana_module
from optimum.habana.sentence_transformers.modeling_utils import (
    adapt_sentence_transformers_to_gaudi,
)

from constants import EMBEDDING_MODEL_NAME


def load_embeddings():
    """Load HuggingFace Embeddings object onto Gaudi or CPU"""
    load_habana_module()
    if torch.hpu.is_available():
        logging.info("Loading embedding model on hpu")

        adapt_sentence_transformers_to_gaudi()
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "hpu"}
        )
    else:
        logging.info("Loading embedding model on cpu")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"}
        )
    return embeddings


def calculate_similarity(model, response, expected_answer):
    """Calculate similarity between response and expected answer using the model"""
    response_embedding = model.client.encode(response, convert_to_tensor=True).squeeze()
    expected_embedding = model.client.encode(
        expected_answer, convert_to_tensor=True
    ).squeeze()
    similarity_score = torch.nn.functional.cosine_similarity(
        response_embedding, expected_embedding, dim=0
    )
    return similarity_score.item()
