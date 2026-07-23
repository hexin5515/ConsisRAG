from .base import BaseEmbeddingModel, EmbeddingConfig


def _get_embedding_model_class(embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    normalized_name = embedding_model_name.lower()
    if embedding_model_name in {"openai-embedding", "openai-compatible-embedding", "api-embedding"} or (
        normalized_name.startswith("text-embedding")
        or normalized_name.startswith("embedding-")
        or normalized_name.endswith("-embedding")
        or "qwen3-embedding" in normalized_name
        or "qwen-embedding" in normalized_name
    ):
        from .OpenAIEmbedding import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel
    raise ValueError(
        "This release keeps only the OpenAI-compatible embedding path used by "
        f"HVM-GraphRAG. Unsupported embedding model: {embedding_model_name}"
    )
