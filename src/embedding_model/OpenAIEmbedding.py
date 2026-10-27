from typing import List, Optional
from copy import deepcopy
import json

import httpx
import numpy as np
from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig


logger = get_logger(__name__)


def _normalize_openai_base_url(base_url: str) -> str:
    if not base_url:
        return base_url
    normalized = base_url.rstrip("/")
    if normalized.endswith("/embeddings"):
        normalized = normalized[: -len("/embeddings")]
    return normalized


class OpenAIEmbeddingModel(BaseEmbeddingModel):


    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        api_key = self.embedding_config.api_key
        base_url = _normalize_openai_base_url(self.embedding_config.base_url)
        if not base_url:
            raise ValueError(
                "OpenAIEmbeddingModel requires a non-empty embedding base URL. "
                "Set MEMGRAPHRAG_EMBED_BASE_URL or configure embedding_base_url explicitly."
            )
        if not self.embedding_model_name:
            raise ValueError(
                "OpenAIEmbeddingModel requires a non-empty embedding model name. "
                "Set MEMGRAPHRAG_EMBED_MODEL or configure embedding_model_name explicitly."
            )

        client = httpx.Client(
            timeout=httpx.Timeout(120.0, read=120.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            trust_env=False,
        )
        self.embedding_model = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=client,
        )
        self.http_client = client
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_dim = int(getattr(self.global_config, "embedding_dim_hint", 0) or 0)

        logger.info(
            "OpenAIEmbeddingModel initialized: model=%s, base_url=%s",
            self.embedding_model_name,
            base_url,
        )

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "api_key": getattr(self.global_config, "embedding_api_key", "EMPTY"),
            "base_url": _normalize_openai_base_url(
                getattr(self.global_config, "embedding_base_url", "")
            ),
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    @staticmethod
    def _extract_embeddings_from_payload(payload: object) -> List[List[float]]:
        if not isinstance(payload, dict):
            return []

        data = payload.get("data")
        if isinstance(data, list):
            embeddings: List[List[float]] = []
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list):
            if embeddings and all(isinstance(x, (int, float)) for x in embeddings):
                return [embeddings]
            nested = [item for item in embeddings if isinstance(item, list)]
            if nested:
                return nested

        result = payload.get("result")
        if isinstance(result, dict):
            return OpenAIEmbeddingModel._extract_embeddings_from_payload(result)

        return []

    def _request_embeddings_via_http(self, batch_texts: List[str]) -> List[List[float]]:
        endpoint = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.embedding_model_name,
            "input": batch_texts,
        }
        response = self.http_client.post(endpoint, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
        embeddings = self._extract_embeddings_from_payload(payload)
        if embeddings:
            return embeddings
        raise ValueError(
            "No embedding data received from raw HTTP response. "
            f"Top-level keys: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}. "
            f"Response preview: {json.dumps(payload, ensure_ascii=False)[:500]}"
        )

    def _request_embeddings_one_by_one(self, batch_texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in batch_texts:
            single = self._request_embeddings_via_http([text])
            if len(single) != 1:
                raise ValueError(
                    "Expected exactly one embedding from single-text fallback request, "
                    f"but received {len(single)}."
                )
            embeddings.extend(single)
        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = int(params.pop("batch_size", self.global_config.embedding_batch_size or 8))
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings: List[List[float]] = []

            try:
                batch_embeddings = self._request_embeddings_via_http(batch_texts)
                if len(batch_embeddings) != len(batch_texts):
                    logger.warning(
                        "Raw HTTP embedding response count mismatch for model=%s: expected=%d got=%d. "
                        "Retrying one text at a time.",
                        self.embedding_model_name,
                        len(batch_texts),
                        len(batch_embeddings),
                    )
                    batch_embeddings = self._request_embeddings_one_by_one(batch_texts)
            except Exception as http_exc:
                logger.warning(
                    "Raw HTTP embedding request failed for model=%s (%s). "
                    "Trying OpenAI SDK compatibility path.",
                    self.embedding_model_name,
                    http_exc,
                )
                response = self.embedding_model.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch_texts,
                )
                response_data = getattr(response, "data", None)
                if isinstance(response_data, list):
                    batch_embeddings = [
                        item.embedding
                        for item in response_data
                        if hasattr(item, "embedding") and isinstance(item.embedding, list)
                    ]
                if not batch_embeddings:
                    logger.warning(
                        "OpenAI SDK returned no embedding data for model=%s. "
                        "Retrying one text at a time via raw HTTP.",
                        self.embedding_model_name,
                    )
                    batch_embeddings = self._request_embeddings_one_by_one(batch_texts)
            all_embeddings.extend(batch_embeddings)

        if not all_embeddings:
            raise ValueError(
                "No embedding data received after all embedding requests. "
                "Please verify the endpoint response format, credentials, and model name."
            )

        results = np.asarray(all_embeddings, dtype=np.float32)
        if self.embedding_dim <= 0 and results.size > 0:
            self.embedding_dim = int(results.shape[1])

        if self.embedding_config.norm and len(results) > 0:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            results = results / norms

        return results

    def close(self) -> None:
        close_fn = getattr(self.http_client, "close", None)
        if callable(close_fn):
            close_fn()
