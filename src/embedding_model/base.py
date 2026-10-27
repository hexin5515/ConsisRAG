import json
from dataclasses import dataclass, field, asdict
from typing import (
    Optional,
    Tuple,
    Any,
    Dict,
    List
)
import numpy as np
import threading
import multiprocessing


from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig


logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:

        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        if key in self._data:
            return self._data[key]

        logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


    def __setattr__(self, key: str, value: Any) -> None:
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:

        if key in self._data:
            return self._data[key]
        logger.error(f"'{key}' not found in configuration.")
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:

        self._data[key] = value

    def __delitem__(self, key: str) -> None:

        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{key}' not found in configuration.")
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:

        return key in self._data


    def batch_upsert(self, updates: Dict[str, Any]) -> None:

        self._data.update(updates)

    def to_dict(self) -> Dict[str, Any]:

        return self._data

    def to_json(self) -> str:

        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":

        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> "LLMConfig":

        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:

        return json.dumps(self._data, indent=4)


from filelock import FileLock
import sqlite3
import hashlib
import os
import torch
def make_cache_embed(encode_func, cache_file_name, device):
    def wrapper(**kwargs):

        instruction = kwargs.get("instruction", "")
        max_length = kwargs.get("max_length", "")

        hash_strs = []
        for prompt in kwargs['prompts']:
            key_str = json.dumps({
                "instruction": instruction,
                "promps": prompt,
                "max_length": max_length
            }, sort_keys=True, default=str)
            hash_strs.append(hashlib.sha256(key_str.encode("utf-8")).hexdigest())

        missed_prompts = []
        lock_file = cache_file_name + ".lock"

        with FileLock(lock_file):
            with sqlite3.connect(cache_file_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        hash TEXT PRIMARY KEY,
                        embedding BLOB
                    )
                ''')
                conn.commit()

                embeddings = []
                for i, hash_str in enumerate(hash_strs):
                    cursor.execute('SELECT embedding FROM embeddings WHERE hash = ?', (hash_str,))
                    result = cursor.fetchone()
                    if result:

                        emb = np.frombuffer(result[0], dtype=np.float32)
                        embeddings.append(emb)
                    else:
                        missed_prompts.append(i)
                        embeddings.append(None)

        if missed_prompts:

            kwargs['prompts'] = [kwargs['prompts'][i] for i in missed_prompts]

            new_embeddings = encode_func(**kwargs)

            for idx, embedding in enumerate(new_embeddings):
                embeddings[missed_prompts[idx]] = embedding


            with FileLock(lock_file):
                with sqlite3.connect(cache_file_name) as conn:
                    cursor = conn.cursor()
                    for i in missed_prompts:
                        hash_str = hash_strs[i]
                        emb = embeddings[i]

                        if isinstance(emb, torch.Tensor):
                            emb_bytes = emb.cpu().numpy().tobytes()
                        else:
                            emb_bytes = emb.tobytes()
                        cursor.execute('INSERT INTO embeddings (hash, embedding) VALUES (?, ?)', (hash_str, emb_bytes))
                    conn.commit()


        final_embeddings = [
            emb if isinstance(emb, torch.Tensor) else torch.Tensor(emb.copy())
            for emb in embeddings
        ]

        final_embeddings = [emb.to(device) for emb in final_embeddings]

        return torch.stack(final_embeddings)

    return wrapper

class BaseEmbeddingModel:
    global_config: BaseConfig
    embedding_model_name: str
    embedding_config: EmbeddingConfig

    embedding_dim: int

    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None:
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else: self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")


        self.embedding_model_name = self.global_config.embedding_model_name

        logger.debug(f"Init {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        raise NotImplementedError


    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):


        return np.dot(query_vec, doc_vecs.T)


class EmbeddingCache:


    _manager = multiprocessing.Manager()
    _cache = _manager.dict()
    _lock = threading.Lock()

    @classmethod
    def get(cls, content):

        return cls._cache.get(content)

    @classmethod
    def set(cls, content, embedding):

        with cls._lock:
            cls._cache[content] = embedding

    @classmethod
    def contains(cls, content):

        return content in cls._cache

    @classmethod
    def clear(cls):

        with cls._lock:
            cls._cache.clear()
