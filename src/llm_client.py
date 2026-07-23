#!/usr/bin/env python3
from __future__ import annotations

"""
Reusable LLM client module
Features:
1. Support OpenAI compatible API calls
2. Support SQLite caching to avoid repeated calls
3. Support parallel batch calls
4. Support JSON format output parsing
"""

import json
import hashlib
import os
import sqlite3
import asyncio
import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, TypeVar
from pathlib import Path
import logging

import httpx
from openai import OpenAI
from filelock import FileLock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
_LOCAL_TEXT_BUNDLES: Dict[str, Any] = {}
_LOCAL_MM_BUNDLES: Dict[str, Any] = {}


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_tuple_style_response(response_text: str) -> Any:
    text = response_text.strip()
    text = text.replace(DEFAULT_COMPLETION_DELIMITER, "")

    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    records = [record.strip() for record in text.split(DEFAULT_RECORD_DELIMITER) if record.strip()]
    for record in records:
        candidate = record.strip()
        if not (candidate.startswith("(") and candidate.endswith(")")):
            continue

        inner = candidate[1:-1].strip()
        parts = [part.strip() for part in inner.split(DEFAULT_TUPLE_DELIMITER)]
        if not parts:
            continue

        record_type = _strip_wrapping_quotes(parts[0]).lower()
        if record_type == "entity" and len(parts) >= 4:
            entities.append(
                {
                    "entity_name": _strip_wrapping_quotes(parts[1]),
                    "entity_type": _strip_wrapping_quotes(parts[2]),
                    "description": _strip_wrapping_quotes(DEFAULT_TUPLE_DELIMITER.join(parts[3:])),
                }
            )
        elif record_type == "relationship" and len(parts) >= 5:
            weight_raw = _strip_wrapping_quotes(parts[-1])
            try:
                weight = float(weight_raw)
            except Exception:
                weight = 1.0
            relationships.append(
                {
                    "src_entity_name": _strip_wrapping_quotes(parts[1]),
                    "tgt_entity_name": _strip_wrapping_quotes(parts[2]),
                    "description": _strip_wrapping_quotes(DEFAULT_TUPLE_DELIMITER.join(parts[3:-1])),
                    "weight": weight,
                }
            )

    if entities or relationships:
        return {"entities": entities, "relationships": relationships}

    raise ValueError("Unable to parse tuple-style response")


@dataclass
class LLMConfig:
    """LLM configuration"""
    model_name: str = ""
    backend: str = "api"
    local_model_path: Optional[str] = None
    api_key: Optional[str] = "openai"
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.0
    seed: int = 42
    cache_dir: str = "./cache"
    max_workers: int = 10
    timeout: float = 300.0
    max_retries: int = 3
    trust_remote_code: bool = True
    device_map: str = "auto"
    
    def __post_init__(self):
        if self.backend == "auto":
            self.backend = "api" if self.base_url else "local"
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.local_model_path is None:
            self.local_model_path = self.model_name


def _decode_data_url_to_image(data_url: str):
    """Decode one `data:image/...;base64,...` URL into a PIL image."""
    if not data_url or not data_url.startswith("data:"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
        if ";base64" not in header:
            return None
        from PIL import Image

        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception:
        return None


def _normalize_message_content(content: Any) -> str:
    """Flatten one chat content payload into plain text for local text-only inference."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                text_parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(str(part.get("text", "")))
            elif part_type == "image_url":
                text_parts.append("[Image attached]")
        return "\n".join(part for part in text_parts if part).strip()
    return str(content)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert mixed chat payloads into plain role/content messages."""
    normalized: List[Dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": _normalize_message_content(message.get("content", "")),
            }
        )
    return normalized


class _LocalModelRuntime:
    """Shared local-model runtime so repeated calls do not reload the model."""

    @staticmethod
    def _text_bundle_key(config: LLMConfig) -> str:
        return f"{config.local_model_path}::{config.device_map}::{config.trust_remote_code}"

    @classmethod
    def _load_text_bundle(cls, config: LLMConfig):
        key = cls._text_bundle_key(config)
        if key in _LOCAL_TEXT_BUNDLES:
            return _LOCAL_TEXT_BUNDLES[key]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.local_model_path,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.local_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype="auto",
            device_map=config.device_map,
        )
        model.eval()
        bundle = (tokenizer, model)
        _LOCAL_TEXT_BUNDLES[key] = bundle
        return bundle

    @classmethod
    def _load_mm_bundle(cls, config: LLMConfig):
        key = cls._text_bundle_key(config)
        if key in _LOCAL_MM_BUNDLES:
            return _LOCAL_MM_BUNDLES[key]

        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.local_model_path,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.local_model_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype="auto",
            device_map=config.device_map,
        )
        model.eval()
        bundle = (processor, model)
        _LOCAL_MM_BUNDLES[key] = bundle
        return bundle

    @staticmethod
    def _manual_prompt(messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = message.get("content", "").strip()
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    @classmethod
    def _generate_text(
        cls,
        config: LLMConfig,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict]:
        import torch

        normalized_messages = _normalize_messages(messages)
        tokenizer, model = cls._load_text_bundle(config)

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = cls._manual_prompt(normalized_messages)

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_token_count = int(encoded["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_token_count:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        metadata = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": int(generated_ids.shape[-1]),
            "finish_reason": "stop",
            "model": config.local_model_path,
            "backend": "local",
        }
        return response_text, metadata

    @classmethod
    def _generate_multimodal(
        cls,
        config: LLMConfig,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict]:
        import torch

        processor, model = cls._load_mm_bundle(config)

        mm_messages: List[Dict[str, Any]] = []
        images = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                mm_messages.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": [{"type": "text", "text": str(content)}],
                    }
                )
                continue

            mm_content: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    mm_content.append({"type": "text", "text": str(part.get("text", ""))})
                elif part_type == "image_url":
                    image_url = (part.get("image_url") or {}).get("url", "")
                    image = _decode_data_url_to_image(str(image_url))
                    if image is not None:
                        images.append(image)
                        mm_content.append({"type": "image"})
            mm_messages.append({"role": str(message.get("role", "user")), "content": mm_content})

        if not images:
            raise ValueError("No decodable local images found for multimodal inference")

        prompt = processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], images=images, return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated_ids = output_ids[0][prompt_token_count:]
        if hasattr(processor, "decode"):
            response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        metadata = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": int(generated_ids.shape[-1]),
            "finish_reason": "stop",
            "model": config.local_model_path,
            "backend": "local_mm",
        }
        return response_text, metadata

    @classmethod
    def infer(
        cls,
        config: LLMConfig,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict]:
        has_image = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in message.get("content", [])
            )
            for message in messages
        )
        if has_image:
            try:
                return cls._generate_multimodal(config, messages, max_tokens, temperature)
            except Exception as exc:
                logger.warning(f"Local multimodal generation failed, falling back to text-only generation: {exc}")
        return cls._generate_text(config, messages, max_tokens, temperature)


class LLMClient:
    """LLM client, supports caching and parallel calls"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client
        
        Args:
            config: LLM configuration, uses default configuration if None
        """
        self.config = config or LLMConfig()
        
        # Create cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file path
        safe_model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        self.cache_file = self.cache_dir / f"{safe_model_name}_cache.sqlite"
        self.lock_file = str(self.cache_file) + ".lock"
        
        # Initialize cache database
        self._init_cache_db()
        
        self.openai_client = None
        if self.config.backend == "api":
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(self.config.timeout, read=self.config.timeout),
                trust_env=False,
            )

            self.openai_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                http_client=client
            )
        
        logger.info(
            f"LLM client initialization completed: model={self.config.model_name}, backend={self.config.backend}"
        )
    
    def _init_cache_db(self):
        """Initialize cache database"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
    
    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate cache key"""
        key_data = {
            "messages": messages,
            "model": self.config.model_name,
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[tuple]:
        """Get result from cache"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute("SELECT response, metadata FROM cache WHERE key = ?", (key,))
            row = c.fetchone()
            conn.close()
            if row:
                return row[0], json.loads(row[1])
            return None
    
    def _save_to_cache(self, key: str, response: str, metadata: dict):
        """Save result to cache"""
        with FileLock(self.lock_file):
            conn = sqlite3.connect(str(self.cache_file))
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO cache (key, response, metadata) VALUES (?, ?, ?)",
                (key, response, json.dumps(metadata))
            )
            conn.commit()
            conn.close()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> tuple[str, dict, bool]:
        """
        Send chat request
        
        Args:
            messages: Chat message list, format: [{"role": "user", "content": "..."}]
            use_cache: Whether to use cache
            **kwargs: Other parameters, like temperature, max_tokens, etc.
            
        Returns:
            (response_text, metadata, cache_hit)
        """
        # Check cache
        cache_key = self._get_cache_key(messages, **kwargs)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key[:16]}...")
                return cached[0], cached[1], True
        
        # Call API
        effective_max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        effective_temperature = kwargs.get("temperature", self.config.temperature)

        if self.config.backend == "local":
            response_text, metadata = _LocalModelRuntime.infer(
                self.config,
                messages,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )
            if use_cache:
                self._save_to_cache(cache_key, response_text, metadata)
            return response_text, metadata, False

        params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temperature,
            "seed": kwargs.get("seed", self.config.seed),
        }

        base_url = (self.config.base_url or "").lower()
        if self.config.backend == "api" and self.config.base_url:
            params["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }
        if "vveai.com" in base_url:
            params["max_tokens"] = min(int(params["max_tokens"]), 8192)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.openai_client.chat.completions.create(**params)
                response_text = response.choices[0].message.content
                
                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.config.model_name,
                }
                
                # Save to cache
                if use_cache:
                    self._save_to_cache(cache_key, response_text, metadata)
                
                return response_text, metadata, False
                
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("API call failed")
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> tuple[Any, dict, bool]:
        """
        Send chat request and parse JSON response
        
        Args:
            messages: Chat message list
            use_cache: Whether to use cache
            **kwargs: Other parameters
            
        Returns:
            (parsed_json, metadata, cache_hit)
        """
        if self.config.backend == "api":
            cache_key = self._get_cache_key(messages, **kwargs)
            if use_cache:
                cached = self._get_from_cache(cache_key)
                if cached:
                    logger.debug(f"Cache hit: {cache_key[:16]}...")
                    try:
                        return json.loads(cached[0]), cached[1], True
                    except Exception:
                        pass

            params = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "seed": kwargs.get("seed", self.config.seed),
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
            }
            base_url = (self.config.base_url or "").lower()
            if "vveai.com" in base_url:
                params["max_tokens"] = min(int(params["max_tokens"]), 8192)

            response = self.openai_client.chat.completions.create(**params)
            response_text = response.choices[0].message.content
            metadata = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "model": self.config.model_name,
            }
            if use_cache:
                self._save_to_cache(cache_key, response_text, metadata)
        else:
            response_text, metadata, cache_hit = self.chat(messages, use_cache, **kwargs)
            # Try to parse JSON
            try:
                # Handle possible markdown code blocks
                text = response_text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]

                parsed = json.loads(text.strip())
                return parsed, metadata, cache_hit
            except json.JSONDecodeError as e:
                try:
                    parsed = _parse_tuple_style_response(response_text)
                    return parsed, metadata, cache_hit
                except Exception:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"Original response: {response_text[:500]}...")
                    raise ValueError(f"Unable to parse JSON response: {e}")

        # Try to parse JSON
        try:
            # Handle possible markdown code blocks
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            parsed = json.loads(text.strip())
            return parsed, metadata, False
        except json.JSONDecodeError as e:
            try:
                parsed = _parse_tuple_style_response(response_text)
                return parsed, metadata, False
            except Exception:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Original response: {response_text[:500]}...")
                raise ValueError(f"Unable to parse JSON response: {e}")
    
    def batch_chat(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[str, dict, bool]]:
        """
        Parallel batch chat requests
        
        Args:
            batch_messages: Batch message list
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            **kwargs: Other parameters
            
        Returns:
            Result list, each element: (response_text, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"Batch processing index {idx} failed: {e}")
                return idx, (None, {"error": str(e)}, False)
        
        completed = 0
        total = len(batch_messages)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_one, i, msgs): i 
                for i, msgs in enumerate(batch_messages)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if show_progress and completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"Batch processing completed: {total} requests")
        
        return results
    
    def batch_chat_json(
        self,
        batch_messages: List[List[Dict[str, str]]],
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[tuple[Any, dict, bool]]:
        """
        Parallel batch chat requests with JSON parsing
        
        Args:
            batch_messages: Batch message list
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            **kwargs: Other parameters
            
        Returns:
            Result list, each element: (parsed_json, metadata, cache_hit)
        """
        results = [None] * len(batch_messages)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> tuple[int, Any]:
            try:
                result = self.chat_json(messages, use_cache, **kwargs)
                return idx, result
            except Exception as e:
                logger.error(f"Batch processing index {idx} failed: {e}")
                return idx, (None, {"error": str(e)}, False)
        
        completed = 0
        total = len(batch_messages)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_one, i, msgs): i 
                for i, msgs in enumerate(batch_messages)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if show_progress and completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
        
        if show_progress:
            logger.info(f"Batch processing completed: {total} requests")
        
        return results


# Convenience functions
_default_client: Optional[LLMClient] = None


def get_default_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Get default LLM client (singleton pattern)"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(config)
    return _default_client


def chat(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[str, dict, bool]:
    """Convenience function: send chat request"""
    client = get_default_client(config)
    return client.chat(messages, **kwargs)


def chat_json(
    messages: List[Dict[str, str]],
    config: Optional[LLMConfig] = None,
    **kwargs
) -> tuple[Any, dict, bool]:
    """Convenience function: send chat request and parse JSON"""
    client = get_default_client(config)
    return client.chat_json(messages, **kwargs)


if __name__ == "__main__":
    # Test example
    config = LLMConfig(
        model_name="qwen3-8b",
        temperature=0.0,
    )
    
    client = LLMClient(config)
    
    messages = [
        {"role": "user", "content": "Say 'Hello, World!' in JSON format: {\"greeting\": \"...\"}"}
    ]
    
    response, metadata, cache_hit = client.chat_json(messages)
    print(f"Response: {response}")
    print(f"Metadata: {metadata}")
    print(f"Cache hit: {cache_hit}")
