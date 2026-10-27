import json
import re
from string import Template
from typing import (
    Optional,
    Union,
    List,
    TypedDict
)

from openai import (
    APIConnectionError,
    RateLimitError,
    Timeout
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config_utils import BaseConfig


class TextChatMessage(TypedDict):

    role: str
    content: Union[str, Template]


def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:


    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}


    placeholder_pattern = re.compile(r'\{(\w+)\}')


    def replace_placeholder(match):
        original_placeholder = match.group(1)


        if original_placeholder in static_values:
            return str(static_values[original_placeholder])


        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'


    template_string = placeholder_pattern.sub(replace_placeholder, original_string)

    return template_string


def safe_unicode_decode(content: Union[bytes, str]) -> str:


    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")


    def replace_unicode_escape(match: re.Match) -> str:

        return chr(int(match.group(1), 16))


    if isinstance(content, bytes):
        content = content.decode("utf-8")
    elif not isinstance(content, str):
        raise AttributeError("Input must be of type bytes or str.")


    decoded_content = unicode_escape_pattern.sub(replace_unicode_escape, content)

    return decoded_content


def dynamic_retry(experiment_config: BaseConfig):


    stop = stop_after_attempt(experiment_config.async_max_retry_attempts)
    wait = wait_exponential(
        multiplier=experiment_config.async_retry_wait_exp_multiplier,
        min=experiment_config.async_retry_min_wait_exp_time,
        max=experiment_config.async_retry_max_wait_exp_time,
    )

    return retry(
        stop=stop,
        wait=wait,
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
    )


def fix_broken_generated_json(json_str: str) -> str:


    def find_unclosed(json_str):


        unclosed = []
        inside_string = False
        escape_next = False

        for char in json_str:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in '{[':
                    unclosed.append(char)
                elif char in '}]':
                    if unclosed and ((char == '}' and unclosed[-1] == '{') or (char == ']' and unclosed[-1] == '[')):
                        unclosed.pop()

        return unclosed

    try:

        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        pass


    last_comma_index = json_str.rfind(',')
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]


    unclosed_elements = find_unclosed(json_str)


    closing_map = {'{': '}', '[': ']'}
    for open_char in reversed(unclosed_elements):
        json_str += closing_map[open_char]

    return json_str


def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:


    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples


PROMPT_JSON_TEMPLATE = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 0
            }
        },
        "required": ["named_entities"]
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["triples"]
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["fact"]
    },
    "json": {
        "type": "object",
    },
    "qa_cot": {
        "type": "object",
        "required": ["Thought", "Answer"],
        "properties": {
            "Thought": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000
            },
            "Answer": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200
            },
        },
    }
}


def num_tokens_by_tiktoken(text: str):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    return len(enc.encode(text))
