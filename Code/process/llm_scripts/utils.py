from transformers import AutoTokenizer
from process.common.config import char_config
from typing import overload
from openai import OpenAI
from openai.types.responses import Response
tokenizer = AutoTokenizer.from_pretrained(char_config['tokenizer_model'])

@overload
def get_llm_token_length(text: str) -> int: ...

@overload
def get_llm_token_length(text: list[str]) -> list[int]: ...

def get_llm_token_length(text: str | list[str]) -> int | list[int]:
    """Returns the number of tokens in a given string, or an array of lengths for each string in a string array"""
    lengths = tokenizer(text, add_special_tokens = False, return_length=True)["length"]
    if isinstance(text, str):
        return lengths[0]
    return lengths

_client = OpenAI(api_key=char_config['api_key'], base_url=char_config['base_url'])
_MODEL = char_config['model']
_MAX_OUTPUT_TOKENS = char_config['presets']['default']['model_params']['max_output_tokens']
_TEMPERATURE = char_config['presets']['default']['model_params']['temperature']
def call_llm_api(messages) -> Response:
    """Core LLM Call"""
    response = _client.responses.create(
        model=_MODEL,
        input=messages,
        max_output_tokens= _MAX_OUTPUT_TOKENS,
        temperature=_TEMPERATURE,
        stream=False,
        text={
            "format": {
            "type": "text"
            }
        },
        store=False,
    )
    return response
