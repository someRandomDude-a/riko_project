from transformers import AutoTokenizer
from process.common.config import char_config
from typing import overload

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

