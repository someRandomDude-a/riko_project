from __future__ import annotations

from typing import overload, Union, List

from process.llm_scripts.llama_server import (
    call_llm_api,
    stream_chat,
    tokenize,
    summarize,
    summarize_client,
    summarize_model_name,
    SentenceStreamer,
    token_callback_for_streamer,
)


@overload
def get_llm_token_length(text: str) -> int: ...
@overload
def get_llm_token_length(text: List[str]) -> List[int]: ...


def get_llm_token_length(text: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Returns the number of tokens in a given string, or a list of lengths
    for a list of strings. Routes through llama-server's /tokenize.
    """
    if isinstance(text, str):
        return tokenize(text)
    return [tokenize(t) for t in text]


# Explicit re-exports for `from utils import call_llm_api` callers
__all__ = [
    "get_llm_token_length",
    "call_llm_api",
    "stream_chat",
    "tokenize",
    "summarize",
    "summarize_client",
    "summarize_model_name",
    "SentenceStreamer",
    "token_callback_for_streamer",
]
