from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, List, Optional

import requests
from openai import OpenAI

from process.common.config import char_config


# Clients

def _build_clients() -> tuple[OpenAI, Optional[OpenAI]]:
    main = char_config["llm_backend"]
    main_client = OpenAI(api_key=main["api_key"], base_url=main["base_url"])

    summ_cfg = char_config.get("summarizer_backend") or {}
    summ_client: Optional[OpenAI] = None
    if summ_cfg.get("base_url"):
        summ_client = OpenAI(api_key=summ_cfg["api_key"], base_url=summ_cfg["base_url"])
    return main_client, summ_client


_MAIN_CLIENT: OpenAI
_SUMM_CLIENT: Optional[OpenAI]
_MAIN_CLIENT, _SUMM_CLIENT = _build_clients()


def client() -> OpenAI:
    """Return the shared main-chat client."""
    return _MAIN_CLIENT


def summarize_client() -> Optional[OpenAI]:
    """Return the summarizer client (None if summarizer_backend.base_url is empty)."""
    return _SUMM_CLIENT


def model_name() -> str:
    return char_config["llm_backend"]["model"]


def summarize_model_name() -> str:
    return (char_config.get("summarizer_backend") or {}).get("model", "")


# Token counting via llama-server

def _llama_base_url() -> str:
    """Return llama-server root URL (strip /v1 suffix for /tokenize endpoint)."""
    base = char_config["llm_backend"]["base_url"].rstrip("/")
    if base.endswith("/v1"):
        return base[: -3]
    return base


def tokenize(text: str) -> int:
    """
    Count tokens using the running llama-server's /tokenize endpoint.

    Falls back to a 4-chars-per-token estimate if the call fails. The
    estimate is intentionally pessimistic (slightly over) so we never
    blow the context budget accidentally.
    """
    base = _llama_base_url()
    auth = char_config["llm_backend"].get("api_key", "")
    try:
        r = requests.post(
            f"{base}/tokenize",
            json={"content": text},
            headers={"Authorization": f"Bearer {auth}"},
            timeout=10,
        )
        if r.ok:
            ids = r.json().get("tokens", [])
            # llama.cpp returns BOS as the first id; subtract to match real usage.
            return max(0, len(ids) - 1)
    except Exception:
        pass
    return max(1, len(text) // 4)


# Prompt cache key (KV reuse)

def _prompt_cache_key(messages: List[dict]) -> str:
    """
    Hash a stable prefix of the conversation so llama-server can match
    cached KV across turns with shifted conversation histories.

    We hash the system prompt + the first user turn. The system prompt is
    the expensive prefix — it includes the Riko character + MCP prompt +
    memory header. As long as those don't change, llama-server reuses the
    cached KV and only computes the new tail tokens.
    """
    salt = char_config["llm_backend"].get("prompt_cache_key_salt", "riko")
    h = hashlib.sha256(salt.encode("utf-8"))
    for m in messages[:2]:
        content = m.get("content", "")
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()[:32]


def _cache_kwargs() -> dict:
    """Build the prompt-cache kwargs to merge into every API call."""
    cfg = char_config["llm_backend"]
    kw: dict = {}
    if cfg.get("slot_id"):
        kw["id_slot"] = cfg["slot_id"]
    if cfg.get("enable_prompt_cache"):
        # `cache_prompt` is a llama-server extension; ignored by strict OpenAI client
        # but the openai SDK passes extra kwargs through.
        kw["cache_prompt"] = True
    return kw


# Chat - streaming

def _flat_message(m: dict) -> dict:
    """Convert Responses-API-shaped message → Chat-Completions-shaped."""
    role = m.get("role", "user")
    c = m.get("content")
    if isinstance(c, str):
        text = c
    elif isinstance(c, list) and c:
        text = c[0].get("text", "")
    else:
        text = ""
    return {"role": role, "content": text}


def _flatten_messages(messages: List[dict]) -> List[dict]:
    return [_flat_message(m) for m in messages if _flat_message(m).get("content")]


def _params() -> dict:
    p = char_config["presets"]["default"]["model_params"]
    return {
        "max_output_tokens": p["max_output_tokens"],
        "temperature": p["temperature"],
    }


def stream_chat(
    messages: List[dict],
    on_token: callable,
    tools: Optional[List[dict]] = None,
) -> Optional[Any]:
    """
    Stream the assistant reply. `on_token(delta: str)` is called per text delta.
    Returns the final `Response` object if the stream produces one (events
    include a `response.completed`/`response.done` carrying the full response),
    otherwise None.

    Auto-falls back to /v1/chat/completions when `use_chat_completions_fallback`
    is true, since some llama-server builds don't expose /v1/responses.
    """
    cfg = char_config["llm_backend"]
    cache_key = _prompt_cache_key(messages)
    cache_kw = _cache_kwargs()
    cache_kw["prompt_cache_key"] = cache_key

    if cfg.get("use_chat_completions_fallback"):
        return _stream_via_chat_completions(messages, on_token, tools)

    try:
        stream = _MAIN_CLIENT.responses.create(
            model=model_name(),
            input=messages,
            stream=True,
            text={"format": {"type": "text"}},
            tools=tools,
            store=False,
            **_params(),
            **cache_kw,
        )
        final_obj = None
        for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    on_token(delta)
            elif et in ("response.completed", "response.done"):
                final_obj = getattr(event, "response", None) or event
        return final_obj
    except Exception as e:  # noqa: BLE001
        # Some llama.cpp builds reject /v1/responses; fall back gracefully.
        if not cfg.get("use_chat_completions_fallback"):
            return _stream_via_chat_completions(messages, on_token, tools)
        raise


def _stream_via_chat_completions(
    messages: List[dict],
    on_token: callable,
    tools: Optional[List[dict]] = None,
) -> Optional[Any]:
    flat = _flatten_messages(messages)
    cache_kw = _cache_kwargs()
    stream = _MAIN_CLIENT.chat.completions.create(
        model=model_name(),
        messages=flat,
        stream=True,
        tools=tools,
        **_params(),
        **cache_kw,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            delta = None
        if delta:
            on_token(delta)
    return chunks  # caller can still parse for tool_calls


# Chat — non-streaming (used by tool-call loop after first iteration)

def call_llm_api(messages: List[dict], tools: Optional[List[dict]] = None):
    """
    Non-streaming call. Used for tool-call iterations and the summarizer.
    Returns the response object (Responses API shape) or a dict-like for
    Chat Completions.
    """
    cfg = char_config["llm_backend"]
    cache_kw = _cache_kwargs()
    cache_kw["prompt_cache_key"] = _prompt_cache_key(messages)

    if cfg.get("use_chat_completions_fallback"):
        flat = _flatten_messages(messages)
        return _MAIN_CLIENT.chat.completions.create(
            model=model_name(),
            messages=flat,
            tools=tools,
            stream=False,
            **_params(),
            **cache_kw,
        )

    return _MAIN_CLIENT.responses.create(
        model=model_name(),
        input=messages,
        stream=False,
        text={"format": {"type": "text"}},
        tools=tools,
        store=False,
        **_params(),
        **cache_kw,
    )


# Summarizer

def summarize(text: str) -> str:
    """Synchronous summarization via second llama-server. Returns input on failure."""
    cfg = char_config.get("summarizer_backend") or {}
    if not cfg.get("base_url") or _SUMM_CLIENT is None:
        return text
    try:
        prefix = cfg.get("prompt_prefix", "")
        max_in = cfg.get("max_input_tokens", 684)
        # cheap char-based pre-trim
        budget_chars = max_in * 4
        prompt = (prefix + text)[:budget_chars]

        if cfg.get("use_completions", True):
            resp = _SUMM_CLIENT.completions.create(
                model=cfg["model"],
                prompt=prompt,
                max_tokens=cfg.get("max_output_tokens", 200),
                temperature=0.0,
            )
            out = (resp.choices[0].text or "").strip()
        else:
            resp = _SUMM_CLIENT.chat.completions.create(
                model=cfg["model"],
                messages=[{
                    "role": "user",
                    "content": f"Summarize concisely in one first-person sentence:\n\n{text}",
                }],
                max_tokens=cfg.get("max_output_tokens", 200),
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
        return out if out else text
    except Exception:
        return text
