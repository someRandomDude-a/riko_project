from __future__ import annotations

import json
import threading
from copy import deepcopy
from typing import Iterable, Sequence

from ..conversation.messages import ChatMessage, ModelResponse, ToolCall
from ..conversation.streaming import WordDeltas
from .responses import response_input, response_tools


class OpenAIProvider:
    """Adapter for OpenAI-compatible Chat Completions and Responses servers."""
    def __init__(self, config):
        from openai import OpenAI
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url,
            timeout=getattr(config, 'request_timeout_seconds', 120), max_retries=1)
        self.api_mode = getattr(config, 'api_mode', 'auto')
        if self.api_mode not in {'auto', 'responses', 'chat_completions'}: raise ValueError('Invalid runtime.api_mode')
        self.responses_enabled = self.api_mode == 'responses' or (self.api_mode == 'auto' and config.provider in {'lm_studio', 'openai'})
        self.response_cache = []
        self.cache_lock = threading.Lock()

    def generate(self, messages: Sequence[ChatMessage], *, tools=None, **options) -> ModelResponse:
        messages = self._pack(messages, tools, options)
        if self.responses_enabled:
            try:
                response = self._responses(messages, tools=tools, **options)
                response.context_messages = messages
                return response
            except Exception as exc:
                if self.api_mode != 'auto' or getattr(exc, 'status_code', None) not in {404, 405, 501}: raise
                self.responses_enabled = False
        kwargs = dict(model=self.config.model, messages=[m.as_dict() for m in messages],
                      temperature=options.get("temperature", self.config.temperature),
                      max_tokens=options.get("max_output_tokens", self.config.max_output_tokens))
        if tools:
            kwargs["tools"] = tools
        if options.get("on_delta"):
            stream = self.client.chat.completions.create(**kwargs, stream=True)
            try:
                return assemble_stream((chunk.model_dump() for chunk in stream), options["on_delta"], on_reasoning=options.get('on_reasoning'))
            finally:
                stream.close()
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        calls = []
        for call in (getattr(msg, "tool_calls", None) or []):
            args = call.function.arguments
            calls.append(ToolCall(call.id, call.function.name, json.loads(args) if isinstance(args, str) else args))
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
        return ModelResponse(ChatMessage("assistant", msg.content or "", tool_calls=calls), choice.finish_reason, usage_dict, response, messages)

    def _pack(self, messages, tools, options):
        from .context_budget import pack_context, estimate_tokens
        return pack_context(messages, lambda value: estimate_tokens(value, tools),
            min(options.get('context_limit', self.config.n_ctx), self.config.n_ctx),
            options.get('max_output_tokens', self.config.max_output_tokens), cancelled=options.get('cancelled', lambda: False))

    def _responses(self, messages, *, tools=None, **options):
        inputs = response_input(messages)
        definitions = response_tools(tools)
        signature = json.dumps({'model': self.config.model, 'tools': definitions}, sort_keys=True)
        previous, prefix = None, []
        if getattr(self.config, 'reuse_response_ids', True):
            with self.cache_lock:
                candidates = [(response_id, history) for key, response_id, history in self.response_cache
                              if key == signature and len(inputs) > len(history) and inputs[:len(history)] == history]
                if candidates: previous, prefix = max(candidates, key=lambda item: len(item[1]))
        kwargs = dict(model=self.config.model, input=inputs[len(prefix):],
                      temperature=options.get('temperature', self.config.temperature),
                      max_output_tokens=options.get('max_output_tokens', self.config.max_output_tokens))
        if definitions: kwargs['tools'] = definitions
        if previous: kwargs['previous_response_id'] = previous
        callback = options.get('on_delta')
        words = WordDeltas(callback) if callback else None
        response = None
        try:
            try:
                result = self.client.responses.create(**kwargs, stream=bool(callback))
            except Exception as exc:
                if not previous or getattr(exc, 'status_code', None) not in {400, 404}: raise
                with self.cache_lock:
                    self.response_cache = [entry for entry in self.response_cache if entry[1] != previous]
                kwargs.pop('previous_response_id')
                kwargs['input'] = inputs
                result = self.client.responses.create(**kwargs, stream=bool(callback))
            if callback:
                try:
                    for event in result:
                        if event.type == 'response.output_text.delta': words.feed(event.delta)
                        elif event.type == 'response.reasoning_summary_text.delta' and options.get('on_reasoning'): options['on_reasoning'](event.delta)
                        elif event.type == 'response.completed': response = event.response
                        elif event.type in {'response.failed', 'error'}: raise RuntimeError(str(event))
                finally: result.close()
                if response is None: raise RuntimeError('Response stream ended without completion')
                words.finish()
            else: response = result
            raw = response.model_dump()
            text, calls = [], []
            for output in raw.get('output', []):
                if output['type'] == 'message':
                    text.extend(part['text'] for part in output.get('content', []) if part['type'] == 'output_text')
                elif output['type'] == 'function_call':
                    arguments = json.loads(output['arguments'])
                    if not isinstance(arguments, dict): raise ValueError('Tool arguments must be an object')
                    calls.append(ToolCall(output['call_id'], output['name'], arguments))
            message = ChatMessage('assistant', ''.join(text), tool_calls=calls)
            # Keep no cancelled/partial responses. Exact history comparison prevents
            # cross-talk between reflection, live conversation and rewritten turns.
            if raw.get('id') and raw.get('status', 'completed') == 'completed':
                with self.cache_lock:
                    self.response_cache.append((signature, raw['id'], deepcopy(inputs + response_input([message]))))
                    self.response_cache = self.response_cache[-8:]
            return ModelResponse(message, 'tool_calls' if calls else 'stop', raw.get('usage') or {}, raw)
        except Exception:
            # Never flush an unfinished word on cancellation or transport failure.
            raise

    def stream(self, messages, *, tools=None, **options) -> Iterable[str]:
        messages = self._pack(messages, tools, options)
        if self.responses_enabled:
            yield from self._stream_responses(messages, tools=tools, **options)
            return
        kwargs = dict(model=self.config.model, messages=[m.as_dict() for m in messages], stream=True,
                      temperature=options.get("temperature", self.config.temperature),
                      max_tokens=options.get("max_output_tokens", self.config.max_output_tokens))
        if tools: kwargs["tools"] = tools
        pending = []
        words = WordDeltas(pending.append)
        stream = self.client.chat.completions.create(**kwargs)
        try:
            for chunk in stream:
                if not chunk.choices: continue
                text = getattr(chunk.choices[0].delta, "content", None)
                if text: words.feed(text)
                while pending: yield pending.pop(0)
            words.finish()
            yield from pending
        finally: stream.close()

    def _stream_responses(self, messages, *, tools=None, **options):
        from .responses import response_input, response_tools
        kwargs = dict(model=self.config.model, input=response_input(messages), stream=True,
            temperature=options.get('temperature', self.config.temperature),
            max_output_tokens=options.get('max_output_tokens', self.config.max_output_tokens))
        if tools: kwargs['tools'] = response_tools(tools)
        try: stream = self.client.responses.create(**kwargs)
        except Exception as exc:
            if self.api_mode != 'auto' or getattr(exc, 'status_code', None) not in {404, 405, 501}: raise
            self.responses_enabled = False
            yield from self.stream(messages, tools=tools, **options)
            return
        pending, completed = [], False
        words = WordDeltas(pending.append)
        try:
            for event in stream:
                if event.type == 'response.output_text.delta': words.feed(event.delta)
                elif event.type in {'response.reasoning_summary_text.delta', 'response.reasoning_text.delta'}:
                    if options.get('on_reasoning'): options['on_reasoning'](event.delta)
                elif event.type in {'response.completed', 'response.incomplete'}:
                    completed = True
                elif event.type in {'response.failed', 'response.cancelled', 'error'}: raise RuntimeError(str(event))
                while pending: yield pending.pop(0)
                if completed: break
            if not completed: raise RuntimeError('Response stream ended without completion')
            words.finish()
            yield from pending
        finally: stream.close()

    def count_tokens(self, messages):
        from .context_budget import estimate_tokens
        return estimate_tokens(messages)

    def count_text_tokens(self, text):
        return len(text.encode('utf-8')) # Conservative fallback, not an exact tokenizer.

    def close(self):
        close = getattr(self.client, "close", None)
        if close: close()


def assemble_stream(chunks, on_delta, *, on_reasoning=None):
    """Assemble fragmented tool calls while forwarding text immediately."""
    text, calls, finish, usage = [], {}, None, {}
    words = WordDeltas(on_delta)
    for chunk in chunks:
        if chunk.get("usage"): usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices: continue
        choice = choices[0]
        finish = choice.get("finish_reason") or finish
        delta = choice.get("delta") or {}
        reasoning = delta.get('reasoning_content') or delta.get('reasoning')
        if on_reasoning and isinstance(reasoning, str): on_reasoning(reasoning)
        if delta.get("content"):
            text.append(delta["content"])
            words.feed(delta["content"])
        for part in delta.get("tool_calls") or []:
            call = calls.setdefault(part["index"], {"id": "", "name": "", "arguments": ""})
            if part.get("id"): call["id"] = part["id"]
            function = part.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
    tools = []
    for index in sorted(calls):
        call = calls[index]
        arguments = json.loads(call["arguments"] or "{}")
        if not isinstance(arguments, dict): raise ValueError("Tool arguments must be an object")
        tools.append(ToolCall(call["id"], call["name"], arguments))
    words.finish()
    return ModelResponse(ChatMessage("assistant", "".join(text), tool_calls=tools), finish, usage)


def create_provider(config):
    provider = config.provider.lower().replace("-", "_")
    if provider == "llama_cpp":
        from .llama_native import InProcessLlamaProvider
        return InProcessLlamaProvider(config)
    if provider in {"openai", "lm_studio", "openai_compatible", "ollama", "local_http"}:
        return OpenAIProvider(config)
    raise ValueError(
        f"Unsupported provider '{config.provider}'. Use llama_cpp, openai, lm_studio, "
        "openai_compatible, ollama, or local_http."
    )
