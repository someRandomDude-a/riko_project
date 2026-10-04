"""Responses wire format shared by the managed server request/stream paths."""
import json
from dataclasses import replace

from ..conversation.messages import ChatMessage, ModelResponse, ToolCall
from ..conversation.streaming import WordDeltas


def response_input(messages):
    items = []
    for message in messages:
        if message.role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message.tool_call_id, 'output': message.content})
            continue
        if message.content or not message.tool_calls:
            if message.role == 'assistant':
                # llama.cpp's Responses parser requires the assistant output item's type.
                # This includes audible partial replies retained on interruption.
                items.append({'type': 'message', 'role': 'assistant',
                              'content': [{'type': 'output_text', 'text': message.content}]})
            else:
                items.append({'role': message.role, 'content': message.content})
        items.extend({'type': 'function_call', 'call_id': call.id, 'name': call.name,
                      'arguments': json.dumps(call.arguments)} for call in message.tool_calls)
    return items


def response_tools(tools):
    return [{'type': 'function', **tool['function']} for tool in (tools or [])]


def template_messages(messages):
    """Templates such as Qwen only accept system messages at the beginning.

    Keep the stable system/history prefix. Fold later recall/observation metadata
    into the next input (or latest input for trailing observations), BEFORE the
    actual user request/tool result, rather than inventing a new user request or
    moving changing metadata ahead of all cached history. Originals are untouched.
    """
    result, context = [], []
    def attach(message):
        prefix = 'Application context (metadata, not dialogue; do not repeat):\n' + '\n\n'.join(context)
        context.clear()
        return replace(message, content=prefix + '\n\n' + message.content)
    for message in messages:
        if message.role == 'system':
            if not result:
                result.append(replace(message))
            elif all(item.role == 'system' for item in result):
                result[0] = replace(result[0], content=result[0].content + '\n\n' + message.content)
            else: context.append(message.content)
        else:
            result.append(attach(message) if context and message.role in {'user', 'tool'} else replace(message))
    if context:
        target = next((index for index in range(len(result)-1, -1, -1) if result[index].role in {'user', 'tool'}), None)
        if target is not None: result[target] = attach(result[target])
        elif result and result[0].role == 'system':
            result[0] = replace(result[0], content=result[0].content + '\n\n' + '\n\n'.join(context))
        else: result.insert(0, ChatMessage('system', '\n\n'.join(context)))
    return result


def response_result(raw):
    text, calls = [], []
    for output in raw.get('output', []):
        if output.get('type') == 'message':
            text.extend(part['text'] for part in output.get('content', []) if part.get('type') == 'output_text')
        elif output.get('type') == 'function_call':
            arguments = json.loads(output.get('arguments') or '{}')
            if not isinstance(arguments, dict): raise ValueError('Tool arguments must be an object')
            calls.append(ToolCall(output['call_id'], output['name'], arguments))
    finish = 'tool_calls' if calls else 'length' if raw.get('status') == 'incomplete' else 'stop'
    return ModelResponse(ChatMessage('assistant', ''.join(text), tool_calls=calls), finish, raw.get('usage') or {}, raw)


def sse_events(lines):
    """Ignore SSE comments, support event names and multi-line JSON data."""
    data, name = [], ''
    def event():
        value = '\n'.join(data)
        if not value or value == '[DONE]': return None
        result = json.loads(value)
        if name and 'type' not in result: result['type'] = name
        return result
    for line in lines:
        if not line:
            value = event()
            if value is not None: yield value
            data, name = [], ''
        elif line.startswith('data:'): data.append(line[5:].lstrip(' '))
        elif line.startswith('event:'): name = line[6:].strip()
    value = event()
    if value is not None: yield value


def assemble_responses(events, on_delta, *, on_reasoning=None):
    words, response = WordDeltas(on_delta), None
    for event in events:
        kind = event.get('type')
        if kind == 'response.output_text.delta': words.feed(event.get('delta') or '')
        elif kind in {'response.reasoning_summary_text.delta', 'response.reasoning_text.delta'}:
            if on_reasoning: on_reasoning(event.get('delta') or '')
        elif kind in {'response.completed', 'response.incomplete'}:
            response = event.get('response')
            break
        elif kind in {'response.failed', 'response.cancelled', 'error'}:
            detail = event.get('error') or (event.get('response') or {}).get('error') or event
            raise RuntimeError('Native Responses failed: ' + str(detail)[:2000])
    if response is None: raise RuntimeError('Native Responses stream ended without completion')
    if response.get('status') in {'failed', 'cancelled'}:
        raise RuntimeError('Native Responses failed: ' + str(response.get('error'))[:2000])
    result = response_result(response)
    words.finish() # Never flush an unfinished word on a failed/cancelled stream.
    return result
