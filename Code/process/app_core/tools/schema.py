"""JSON schemas for local tools, read from the signature of the function that runs them."""
from __future__ import annotations

import inspect
import types
from typing import Any, get_args, get_origin, get_type_hints


def schema_for(annotation):
    if get_origin(annotation) is types.UnionType:
        return {'anyOf': [schema_for(value) if value is not type(None) else {'type': 'null'} for value in get_args(annotation)]}
    origin = getattr(annotation, "__origin__", None)
    if annotation in (int,): return {"type": "integer"}
    if annotation in (float,): return {"type": "number"}
    if annotation in (bool,): return {"type": "boolean"}
    if annotation in (list,): return {"type": "array"}
    if annotation in (dict,): return {"type": "object"}
    if origin is list: return {"type": "array"}
    return {"type": "string"}


def signature_schema(function, choices=None) -> dict[str, Any]:
    """Each parameter of function (a bound method: no self) is a property, required unless it has a default; choices
    adds an enum to the parameters it names."""
    signature, annotations = inspect.signature(function), get_type_hints(function)
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {**schema_for(annotations.get(name, parameter.annotation)), "description": f"Parameter: {name}"}
        if parameter.default is inspect.Parameter.empty: required.append(name)
    schema = {"type": "object", "properties": properties, "required": required}
    for key, options in (choices or {}).items():
        if key in properties: properties[key]['enum'] = list(options)
    return schema


def local_definition(tool) -> dict[str, Any]:
    """The MCP-style definition of a local tool object (tools.tool.local_tool), as the registry once exposed it."""
    from .tool import local_tool
    entry = local_tool(tool)
    return {"name": entry.name, "description": entry.description, "inputSchema": entry.schema}
